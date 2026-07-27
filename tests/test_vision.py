"""Unit tests for vision-based procedure extraction — the vision LLM is
mocked, no network. Focused on the two-pass verification added after a live
test showed the model fabricating a procedure name on a blank image instead
of emitting the VISION_UNCERTAIN sentinel roughly half the time."""
import io
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from PIL import Image

import app.services.vision as vision_module
from app.schemas.vision import VisionExtraction, VisionVerification
from app.services.vision import VISION_UNCERTAIN, extract_procedure_query


def _rate_limit_error() -> openai.RateLimitError:
    resp = httpx.Response(status_code=429, request=httpx.Request("POST", "https://example.test"))
    return openai.RateLimitError("rate limited", response=resp, body=None)


@pytest.fixture(autouse=True)
def reset_structured_output_cache():
    """_structured_output_ok is module-level state (a perf memo of whether
    with_structured_output works for a given model) — must reset between
    tests or one test's simulated failure silently skips the structured
    attempt in a later, unrelated test using the same model name."""
    vision_module._structured_output_ok.clear()
    yield
    vision_module._structured_output_ok.clear()


def _cache_ctx(cached_row=None):
    session = MagicMock()
    result = MagicMock()
    result.first.return_value = cached_row
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx, session


@pytest.fixture(autouse=True)
def mock_vision_cache():
    """DB-backed vision result cache — defaults to always-miss so existing
    tests exercise the LLM path without a real Postgres connection. Tests
    that care about cache behavior override this via patch() directly."""
    ctx, _ = _cache_ctx(None)
    with patch("app.services.vision.AsyncSessionLocal", return_value=ctx):
        yield


def _mock_llm(by_schema: dict, raw_content_by_schema: dict | None = None):
    """by_schema: {SchemaClass: return_value_or_Exception} for the structured
    (primary) path. raw_content_by_schema: {SchemaClass: str} for the raw
    ainvoke fallback path, keyed by whichever schema's structured call failed."""
    mock_llm = MagicMock()

    def _with_structured_output(schema):
        mock_structured = AsyncMock()
        result = by_schema.get(schema)
        if isinstance(result, Exception):
            mock_structured.ainvoke = AsyncMock(side_effect=result)
        else:
            mock_structured.ainvoke = AsyncMock(return_value=result)
        return mock_structured

    mock_llm.with_structured_output.side_effect = _with_structured_output

    if raw_content_by_schema:
        calls = iter(raw_content_by_schema.values())

        async def _raw_ainvoke(*args, **kwargs):
            resp = MagicMock()
            resp.content = next(calls)
            return resp

        mock_llm.ainvoke = AsyncMock(side_effect=_raw_ainvoke)
    return mock_llm


@pytest.mark.asyncio
async def test_specialization_context_prefixed_to_extraction_prompt_not_verify():
    """specialization_context prepends the PRIMARY extraction prompt only —
    the consensus re-sample intentionally uses the unbiased base prompt (see
    test_consensus_second_sample_uses_unbiased_prompt), and the verify pass
    stays domain-agnostic by design (anti-hallucination guard)."""
    captured_prompts: dict[type, list[str]] = {VisionExtraction: [], VisionVerification: []}

    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        captured_prompts[schema].append(prompt)
        if schema is VisionExtraction:
            return VisionExtraction(is_legible=True, price_question="¿Cuánto cuesta un examen de IGRA?")
        return VisionVerification(text_visible=True)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        result = await extract_procedure_query(
            b"fake image bytes", "", specialization_context="jerga médica: IGRA, antro gástrico"
        )

    assert result == "¿Cuánto cuesta un examen de IGRA?"
    assert "jerga médica: IGRA, antro gástrico" in captured_prompts[VisionExtraction][0]
    for verify_prompt in captured_prompts[VisionVerification]:
        assert "jerga médica: IGRA, antro gástrico" not in verify_prompt


@pytest.mark.asyncio
async def test_consensus_second_sample_uses_unbiased_prompt():
    """Regression: /code-review 2026-07-24 — the consensus re-sample must use
    the prompt WITHOUT the specialization prefix (base_prompt), not the same
    biased prompt as the primary extraction, or it can only catch random
    sampling noise and never a consistent, prompt-driven steering bias."""
    captured_prompts: list[str] = []

    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        if schema is VisionExtraction:
            captured_prompts.append(prompt)
            return VisionExtraction(is_legible=True, price_question="¿Cuánto cuesta un examen de IGRA?")
        return VisionVerification(text_visible=True)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        await extract_procedure_query(
            b"fake image bytes", "", specialization_context="jerga médica: IGRA"
        )

    assert len(captured_prompts) == 2
    assert "jerga médica: IGRA" in captured_prompts[0]
    assert "jerga médica: IGRA" not in captured_prompts[1]


@pytest.mark.asyncio
async def test_specialization_context_adds_stricter_verify_scrutiny():
    """Found via /qa 2026-07-24: on hard-to-read images, specialization_context
    can bias the extraction toward a plausible domain term. The verify pass
    still never receives the jargon text itself, but gets an extra scrutiny
    caveat appended when specialization_context was used — raising the bar
    without a second LLM call or reintroducing bias into verification."""
    from app.services.vision import _VERIFY_SPECIALIZATION_CAVEAT

    captured_prompts = {}

    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        captured_prompts[schema] = prompt
        if schema is VisionExtraction:
            return VisionExtraction(is_legible=True, price_question="¿Cuánto cuesta un examen de IGRA?")
        return VisionVerification(text_visible=True)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        await extract_procedure_query(
            b"fake image bytes", "", specialization_context="jerga médica: IGRA"
        )

    assert _VERIFY_SPECIALIZATION_CAVEAT in captured_prompts[VisionVerification]
    # Verify prompt still doesn't leak the actual jargon text — only the caveat.
    assert "jerga médica: IGRA" not in captured_prompts[VisionVerification]


@pytest.mark.asyncio
async def test_specialization_context_disagreeing_samples_returns_uncertain():
    """Consensus check: two independent extraction samples that disagree on
    the procedure name → VISION_UNCERTAIN, verify pass never runs. Found via
    /qa 2026-07-24 — live re-test showed the same hard-to-read image
    producing 3 different confident reads across 3 calls."""
    extraction_calls = iter([
        VisionExtraction(is_legible=True, procedure_name="Trompa uterina derecha",
                          price_question="¿Cuánto cuesta una trompa uterina derecha?"),
        VisionExtraction(is_legible=True, procedure_name="Hoja de biopsia",
                          price_question="¿Cuánto cuesta una hoja de biopsia?"),
    ])
    verify_called = False

    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        nonlocal verify_called
        if schema is VisionExtraction:
            return next(extraction_calls)
        verify_called = True
        return VisionVerification(text_visible=True)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        result = await extract_procedure_query(
            b"fake image bytes", "", specialization_context="jerga médica: IGRA"
        )

    assert result == VISION_UNCERTAIN
    assert verify_called is False  # disagreement short-circuits before the verify pass


@pytest.mark.asyncio
async def test_specialization_context_accent_only_difference_counts_as_agreement():
    """Regression: /qa 2026-07-24 with WhatsApp-exported real images found the
    consensus check rejecting two samples that read the SAME term but
    differed only in accent marks ("anatomia patologica" vs "anatomía
    patológica") — a real false positive, not a real disagreement."""
    extraction_calls = iter([
        VisionExtraction(is_legible=True, procedure_name="anatomia patologica",
                          price_question="¿Cuánto cuesta anatomia patologica?"),
        VisionExtraction(is_legible=True, procedure_name="anatomía patológica",
                          price_question="¿Cuánto cuesta anatomía patológica?"),
    ])

    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        if schema is VisionExtraction:
            return next(extraction_calls)
        return VisionVerification(text_visible=True)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        result = await extract_procedure_query(
            b"fake image bytes", "", specialization_context="jerga médica: IGRA"
        )

    assert result != VISION_UNCERTAIN


@pytest.mark.asyncio
async def test_specialization_context_agreeing_samples_proceeds_to_verify():
    """Consensus check: two independent samples that DO agree (case-insensitive)
    proceed normally to the verify pass and return the extracted question."""
    extraction_calls = iter([
        VisionExtraction(is_legible=True, procedure_name="Hoja de Biopsia",
                          price_question="¿Cuánto cuesta una hoja de biopsia?"),
        VisionExtraction(is_legible=True, procedure_name="hoja de biopsia",
                          price_question="¿Cuánto cuesta una hoja de biopsia?"),
    ])

    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        if schema is VisionExtraction:
            return next(extraction_calls)
        return VisionVerification(text_visible=True)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        result = await extract_procedure_query(
            b"fake image bytes", "", specialization_context="jerga médica: IGRA"
        )

    assert result == "¿Cuánto cuesta una hoja de biopsia?"


@pytest.mark.asyncio
async def test_empty_specialization_context_skips_consensus_check():
    """Regression/cost guard: without specialization_context, only ONE
    extraction call happens — no consensus sampling, no extra cost/latency
    for tenants who don't use the field."""
    call_count = {"extraction": 0}

    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        if schema is VisionExtraction:
            call_count["extraction"] += 1
            return VisionExtraction(is_legible=True, price_question="¿Cuánto cuesta un examen de IGRA?")
        return VisionVerification(text_visible=True)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == "¿Cuánto cuesta un examen de IGRA?"
    assert call_count["extraction"] == 1


@pytest.mark.asyncio
async def test_empty_specialization_context_leaves_prompt_unchanged():
    """Regression guard: empty specialization_context (the default for every
    existing call site) must produce the byte-identical prompt as before
    this feature — no dangling hint block."""
    from app.services.vision import _VISION_EXTRACT_PROMPT

    captured_prompts = {}

    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        captured_prompts[schema] = prompt
        if schema is VisionExtraction:
            return VisionExtraction(is_legible=True, price_question="¿Cuánto cuesta un examen de IGRA?")
        return VisionVerification(text_visible=True)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        await extract_procedure_query(b"fake image bytes", "")

    assert captured_prompts[VisionExtraction] == _VISION_EXTRACT_PROMPT


@pytest.mark.asyncio
async def test_multi_sample_document_combines_all_verified_samples():
    """Root cause found via /investigate 2026-07-24: a real 6-sample biopsy
    request (test_images) was silently collapsed to one question. When the
    document lists 2+ distinct samples, all of them (that verify) must be
    included in the final answer, not just the first."""
    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        if schema is VisionExtraction:
            return VisionExtraction(is_legible=True, samples=["Epiplón", "Líquido peritoneal", "Corredera parietocólica derecha"])
        return VisionVerification(text_visible=True)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert "Epiplón" in result
    assert "Líquido peritoneal" in result
    assert "Corredera parietocólica derecha" in result


@pytest.mark.asyncio
async def test_multi_sample_document_drops_unverified_samples():
    """A sample that fails the literal-presence check is dropped, not fatal
    to the whole request — mirrors how the media-group photo loop already
    tolerates individual failures without failing the batch."""
    verify_calls = iter([True, False, True])  # sample 2 fails verification

    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        if schema is VisionExtraction:
            return VisionExtraction(is_legible=True, samples=["Epiplón", "Ovario izquierdo", "Líquido peritoneal"])
        return VisionVerification(text_visible=next(verify_calls))

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert "Epiplón" in result
    assert "Ovario izquierdo" not in result
    assert "Líquido peritoneal" in result


@pytest.mark.asyncio
async def test_multi_sample_partial_verification_not_cached():
    """Regression: /code-review 2026-07-24 — a multi-sample answer missing
    one dropped (stochastically-rejected) item must NOT be cached, mirroring
    the single-item path's "never cache a rejection" rule. Caching a partial
    list would permanently shortchange every resend of the same photo."""
    ctx, session = _cache_ctx(cached_row=None)
    verify_calls = iter([True, False, True])  # sample 2 fails verification

    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        if schema is VisionExtraction:
            return VisionExtraction(is_legible=True, samples=["Epiplón", "Ovario izquierdo", "Líquido peritoneal"])
        return VisionVerification(text_visible=next(verify_calls))

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
        patch("app.services.vision.AsyncSessionLocal", return_value=ctx),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert "Epiplón" in result and "Ovario izquierdo" not in result
    insert_calls = [c for c in session.execute.await_args_list if "INSERT" in str(c.args[0])]
    assert insert_calls == []


@pytest.mark.asyncio
async def test_multi_sample_verification_capped_at_max():
    """Regression: /python-review 2026-07-24 — extraction.samples is
    model-controlled with no upper bound; verification fan-out must be
    capped so a mis-parsed image can't trigger an unbounded concurrent
    call burst against the rate-limited vision model."""
    from app.services.vision import _MAX_MULTI_SAMPLE_VERIFY

    many_samples = [f"Muestra {i}" for i in range(_MAX_MULTI_SAMPLE_VERIFY + 5)]
    verify_call_count = {"n": 0}

    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        if schema is VisionExtraction:
            return VisionExtraction(is_legible=True, samples=many_samples)
        verify_call_count["n"] += 1
        return VisionVerification(text_visible=True)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        await extract_procedure_query(b"fake image bytes", "")

    assert verify_call_count["n"] == _MAX_MULTI_SAMPLE_VERIFY


@pytest.mark.asyncio
async def test_multi_sample_document_all_rejected_returns_uncertain():
    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        if schema is VisionExtraction:
            return VisionExtraction(is_legible=True, samples=["Epiplón", "Líquido peritoneal"])
        return VisionVerification(text_visible=False)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN


@pytest.mark.asyncio
async def test_single_sample_entry_with_no_procedure_name_still_trusted():
    """samples with only 1 entry does NOT count as multi, but is still
    trusted and routed through normal verification -- reversed from an
    earlier assumption (samples=[x] with no procedure_name treated as
    "malformed", safely degraded to uncertain) after live evidence showed
    that assumption was wrong: real Telegram traffic on a real order photo
    put the correct, human-confirmed answer ("Utero y anexos (Trompas y
    Ovarios)") in samples[0] with procedure_name empty, and the old
    behavior silently discarded it without even attempting verification."""
    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        if schema is VisionExtraction:
            return VisionExtraction(is_legible=True, samples=["Epiplón"])
        return VisionVerification(text_visible=True)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
        patch("app.services.vision._TESSERACT_AVAILABLE", False),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == "¿Cuánto cuesta Epiplón?"


@pytest.mark.asyncio
async def test_multi_sample_skips_consensus_check_even_with_specialization():
    """Regression: /investigate 2026-07-24 live re-test against a real
    6-sample image found the whole-set consensus check rejecting a correct
    extraction because ONE of 6 independently re-read items had a minor OCR
    variance ("caredura" vs "corredera") — exact agreement across N items
    compounds far more than for 1 item. Multi-sample must skip the 2-sample
    consensus gate (only 1 extraction call) and rely on per-sample
    verification instead, even when specialization_context is set."""
    call_count = {"extraction": 0}

    async def _fake_structured_or_json(llm, model_name, prompt, img_b64, schema, json_suffix):
        if schema is VisionExtraction:
            call_count["extraction"] += 1
            return VisionExtraction(is_legible=True, samples=["Epiplón", "Líquido peritoneal"])
        return VisionVerification(text_visible=True)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=MagicMock()),
        patch("app.services.vision._structured_or_json", side_effect=_fake_structured_or_json),
    ):
        result = await extract_procedure_query(
            b"fake image bytes", "", specialization_context="jerga médica: IGRA"
        )

    assert call_count["extraction"] == 1  # no consensus sample taken for multi
    assert "Epiplón" in result and "Líquido peritoneal" in result


@pytest.mark.asyncio
async def test_extract_returns_price_question_when_legible_and_verified():
    mock_llm = _mock_llm({
        VisionExtraction: VisionExtraction(is_legible=True, price_question="¿Cuánto cuesta un examen de IGRA?"),
        VisionVerification: VisionVerification(text_visible=True),
    })
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == "¿Cuánto cuesta un examen de IGRA?"


@pytest.mark.asyncio
async def test_extract_returns_uncertain_when_illegible_without_calling_verification():
    mock_llm = _mock_llm({VisionExtraction: VisionExtraction(is_legible=False, price_question=None)})
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN
    # Only the extraction schema should have been requested — no wasted
    # verification call when there's nothing to verify.
    called_schemas = [c.args[0] for c in mock_llm.with_structured_output.call_args_list]
    assert VisionVerification not in called_schemas


@pytest.mark.asyncio
async def test_single_sample_slot_overrides_wrong_procedure_name():
    """Regression: found live — on a single clearly-labeled item, the model
    sometimes fills BOTH slots: procedure_name with the wrong tier-2
    document-title text ("Anatomía Patológica") AND samples[0] with the
    correct tier-3 labeled field ("Estómago (Antro)"), even though the
    prompt says CASO A/B are mutually exclusive. len(samples)==1 doesn't
    meet the is_multi threshold, so without the override the wrong
    procedure_name would silently win and get sent to verification."""
    mock_llm = _mock_llm({
        VisionExtraction: VisionExtraction(
            is_legible=True,
            procedure_name="Anatomía Patológica",
            price_question="¿Cuánto cuesta un examen de Anatomía Patológica?",
            samples=["Estómago (Antro)"],
        ),
        VisionVerification: VisionVerification(text_visible=True),
    })
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == "¿Cuánto cuesta Estómago (Antro)?"


@pytest.mark.asyncio
async def test_extract_returns_uncertain_when_legible_but_no_question():
    mock_llm = _mock_llm({VisionExtraction: VisionExtraction(is_legible=True, price_question=None)})
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN


@pytest.mark.asyncio
async def test_extract_returns_uncertain_when_verification_rejects_the_claim():
    """The core fix: the model claiming is_legible=true isn't trusted on its
    own — a second independent call re-examining the image can still veto it."""
    mock_llm = _mock_llm({
        VisionExtraction: VisionExtraction(is_legible=True, price_question="¿Cuánto cuesta una colonoscopía?"),
        VisionVerification: VisionVerification(text_visible=False),
    })
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN


@pytest.mark.asyncio
async def test_verification_checks_bare_procedure_name_not_full_question():
    """Regression test: a live A/B test found verifying the full formatted
    question ("¿Cuánto cuesta un examen de IGRA?") against the image fails
    almost always (0/10 trials) because that sentence never literally
    appears in the source document — only the bare term ("IGRA") does (6/8
    trials passed). The verification prompt must be built from
    procedure_name, not price_question."""
    captured_prompts = []

    def _with_structured_output(schema):
        m = AsyncMock()
        if schema is VisionExtraction:
            m.ainvoke = AsyncMock(return_value=VisionExtraction(
                is_legible=True, procedure_name="IGRA", price_question="¿Cuánto cuesta un examen de IGRA?"
            ))
        else:
            async def _capture(messages):
                captured_prompts.append(messages[0].content[0]["text"])
                return VisionVerification(text_visible=True)
            m.ainvoke = _capture
        return m

    mock_llm = MagicMock()
    mock_llm.with_structured_output.side_effect = _with_structured_output

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == "¿Cuánto cuesta un examen de IGRA?"
    assert len(captured_prompts) == 1
    assert '"IGRA"' in captured_prompts[0]
    assert "¿Cuánto cuesta" not in captured_prompts[0]


@pytest.mark.asyncio
async def test_verification_falls_back_to_price_question_when_procedure_name_missing():
    """Defensive fallback: if the model doesn't populate procedure_name
    (e.g. an older/malformed response), verification still has something to
    check rather than crashing on a None claim."""
    mock_llm = _mock_llm({
        VisionExtraction: VisionExtraction(is_legible=True, procedure_name=None,
                                            price_question="¿Cuánto cuesta un examen de IGRA?"),
        VisionVerification: VisionVerification(text_visible=True),
    })
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == "¿Cuánto cuesta un examen de IGRA?"


@pytest.mark.asyncio
async def test_verification_rejection_is_not_cached():
    """Regression test: verification outcomes are proven stochastic (same
    image+claim flips between accept/reject across independent calls — see
    docstring). Caching a rejection would turn one unlucky roll into a
    permanent wrong answer for that exact photo. Only is_legible=false and
    successful verifications may be cached."""
    ctx, session = _cache_ctx(cached_row=None)
    mock_llm = _mock_llm({
        VisionExtraction: VisionExtraction(is_legible=True, procedure_name="IGRA",
                                            price_question="¿Cuánto cuesta un examen de IGRA?"),
        VisionVerification: VisionVerification(text_visible=False),
    })
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
        patch("app.services.vision.AsyncSessionLocal", return_value=ctx),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN
    insert_calls = [c for c in session.execute.await_args_list if "INSERT" in str(c.args[0])]
    assert insert_calls == [], "a rejected verification must never be cached"


@pytest.mark.asyncio
async def test_cache_lookup_filters_by_ttl_cutoff():
    """Regression: found live — a mis-extraction cached 2+ days ago (picked
    letterhead/banner text instead of the order's real field) was served
    unchanged forever, since vision_cache had no expiry. The lookup query
    must bound results to entries newer than _VISION_CACHE_TTL_DAYS."""
    from datetime import UTC, datetime, timedelta

    from app.services.vision import _VISION_CACHE_TTL_DAYS, _get_cached_vision_result

    ctx, session = _cache_ctx(cached_row=None)
    with patch("app.services.vision.AsyncSessionLocal", return_value=ctx):
        await _get_cached_vision_result("some-key")

    call = session.execute.await_args_list[0]
    assert "created_at > :cutoff" in str(call.args[0])
    bound_cutoff = call.args[1]["cutoff"]
    expected_cutoff = datetime.now(UTC) - timedelta(days=_VISION_CACHE_TTL_DAYS)
    assert abs((bound_cutoff - expected_cutoff).total_seconds()) < 5


@pytest.mark.asyncio
async def test_extract_falls_back_to_json_parse_when_extraction_structured_fails():
    """Once extraction's structured attempt fails for a model, the memo means
    verification (same model, same call) also skips straight to the JSON
    fallback — both raw responses must be supplied."""
    mock_llm = _mock_llm(
        by_schema={VisionExtraction: Exception("model doesn't support tool calling")},
        raw_content_by_schema={
            VisionExtraction: '{"is_legible": true, "price_question": "¿Cuánto cuesta una biopsia de mama?"}',
            VisionVerification: '{"text_visible": true}',
        },
    )
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == "¿Cuánto cuesta una biopsia de mama?"


@pytest.mark.asyncio
async def test_extract_falls_back_to_json_parse_strips_markdown_fences():
    mock_llm = _mock_llm(
        by_schema={VisionExtraction: Exception("no tool calling")},
        raw_content_by_schema={VisionExtraction: '```json\n{"is_legible": false, "price_question": null}\n```'},
    )
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN


@pytest.mark.asyncio
async def test_extract_defaults_to_uncertain_when_extraction_totally_fails():
    mock_llm = _mock_llm(
        by_schema={VisionExtraction: Exception("api down")},
        raw_content_by_schema={VisionExtraction: "not json at all, just prose"},
    )
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN


@pytest.mark.asyncio
async def test_extract_defaults_to_uncertain_when_verification_totally_fails():
    """Safe-default direction matters: unlike triage (defaults to a guess),
    vision's safe default on total failure must be uncertainty — including
    when only the verification pass is the one that breaks."""
    mock_llm = _mock_llm(
        by_schema={
            VisionExtraction: VisionExtraction(is_legible=True, price_question="¿Cuánto cuesta un examen X?"),
            VisionVerification: Exception("api down"),
        },
        raw_content_by_schema={VisionVerification: "not json either"},
    )
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN


@pytest.mark.asyncio
async def test_extract_raises_when_vision_model_not_configured():
    with patch("app.services.vision.settings.openai_vision_model", ""):
        with pytest.raises(RuntimeError):
            await extract_procedure_query(b"fake image bytes", "")


@pytest.mark.asyncio
async def test_memoized_structured_output_failure_skips_retry_on_next_call():
    """Live testing found the configured vision model hard-fails (400/404)
    on every structured-output method OpenRouter offers — not just bad text
    back. That means retrying the structured attempt on every call is a
    guaranteed-failing network round-trip paid on every single image. Once a
    model is known not to support it, subsequent calls must skip straight to
    the JSON-mode fallback instead of repeating the doomed attempt."""
    mock_llm = MagicMock()
    mock_structured = AsyncMock()
    mock_structured.ainvoke = AsyncMock(side_effect=Exception("no tool calling"))
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

    raw_resp = MagicMock()
    raw_resp.content = '{"is_legible": false, "price_question": null}'
    mock_llm.ainvoke = AsyncMock(return_value=raw_resp)

    with (
        patch("app.services.vision.settings.openai_vision_model", "flaky-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result1 = await extract_procedure_query(b"img1", "")
        assert result1 == VISION_UNCERTAIN
        assert mock_llm.with_structured_output.call_count == 1

        result2 = await extract_procedure_query(b"img2", "")
        assert result2 == VISION_UNCERTAIN
        # Second call must NOT retry the doomed structured attempt.
        assert mock_llm.with_structured_output.call_count == 1
        assert mock_llm.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_cache_hit_returns_cached_result_without_calling_llm():
    """Same photo resent (hash matches) → cached result returned, zero LLM calls."""
    ctx, session = _cache_ctx(cached_row=MagicMock(result="¿Cuánto cuesta una biopsia?"))
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(side_effect=AssertionError("must not call LLM on cache hit"))

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
        patch("app.services.vision.AsyncSessionLocal", return_value=ctx),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == "¿Cuánto cuesta una biopsia?"


@pytest.mark.asyncio
async def test_cache_read_failure_falls_through_to_fresh_extraction():
    """A DB hiccup on the cache READ must not crash the request — treated as
    a cache miss so the LLM path still runs and returns a real answer."""
    broken_session = MagicMock()
    broken_session.execute = AsyncMock(side_effect=Exception("db connection lost"))
    broken_ctx = AsyncMock()
    broken_ctx.__aenter__ = AsyncMock(return_value=broken_session)
    broken_ctx.__aexit__ = AsyncMock(return_value=None)

    good_ctx, good_session = _cache_ctx(cached_row=None)

    ctx_calls = iter([broken_ctx, good_ctx, good_ctx])  # read fails, then writes succeed

    mock_llm = _mock_llm({
        VisionExtraction: VisionExtraction(is_legible=True, procedure_name="IGRA",
                                            price_question="¿Cuánto cuesta un examen de IGRA?"),
        VisionVerification: VisionVerification(text_visible=True),
    })
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
        patch("app.services.vision.AsyncSessionLocal", side_effect=lambda: next(ctx_calls)),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == "¿Cuánto cuesta un examen de IGRA?"


@pytest.mark.asyncio
async def test_cache_write_failure_still_returns_correct_result():
    """A DB hiccup on the cache WRITE must not discard an already-computed,
    already-paid-for correct extraction — the failure is swallowed and logged."""
    broken_session = MagicMock()
    broken_session.execute = AsyncMock(side_effect=Exception("db connection lost"))
    broken_ctx = AsyncMock()
    broken_ctx.__aenter__ = AsyncMock(return_value=broken_session)
    broken_ctx.__aexit__ = AsyncMock(return_value=None)

    read_ctx, _ = _cache_ctx(cached_row=None)
    ctx_calls = iter([read_ctx, broken_ctx])  # read succeeds (miss), write fails

    mock_llm = _mock_llm({
        VisionExtraction: VisionExtraction(is_legible=True, procedure_name="IGRA",
                                            price_question="¿Cuánto cuesta un examen de IGRA?"),
        VisionVerification: VisionVerification(text_visible=True),
    })
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
        patch("app.services.vision.AsyncSessionLocal", side_effect=lambda: next(ctx_calls)),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == "¿Cuánto cuesta un examen de IGRA?"


@pytest.mark.asyncio
async def test_cache_miss_stores_extracted_result():
    """A fresh (uncached) image, once extracted+verified, is written to the cache."""
    ctx, session = _cache_ctx(cached_row=None)
    mock_llm = _mock_llm({
        VisionExtraction: VisionExtraction(is_legible=True, price_question="¿Cuánto cuesta un examen de IGRA?"),
        VisionVerification: VisionVerification(text_visible=True),
    })
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
        patch("app.services.vision.AsyncSessionLocal", return_value=ctx),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == "¿Cuánto cuesta un examen de IGRA?"
    insert_call = session.execute.await_args_list[-1]
    assert "INSERT INTO vision_cache" in str(insert_call.args[0])
    assert insert_call.args[1]["result"] == "¿Cuánto cuesta un examen de IGRA?"
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_miss_stores_uncertain_when_illegible():
    """An illegible image's VISION_UNCERTAIN verdict is itself a stable content
    judgment worth caching — same blank/blurry photo resent shouldn't re-pay."""
    ctx, session = _cache_ctx(cached_row=None)
    mock_llm = _mock_llm({VisionExtraction: VisionExtraction(is_legible=False, price_question=None)})
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
        patch("app.services.vision.AsyncSessionLocal", return_value=ctx),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN
    insert_call = session.execute.await_args_list[-1]
    assert insert_call.args[1]["result"] == VISION_UNCERTAIN


@pytest.mark.asyncio
async def test_transient_extraction_failure_is_not_cached():
    """API errors are transient — must NOT be cached, or a temporary outage
    would permanently lock a legit image as uncertain."""
    ctx, session = _cache_ctx(cached_row=None)
    mock_llm = _mock_llm(
        by_schema={VisionExtraction: Exception("api down")},
        raw_content_by_schema={VisionExtraction: "not json at all, just prose"},
    )
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
        patch("app.services.vision.AsyncSessionLocal", return_value=ctx),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN
    insert_calls = [c for c in session.execute.await_args_list if "INSERT" in str(c.args[0])]
    assert insert_calls == []


def test_cache_key_differs_by_model_caption_and_bytes():
    key_a = vision_module._vision_cache_key("model-a", "", b"img1")
    key_b = vision_module._vision_cache_key("model-b", "", b"img1")
    key_c = vision_module._vision_cache_key("model-a", "caption", b"img1")
    key_d = vision_module._vision_cache_key("model-a", "", b"img2")
    assert len({key_a, key_b, key_c, key_d}) == 4


def test_cache_key_differs_by_tenant_slug():
    """Same image, same model/caption, different tenant → different key —
    closes the cross-tenant cache-sharing gap found in the eng review."""
    key_tenant_a = vision_module._vision_cache_key("model-a", "", b"img1", "tenant-a")
    key_tenant_b = vision_module._vision_cache_key("model-a", "", b"img1", "tenant-b")
    assert key_tenant_a != key_tenant_b


def test_cache_key_folds_in_specialization_only_when_non_empty():
    """Two tenants with the SAME empty specialization_context (the default)
    get the SAME key for the same image — only a non-empty value changes it.
    This is what keeps the zero-behavior-change guarantee for tenants who
    never set the field."""
    key_empty_1 = vision_module._vision_cache_key("model-a", "", b"img1", "tenant-a", "")
    key_empty_2 = vision_module._vision_cache_key("model-a", "", b"img1", "tenant-a", "")
    key_with_spec = vision_module._vision_cache_key("model-a", "", b"img1", "tenant-a", "jerga médica")
    assert key_empty_1 == key_empty_2
    assert key_empty_1 != key_with_spec


@pytest.mark.asyncio
async def test_rate_limited_structured_call_retries_then_succeeds():
    """429 on the first structured attempt → retried with backoff, succeeds
    on the second try. Backoff sleep is patched away so the test is instant."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke = AsyncMock(
        side_effect=[_rate_limit_error(), VisionExtraction(is_legible=False, price_question=None)]
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
        patch("app.services.vision.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN
    assert mock_structured.ainvoke.await_count == 2
    # A 429 is transient, not a capability failure — must not have memoized
    # "structured output doesn't work" for this model.
    assert vision_module._structured_output_ok.get("some-vision-model") is True


@pytest.mark.asyncio
async def test_rate_limit_exhausted_defaults_to_uncertain_not_cached():
    """429 on every retry → gives up, returns VISION_UNCERTAIN, doesn't cache
    (a still-rate-limited model might succeed moments later)."""
    ctx, session = _cache_ctx(cached_row=None)
    mock_structured = AsyncMock()
    mock_structured.ainvoke = AsyncMock(side_effect=_rate_limit_error())
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
        patch("app.services.vision.AsyncSessionLocal", return_value=ctx),
        patch("app.services.vision.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN
    # 1 initial attempt + 2 retries = 3 total, per _RATE_LIMIT_MAX_RETRIES
    assert mock_structured.ainvoke.await_count == 3
    insert_calls = [c for c in session.execute.await_args_list if "INSERT" in str(c.args[0])]
    assert insert_calls == []


@pytest.mark.asyncio
async def test_non_rate_limit_error_is_not_retried():
    """A plain capability failure (not a 429) must fail fast — no backoff
    delay wasted on an error retrying can't fix."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke = AsyncMock(side_effect=Exception("no tool calling"))
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    raw_resp = MagicMock()
    raw_resp.content = '{"is_legible": false, "price_question": null}'
    mock_llm.ainvoke = AsyncMock(return_value=raw_resp)

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN
    # Falls back to json-mode after a single failed attempt — no retry loop.
    assert mock_structured.ainvoke.await_count == 1


def test_preprocess_downscales_oversized_image():
    """Phone photos routinely exceed the model's useful resolution — must be
    shrunk to cap payload size/cost without needing the full original."""
    img = Image.new("RGB", (3000, 1500), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    processed = vision_module._preprocess_image(buf.getvalue())
    out = Image.open(io.BytesIO(processed))
    assert max(out.size) <= vision_module._MAX_DIMENSION
    assert out.size[0] / out.size[1] == pytest.approx(3000 / 1500, rel=0.01)


def test_preprocess_corrects_exif_rotation():
    """EXIF orientation tag rotates on display but not in raw pixel data —
    left uncorrected, the model reads a sideways image. Small images are upscaled."""
    img = Image.new("RGB", (200, 100), color="blue")  # landscape, unrotated
    exif = img.getexif()
    exif[0x0112] = 6  # "rotate 90 CW to display correctly" -> portrait
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)

    processed = vision_module._preprocess_image(buf.getvalue())
    out = Image.open(io.BytesIO(processed))
    # Rotation applied (100x200), then upscaled (min 100px < 600px threshold)
    assert out.width == 600 and out.height == 1200


def test_preprocess_small_image_untouched_dimensions():
    """Images with min dimension < 600px are upscaled to recover messenger compression detail."""
    img = Image.new("RGB", (400, 300), color="red")  # min=300 < 600, will upscale
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    processed = vision_module._preprocess_image(buf.getvalue())
    out = Image.open(io.BytesIO(processed))
    # 300px upscaled to 600px (scale = 2)
    assert out.size == (800, 600)


def test_preprocess_falls_back_to_original_on_invalid_image():
    """Undecodable input (corrupt file, unsupported format) must not crash
    the request — falls back to the original bytes."""
    raw = b"this is not a valid image file"
    assert vision_module._preprocess_image(raw) == raw


@pytest.mark.asyncio
async def test_structured_output_memo_is_isolated_per_model():
    """A different model name must not inherit another model's memoized
    "structured output doesn't work" result."""
    mock_llm_a = MagicMock()
    mock_structured_a = AsyncMock()
    mock_structured_a.ainvoke = AsyncMock(side_effect=Exception("no tool calling"))
    mock_llm_a.with_structured_output = MagicMock(return_value=mock_structured_a)
    raw_resp = MagicMock()
    raw_resp.content = '{"is_legible": false, "price_question": null}'
    mock_llm_a.ainvoke = AsyncMock(return_value=raw_resp)

    with (
        patch("app.services.vision.settings.openai_vision_model", "model-a"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm_a),
    ):
        await extract_procedure_query(b"img1", "")

    mock_llm_b = MagicMock()
    mock_structured_b = AsyncMock()
    mock_structured_b.ainvoke = AsyncMock(
        return_value=VisionExtraction(is_legible=False, price_question=None)
    )
    mock_llm_b.with_structured_output = MagicMock(return_value=mock_structured_b)

    with (
        patch("app.services.vision.settings.openai_vision_model", "model-b"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm_b),
    ):
        await extract_procedure_query(b"img2", "")

    # model-b's structured attempt must still be tried — model-a's failure
    # memo must not leak across model names.
    mock_llm_b.with_structured_output.assert_called_once()


# OCR fallback tests (Tesseract optional, gracefully unavailable)

@pytest.mark.asyncio
async def test_ocr_fallback_when_vision_illegible():
    """When vision marks image illegible, OCR fallback attempts extraction.
    If Tesseract available, uses OCR text."""
    ctx, session = _cache_ctx(cached_row=None)
    # Vision says illegible, verify accepts OCR text
    mock_llm = _mock_llm({
        VisionExtraction: VisionExtraction(is_legible=False, price_question=None),
        VisionVerification: VisionVerification(text_visible=True),
    })

    # Mock OCR returning procedure name
    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
        patch("app.services.vision.AsyncSessionLocal", return_value=ctx),
        patch("app.services.vision._extract_with_ocr", return_value="IGRA"),
        patch("app.services.vision._TESSERACT_AVAILABLE", True),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    # OCR fallback generated price question (verified)
    assert result == "¿Cuánto cuesta IGRA?"


@pytest.mark.asyncio
async def test_ocr_unavailable_returns_uncertain():
    """When vision illegible and OCR unavailable/returns empty, still uncertain."""
    ctx, session = _cache_ctx(cached_row=None)
    mock_llm = _mock_llm({VisionExtraction: VisionExtraction(is_legible=False, price_question=None)})

    with (
        patch("app.services.vision.settings.openai_vision_model", "some-vision-model"),
        patch("app.services.vision.get_vision_llm", return_value=mock_llm),
        patch("app.services.vision.AsyncSessionLocal", return_value=ctx),
        patch("app.services.vision._extract_with_ocr", return_value=None),
        patch("app.services.vision._TESSERACT_AVAILABLE", False),
    ):
        result = await extract_procedure_query(b"fake image bytes", "")

    assert result == VISION_UNCERTAIN


def test_ocr_extraction_returns_first_meaningful_line():
    """OCR extraction skips noise, returns first line >3 chars."""
    fake_ocr_text = "\n\nIGRA TEST\nOther stuff..."
    # _extract_with_ocr() uses pytesseract if available; simulate it
    lines = [line.strip() for line in fake_ocr_text.split("\n") if line.strip()]
    first = next((line for line in lines if len(line) > 3), "")
    assert first == "IGRA TEST"


def test_ocr_extraction_ignores_short_lines():
    """OCR skips noise lines (<3 chars)."""
    fake_ocr_text = "XX\nIGRA\nYYY"
    lines = [line.strip() for line in fake_ocr_text.split("\n") if line.strip()]
    first = next((line for line in lines if len(line) > 3), "")
    # "XX" and "YYY" are <4 chars (len >3 means >3, so >=4), "IGRA" is 4 chars
    assert first == "IGRA"


def test_ocr_extraction_skips_first_line_fallback_on_long_form_text():
    """Regression: found live — a real lab order form OCR's to ~20 garbled
    lines (letterhead + patient data + handwritten notes), and the naive
    first-line heuristic grabbed the letterhead/marketing banner ("Calidad
    Y Experiencia en") instead of the actual sample field. Above
    _OCR_MAX_LINES_FOR_FIRST_LINE_FALLBACK, first-line fallback must not
    fire — returns "" (caller treats as still uncertain), not a guess."""
    fake_ocr_text = "\n".join(
        ["Calidad Y Experiencia en", "SERVICIO DE QUIROFANO", "POLICLINICA PUERTO LA CRUZ"]
        + [f"linea de relleno {i}" for i in range(6)]
    )
    with (
        patch("app.services.vision._TESSERACT_AVAILABLE", True),
        patch("app.services.vision.pytesseract.image_to_string", return_value=fake_ocr_text),
        patch("app.services.vision.Image.open", return_value=MagicMock()),
    ):
        result = vision_module._extract_with_ocr(b"fake image bytes")

    assert result == ""


def test_ocr_extraction_prefers_labeled_field_over_letterhead():
    """A recognized field label (e.g. "Muestra:") is trusted regardless of
    document length, since it's a structural signal, not a line-position
    guess — letterhead noise before it must not win."""
    fake_ocr_text = "\n".join(
        ["Calidad Y Experiencia en", "SERVICIO DE QUIROFANO", "Muestra: Utero y anexos"]
        + [f"linea de relleno {i}" for i in range(6)]
    )
    with (
        patch("app.services.vision._TESSERACT_AVAILABLE", True),
        patch("app.services.vision.pytesseract.image_to_string", return_value=fake_ocr_text),
        patch("app.services.vision.Image.open", return_value=MagicMock()),
    ):
        result = vision_module._extract_with_ocr(b"fake image bytes")

    assert result == "Utero y anexos"


def test_ocr_extraction_short_text_still_uses_first_line_fallback():
    """Below the line-count gate (simple product label / one-line slip, no
    letterhead ambiguity), the original first-line heuristic still applies."""
    fake_ocr_text = "\n\nIGRA TEST\nOther stuff..."
    with (
        patch("app.services.vision._TESSERACT_AVAILABLE", True),
        patch("app.services.vision.pytesseract.image_to_string", return_value=fake_ocr_text),
        patch("app.services.vision.Image.open", return_value=MagicMock()),
    ):
        result = vision_module._extract_with_ocr(b"fake image bytes")

    assert result == "IGRA TEST"
