import asyncio
import base64
import hashlib
import io
import json
import logging
import re
import shutil
import unicodedata
from pathlib import Path

import openai
from langchain_core.messages import HumanMessage
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pydantic import BaseModel
from sqlalchemy import text

from app.config import settings
from app.db import AsyncSessionLocal
from app.schemas.vision import VisionExtraction, VisionVerification
from app.services.llm import get_vision_llm

logger = logging.getLogger(__name__)

# OCR optional — graceful fallback if Tesseract not installed
_TESSERACT_AVAILABLE = False
try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    logger.info("pytesseract not installed — OCR fallback unavailable")

# ponytail: Windows dev machines commonly get the tesseract binary via
# winget/choco with neither the exe on PATH nor the spa language pack
# bundled (apt's tesseract-ocr-spa has no Windows equivalent package) —
# without this, local dev silently runs English-only OCR on Spanish medical
# text and gets garbage. Docker/production is unaffected: apt puts the
# binary on PATH and tesseract-ocr-spa ships the language data directly.
# Upgrade path: if a future base install bundles spa by default, this
# becomes a no-op (shutil.which finds it, override stays empty).
_TESSDATA_OVERRIDE = ""
if _TESSERACT_AVAILABLE and not shutil.which("tesseract"):
    _win_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if _win_tesseract.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_win_tesseract)
        _user_tessdata = Path.home() / ".tessdata"
        if (_user_tessdata / "spa.traineddata").exists():
            # No quotes: pytesseract's config parser uses shlex.split(config,
            # posix=False) on Windows, which does not strip quote chars the
            # way posix shlex does — quoting here embeds literal quotes into
            # the path tesseract receives and breaks --tessdata-dir entirely.
            _TESSDATA_OVERRIDE = f"--tessdata-dir {_user_tessdata}"

MAX_MEDIA_BYTES = 10 * 1024 * 1024  # 10 MB — shared cap for voice/audio/image downloads

VISION_UNCERTAIN = "__VISION_UNCERTAIN__"

_VISION_EXTRACT_PROMPT = """\
Analiza esta imagen (documento, orden, comprobante, etiqueta de producto, ficha...).

Marca is_legible=true SOLO si podés leer con certeza el nombre del ítem o \
procedimiento escrito en la imagen, letra por letra. Si el texto está borroso, \
cortado, o tenés la MÍNIMA duda sobre qué dice — marcá is_legible=false y dejá \
procedure_name, price_question y samples vacíos. NUNCA adivines ni asumas un \
ítem "parecido" solo porque el contexto del negocio te resulte familiar.

CASO A — un solo ítem/procedimiento/muestra (el caso más común): completá:
- procedure_name: el nombre EXACTO y LITERAL del ítem o procedimiento tal como \
está escrito en la imagen, sin agregar palabras (ejemplo: "IGRA", "zapatilla \
running talla 42").
- price_question: una pregunta de precio en español usando ese mismo texto, por \
ejemplo: "¿Cuánto cuesta un examen de IGRA?" o "¿Cuánto cuesta una zapatilla \
running talla 42?". No traduzcas a un sinónimo ni asumas qué ítem "similar" \
podría ser.
- samples: dejalo vacío (null).

CASO B — la imagen lista 2 O MÁS muestras/ítems DISTINTOS bajo una misma orden \
(ejemplo: una solicitud de biopsia con "Muestra #1: Epiplón", "Muestra #2: \
Líquido peritoneal", etc.) — NO es motivo para is_legible=false, es un caso \
válido y frecuente:
- procedure_name y price_question: dejalos vacíos (null).
- samples: una lista con el nombre EXACTO y LITERAL de cada muestra/ítem, uno \
por elemento, en el mismo orden en que aparecen (ejemplo: ["Epiplón", "TU de \
ovario izquierdo (fracción)", "Líquido peritoneal"]). No incluyas valores de \
laboratorio ni firmas, solo los nombres de muestra/ítem en sí.
"""

_EXTRACT_JSON_SUFFIX = (
    '\nReply ONLY with JSON: {"is_legible": <true|false>, '
    '"procedure_name": <string or null>, "price_question": <string or null>, '
    '"samples": <array of strings or null>}'
)

_VERIFY_PROMPT_TEMPLATE = """\
Mirá esta imagen otra vez, con atención. ¿Aparece literalmente escrito en la imagen \
el siguiente texto, o una variante muy cercana del mismo ítem/procedimiento?

Texto a verificar: "{claim}"

Marcá text_visible=true SOLO si podés señalar con certeza dónde en la imagen aparece \
ese texto o uno equivalente. Si no lo ves, no estás seguro, o la imagen no contiene \
ese texto, marcá text_visible=false. No asumas por contexto general — esto es \
una verificación de presencia literal del texto, no un juicio de plausibilidad.
"""

_VERIFY_JSON_SUFFIX = '\nReply ONLY with JSON: {"text_visible": <true|false>}'

# Appended to the verify prompt only when specialization_context was used for
# extraction — found via /qa 2026-07-24: on a hard-to-read image, the
# specialization hint can steer the extraction toward a plausible domain term
# that the verify pass then "confirms" is present, when an unbiased
# extraction of the SAME image read something else entirely. The base verify
# prompt already warns against plausibility judgments in general; this raises
# the bar specifically for the case that caused it, without a second LLM call
# or new schema field. No-op (not appended) when specialization_context is
# empty — byte-identical verify prompt for every tenant that doesn't use it.
_VERIFY_SPECIALIZATION_CAVEAT = """
ADVERTENCIA: el texto a verificar pudo haber sido influenciado por jerga \
específica del negocio durante la extracción. Sé especialmente estricto acá — \
si tenés cualquier duda de que el texto exacto esté escrito en la imagen (en vez \
de sonar simplemente "razonable" para este tipo de negocio), marcá \
text_visible=false.
"""


def _strip_fences(content: str) -> str:
    content = content.strip()
    content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
    return re.sub(r"\s*```$", "", content).strip()


def _normalize_for_comparison(text_value: str) -> str:
    """Lowercase + accent-strip (NFKD, drop combining marks) so two
    independent vision samples that read the SAME term but differ only in
    accent marks (e.g. "anatomia patologica" vs "anatomía patológica" — real
    /qa 2026-07-24 finding) aren't misclassified as a consensus disagreement."""
    normalized = unicodedata.normalize("NFKD", text_value.strip().lower())
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _consensus_agrees(name_1: str, name_2: str) -> bool:
    """Two consensus samples agree if they're identical, OR one is a
    substring of the other (found in /code-review): VisionExtraction's
    procedure_name and price_question are independent optional fields, so
    one call filling only the bare term ("igra") and the other filling only
    the full formatted question ("¿cuánto cuesta un examen de igra?") for
    the SAME correctly-read item would otherwise never match exactly."""
    if not name_1 or not name_2:
        return False
    return name_1 == name_2 or name_1 in name_2 or name_2 in name_1


# Per-model memo of whether with_structured_output actually works. Live
# testing showed the configured vision model fails structured output with a
# hard 400/404 API error (no tool-use / json_schema / json_mode support at
# all via OpenRouter for this model) — not just malformed text back. That
# means the first attempt is a guaranteed-failing network round-trip on
# every single call until proven otherwise. Once a model is known not to
# support it, skip straight to the prompted-JSON retry instead of paying for
# a call that can't succeed — halves the vision calls per stage in steady
# state. Resets on process restart, which is fine: it's a perf cache, not a
# correctness dependency, and self-heals with one wasted probe.
_structured_output_ok: dict[str, bool] = {}

# High photo volume makes concurrent uploads far more likely to collide with
# OpenRouter's per-model rate limit than a single request ever would — a 429
# is transient (the request itself was fine, just mistimed), unlike a 400/404
# capability failure, so it's worth a couple of short backoff retries instead
# of giving up immediately and asking the user to retype their question.
_RATE_LIMIT_MAX_RETRIES = 2
_RATE_LIMIT_BASE_DELAY = 1.5  # seconds

# Multi-sample verification fans out one concurrent LLM call per sample
# (found in /python-review) — extraction.samples is model-controlled with no
# upper bound, so a mis-parsed image (or a burst of multi-sample photos in
# one media group) could otherwise trigger an unbounded concurrent-call
# spike against the same rate-limited vision model. 20 comfortably covers
# any realistic multi-sample order; ceiling upgrade if that ever proves too
# small — cap it, not the underlying gather logic.
_MAX_MULTI_SAMPLE_VERIFY = 20


def _is_rate_limited(exc: BaseException) -> bool:
    return isinstance(exc, openai.RateLimitError) or getattr(exc, "status_code", None) == 429


async def _call_with_rate_limit_retry(fn, *args, **kwargs):
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            if not _is_rate_limited(exc) or attempt == _RATE_LIMIT_MAX_RETRIES:
                raise
            delay = _RATE_LIMIT_BASE_DELAY * (2**attempt)
            logger.warning("vision_rate_limited attempt=%d retrying_in=%.1fs", attempt + 1, delay)
            await asyncio.sleep(delay)


# Phone photos routinely arrive at 3000-4000px on the long side, and/or
# rotated via an EXIF orientation tag that most viewers apply automatically
# but raw bytes don't — left as-is that's both a bloated base64 payload
# (cost/latency) and a real risk of the model reading a sideways image wrong.
_MAX_DIMENSION = 1600
_JPEG_QUALITY = 85


def _preprocess_image(img_bytes: bytes) -> bytes:
    """Enhance medical order images (WhatsApp/Telegram degraded input).

    Pipeline: EXIF rotation → upscale small images → denoise → contrast boost → downscale.
    Upscale recovers detail lost in messenger compression. Denoise reduces artifacts.
    Contrast makes text sharper for vision + OCR. Falls back to original on any error."""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Upscale small images (WhatsApp/Telegram compress to ~500-800px; recover detail)
        min_dim = min(img.size)
        if min_dim < 600:
            scale = 600 / min_dim
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug("vision_upscale from=%dpx to=%dpx", min_dim, int(min_dim * scale))

        # Denoise: reduce compression artifacts (slight blur reduces noise)
        img = img.filter(ImageFilter.MedianFilter(size=3))

        # Contrast boost: make text sharper (critical for OCR + vision)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)  # 50% contrast boost

        # Downscale to model limit (final size)
        if max(img.size) > _MAX_DIMENSION:
            img.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("vision_preprocess_failed err=%s using original bytes", exc)
        return img_bytes


def _extract_with_ocr(img_bytes: bytes) -> str | None:
    """Extract procedure name with OCR (fallback for vision confidence boost).

    Returns extracted text or None if Tesseract unavailable.
    Returns empty string if Tesseract available but found no text."""
    if not _TESSERACT_AVAILABLE:
        return None

    try:
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img, lang="spa+eng", config=_TESSDATA_OVERRIDE)
        # Extract first meaningful line (procedure name typically appears early)
        for line in text.split("\n"):
            line = line.strip()
            if len(line) > 3:  # Filter noise
                return line
        return ""
    except Exception as exc:
        logger.warning("ocr_extraction_failed err=%s", exc)
        return None


def _vision_cache_key(
    model_name: str, caption: str, img_bytes: bytes,
    tenant_slug: str = "", specialization_context: str = "",
) -> str:
    # tenant_slug always folded in (vision_cache is a global table with zero
    # tenant scoping otherwise — two tenants sending a byte-identical image
    # would share a cached result). specialization_context only folded in
    # when non-empty so tenants who never set it don't pay a second cache
    # invalidation on top of the one-time tenant_slug invalidation.
    h = hashlib.sha256()
    h.update(f"{model_name}:{caption}:{tenant_slug}:".encode())
    if specialization_context:
        h.update(f"{specialization_context}:".encode())
    h.update(img_bytes)
    return h.hexdigest()


async def _get_cached_vision_result(key: str) -> str | None:
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            text("SELECT result FROM vision_cache WHERE key = :key"), {"key": key}
        )).first()
        return row.result if row else None


async def _store_vision_result(key: str, result: str) -> None:
    """Best-effort: a cache WRITE failure must never discard an already-computed
    (and already-paid-for) correct result — log and move on."""
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                text("INSERT INTO vision_cache (key, result) VALUES (:key, :result) ON CONFLICT (key) DO NOTHING"),
                {"key": key, "result": result},
            )
            await db.commit()
    except Exception as exc:
        logger.warning("vision_cache_write_failed key=%s err=%s", key[:12], exc)


async def _structured_or_json(
    llm, model_name: str, prompt: str, img_b64: str, schema: type[BaseModel], json_suffix: str
):
    """Ask for a structured field via with_structured_output; if that fails,
    retry once with an explicit "reply only with JSON" instruction and parse
    it by hand — same primary/fallback shape as triage.py."""
    image_block = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}

    if _structured_output_ok.get(model_name, True):
        try:
            message = HumanMessage(content=[{"type": "text", "text": prompt}, image_block])
            result = await llm.with_structured_output(schema).ainvoke([message])
            _structured_output_ok[model_name] = True
            return result
        except Exception as exc:
            if _is_rate_limited(exc):
                # Transient — the model DOES support structured output, it's
                # just rate-limited right now. Must not memoize this as a
                # capability failure, or one busy moment permanently disables
                # structured output for the model.
                raise
            logger.warning("vision_structured_failed model=%s err=%s falling back to json parse", model_name, exc)
            _structured_output_ok[model_name] = False

    json_message = HumanMessage(content=[{"type": "text", "text": prompt + json_suffix}, image_block])
    resp = await llm.ainvoke([json_message])
    return schema.model_validate(json.loads(_strip_fences(resp.content)))


async def _run_consensus_check(
    llm, model_name: str, extraction: VisionExtraction, base_prompt: str, img_b64: str,
) -> bool:
    """Consensus check for the single-item path (found via /qa 2026-07-24):
    specialization_context can steer extraction toward a plausible domain
    term on hard-to-read images, and the model's own confidence is
    unreliable — but disagreement between two INDEPENDENT samples is a
    direct, measured signal, not a proxy.

    Uses base_prompt (WITHOUT the specialization prefix), not the biased
    prompt used for the primary extraction (found in /code-review):
    re-sending the identical biased prompt can only catch stochastic
    sampling noise, never a consistent, prompt-driven steering toward the
    same wrong term. Reading the image again with NO domain hint is a
    genuinely independent second opinion.

    Returns True if it's safe to proceed (the two samples agree). False
    means the caller should return VISION_UNCERTAIN, uncached — same
    stochastic-rejection reasoning as a verification rejection; one unlucky
    pair of samples must not become a permanent wrong answer. Covers both a
    real disagreement and a transient failure on the resample call itself —
    both resolve to the identical caller action, so callers don't need to
    branch on which one happened.
    """
    try:
        extraction_2: VisionExtraction = await _call_with_rate_limit_retry(
            _structured_or_json, llm, model_name, base_prompt, img_b64, VisionExtraction, _EXTRACT_JSON_SUFFIX
        )
    except Exception as exc:
        logger.warning("vision_consensus_sample_failed=%s defaulting to uncertain", exc)
        return False

    name_1 = _normalize_for_comparison(extraction.procedure_name or extraction.price_question or "")
    name_2 = _normalize_for_comparison(extraction_2.procedure_name or extraction_2.price_question or "")
    if not _consensus_agrees(name_1, name_2):
        logger.warning("vision_consensus_disagreement first_len=%d second_len=%d", len(name_1), len(name_2))
        return False
    return True


async def _handle_multi_sample(
    llm, model_name: str, samples: list[str], img_b64: str,
    specialization_context: str, cache_key: str,
) -> str:
    """Multi-sample document handling (real /qa 2026-07-24 finding: a
    6-sample biopsy request was silently collapsed to one question). Verify
    EACH sample independently — same anti-hallucination bar as the
    single-item path, applied per item, not skipped for the list case. A
    sample that fails verification is dropped, not fatal to the whole
    request (mirrors how the media-group loop in telegram.py already
    tolerates individual photo failures without failing the batch). Returns
    VISION_UNCERTAIN if none verify.
    """
    async def _verify_sample(sample: str) -> str | None:
        verify_prompt = _VERIFY_PROMPT_TEMPLATE.format(claim=sample)
        if specialization_context:
            verify_prompt += _VERIFY_SPECIALIZATION_CAVEAT
        try:
            verification: VisionVerification = await _call_with_rate_limit_retry(
                _structured_or_json, llm, model_name, verify_prompt, img_b64,
                VisionVerification, _VERIFY_JSON_SUFFIX,
            )
        except Exception as exc:
            logger.warning("vision_sample_verification_failed sample_len=%d err=%s", len(sample), exc)
            return None
        if verification.text_visible:
            return sample
        logger.warning("vision_sample_verification_rejected sample_len=%d", len(sample))
        return None

    samples_to_verify = samples[:_MAX_MULTI_SAMPLE_VERIFY]
    if len(samples) > _MAX_MULTI_SAMPLE_VERIFY:
        logger.warning(
            "vision_multi_sample_truncated total=%d cap=%d",
            len(samples), _MAX_MULTI_SAMPLE_VERIFY,
        )

    # Concurrent, not sequential (found in /code-review) — each sample's
    # verification is an independent LLM call with no shared state that
    # requires ordering, so N samples cost one round-trip's latency instead
    # of N.
    results = await asyncio.gather(*(_verify_sample(s) for s in samples_to_verify))
    verified_samples = [s for s in results if s is not None]

    if not verified_samples:
        # Deliberately NOT cached — same stochastic-rejection reasoning as
        # the single-item path.
        return VISION_UNCERTAIN

    # Vertical-neutral phrasing (found in /code-review) — extract_procedure_query
    # is documented as vertical-agnostic (medical order, clothing tag, shoe
    # box...), so the combined question must not hardcode lab-specific
    # "orden"/"muestras" wording for a menu or product-shelf photo that also
    # happens to show 2+ distinct items.
    combined_question = "Veo varios ítems distintos en la imagen, ¿cuánto cuesta cada uno?\n" + "\n".join(
        f"- {s}" for s in verified_samples
    )

    if len(verified_samples) < len(samples_to_verify):
        # Partial result — one or more samples failed verification, which is
        # documented as stochastic (identical image+claim pairs flip between
        # accepted/rejected across independent calls). Caching an incomplete
        # list as if it were final would permanently shortchange every
        # resend of this photo (found in /code-review) — only a
        # fully-verified answer is cached.
        logger.warning(
            "vision_multi_sample_partial verified=%d total=%d",
            len(verified_samples), len(samples),
        )
        return combined_question

    await _store_vision_result(cache_key, combined_question)
    return combined_question


async def _get_cached_result_or_none(cache_key: str) -> str | None:
    """A cache-read failure degrades to a cache miss, not an error — the
    extraction path below is the safe fallback either way."""
    try:
        return await _get_cached_vision_result(cache_key)
    except Exception as exc:
        logger.warning("vision_cache_read_failed err=%s treating as cache miss", exc)
        return None


async def _finalize_single_item(
    llm, model_name: str, extraction: VisionExtraction, specialization_context: str,
    img_b64: str, cache_key: str,
) -> str:
    """Verify the bare literal term, not the formatted question — a document
    never contains the full interrogative sentence, only the term itself.
    Returns VISION_UNCERTAIN (uncached) on any failure or rejection — see
    extract_procedure_query's docstring for why rejections are never cached.
    """
    verify_claim = extraction.procedure_name or extraction.price_question
    verify_prompt = _VERIFY_PROMPT_TEMPLATE.format(claim=verify_claim)
    if specialization_context:
        verify_prompt += _VERIFY_SPECIALIZATION_CAVEAT
    try:
        verification: VisionVerification = await _call_with_rate_limit_retry(
            _structured_or_json, llm, model_name, verify_prompt, img_b64, VisionVerification, _VERIFY_JSON_SUFFIX
        )
    except Exception as exc:
        logger.warning("vision_verification_failed=%s defaulting to uncertain", exc)
        return VISION_UNCERTAIN  # transient — not cached, retry may succeed

    if not verification.text_visible:
        logger.warning("vision_verification_rejected claim_len=%d", len(verify_claim))
        # Deliberately NOT cached — proven stochastic (see docstring); one
        # unlucky roll must not become a permanent wrong answer.
        return VISION_UNCERTAIN

    await _store_vision_result(cache_key, extraction.price_question)
    return extraction.price_question


async def extract_procedure_query(
    img_bytes: bytes, caption: str,
    tenant_slug: str = "", specialization_context: str = "",
) -> str:
    """Vision-transcribe a document/product image into a literal price question.

    Shared by every channel (Telegram, WhatsApp, ...) that accepts image uploads,
    so the anti-hallucination prompt (literal transcription, explicit uncertainty
    signal) stays identical across channels instead of drifting. Vertical-agnostic
    by design — the vision model reads whatever business context the photo shows
    (medical order, clothing tag, shoe box...) without needing per-tenant wording.

    Two explicit verification layers, added after a live test showed the
    model confidently fabricating a procedure name on a blank image instead
    of flagging uncertainty:

    1. is_legible is a structured boolean field the model must set, not a
       sentinel string buried in free text it can choose to skip.
    2. Even with that field, the model still asserted is_legible=true on a
       blank image in roughly half of live runs — a self-report isn't
       trustworthy on its own. So a second, independent call re-examines the
       SAME image and is asked to specifically confirm or reject the first
       call's claimed text, rather than just re-stating confidence in it.
       This second pass verifies extraction.procedure_name — the bare literal
       term (e.g. "IGRA") — NOT price_question (the formatted customer-facing
       question, e.g. "¿Cuánto cuesta un examen de IGRA?"). A live A/B test
       found verifying the full formatted question fails almost always (0/10
       trials on a clearly legible image) because that sentence never
       literally appears in the source document — only the bare term does
       (6/8 trials passed when verifying the bare term instead).

    Any failure at any stage — API error, malformed output, is_legible=false,
    or the second pass rejecting the claim — resolves to VISION_UNCERTAIN.
    The safe default here is "ask the user", never a guess.

    Results are cached by hash(model, caption, image bytes) — the same photo
    resent (users routinely retake/reforward the same order) skips straight to
    the cached answer instead of re-paying two LLM calls. Only stable content
    judgments are cached: is_legible=false, and a verified extraction. A
    verification REJECTION is deliberately never cached — the same live A/B
    test showed identical (image, claim) pairs flip between accepted/rejected
    across independent calls, so caching one bad roll would make it a
    permanent wrong answer for that exact photo instead of a fresh roll on
    the next resend. API errors are never cached either, since a retry might
    succeed once the upstream recovers.

    Each stage retries with backoff specifically on 429 (rate limited) —
    a real chance under photo bursts — and fails fast on anything else.

    The image is preprocessed first (EXIF rotation corrected, downscaled if
    oversized) — see _preprocess_image.
    """
    if not settings.openai_vision_model:
        raise RuntimeError("OPENAI_VISION_MODEL not configured")

    img_bytes = _preprocess_image(img_bytes)
    model_name = settings.openai_vision_model
    cache_key = _vision_cache_key(model_name, caption, img_bytes, tenant_slug, specialization_context)
    cached = await _get_cached_result_or_none(cache_key)
    if cached is not None:
        logger.info("vision_cache_hit key=%s", cache_key[:12])
        return cached

    llm = get_vision_llm()
    img_b64 = base64.b64encode(img_bytes).decode()
    base_prompt = f"{caption}\n\n{_VISION_EXTRACT_PROMPT}" if caption else _VISION_EXTRACT_PROMPT
    prompt = base_prompt
    if specialization_context:
        # Extraction prompt only — the verify pass below never receives the
        # actual jargon text (stays domain-agnostic, protecting the
        # anti-hallucination two-pass design), but DOES get a stricter
        # scrutiny instruction (_VERIFY_SPECIALIZATION_CAVEAT below) since a
        # biased extraction needs a higher bar to pass verification.
        prompt = f"Contexto del negocio: {specialization_context}\n\n{prompt}"
        logger.debug("vision_specialization_applied len=%d", len(specialization_context))

    try:
        extraction: VisionExtraction = await _call_with_rate_limit_retry(
            _structured_or_json, llm, model_name, prompt, img_b64, VisionExtraction, _EXTRACT_JSON_SUFFIX
        )
    except Exception as exc:
        logger.warning("vision_extraction_failed=%s defaulting to uncertain", exc)
        return VISION_UNCERTAIN  # transient — not cached, retry may succeed

    is_multi = bool(extraction.is_legible and extraction.samples and len(extraction.samples) >= 2)

    if not extraction.is_legible or (not extraction.price_question and not is_multi):
        # Vision uncertain; try OCR fallback if Tesseract available
        ocr_text = _extract_with_ocr(img_bytes) if _TESSERACT_AVAILABLE else None
        if ocr_text and len(ocr_text) > 3:
            logger.info("vision_uncertain fallback_to_ocr text_len=%d", len(ocr_text))
            extraction.procedure_name = ocr_text
            extraction.price_question = f"¿Cuánto cuesta {ocr_text}?"
            extraction.is_legible = True
            is_multi = False  # OCR fallback only ever produces one literal string
        else:
            await _store_vision_result(cache_key, VISION_UNCERTAIN)
            return VISION_UNCERTAIN
    elif specialization_context and not is_multi:
        # Only paid when specialization_context is set (the condition that
        # triggers the bias risk) and only on the confident direct-read path
        # — the OCR-fallback branch above is a different, non-LLM-biased
        # signal and doesn't need a second opinion. NOT applied to the
        # multi-sample case (real /investigate 2026-07-24 finding): requiring
        # exact agreement across N independently re-read items compounds —
        # the per-sample verification loop in _handle_multi_sample is a more
        # precise, more granular check for that case.
        if not await _run_consensus_check(llm, model_name, extraction, base_prompt, img_b64):
            return VISION_UNCERTAIN

    if is_multi:
        return await _handle_multi_sample(
            llm, model_name, extraction.samples, img_b64, specialization_context, cache_key
        )

    return await _finalize_single_item(llm, model_name, extraction, specialization_context, img_b64, cache_key)
