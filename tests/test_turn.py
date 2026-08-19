"""Tests for the inbound turn (app/channels/turn.py).

The turn is channel-agnostic, so it is tested once here through a FakeAdapter
rather than once per channel. Channel-specific behaviour — payload parsing,
auth, Telegram's album buffering — is tested in the per-channel files.

FakeAdapter is the second adapter that makes the seam real: two in production
(Telegram, WhatsApp) plus this one in tests.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.channels import turn as turn_module
from app.channels.base import Inbound, MediaRef, MediaTooLarge
from app.channels.turn import run_turn

VISION_MODEL = "app.channels.turn.settings.openai_vision_model"
EXTRACT = "app.channels.turn.extract_procedure_query"
SPECIALIZATION = "app.channels.turn.get_tenant_specialization"
TRANSCRIBE = "app.services.stt.transcribe"


class FakeAdapter:
    """In-memory ChannelAdapter. Records what the turn asked it to do."""

    channel = "fake"

    def __init__(self, media: bytes | Exception = b"media-bytes") -> None:
        self.sent: list[str] = []
        self.acknowledged = 0
        self.fetched: list[MediaRef] = []
        self._media = media
        self.acknowledge_error: Exception | None = None
        self.send_error: Exception | None = None

    async def verify(self, request) -> bool:
        return True

    def dedup_key(self, body: dict) -> str | None:
        return body.get("id")

    async def parse(self, body: dict) -> list[Inbound]:
        return []

    async def acknowledge(self, inbound: Inbound) -> None:
        if self.acknowledge_error:
            raise self.acknowledge_error
        self.acknowledged += 1

    async def fetch_media(self, ref: MediaRef) -> bytes:
        self.fetched.append(ref)
        if isinstance(self._media, Exception):
            raise self._media
        return self._media

    async def send(self, inbound: Inbound, text: str) -> bool:
        if self.send_error:
            raise self.send_error
        self.sent.append(text)
        return True


def make_inbound(**overrides) -> Inbound:
    base = {
        "tenant_slug": "demo",
        "channel": "fake",
        "user_id": "42",
        "chat_id": "100",
        "message_id": "7",
    }
    return Inbound(**{**base, **overrides})


def image(size: int | None = 1024) -> MediaRef:
    return MediaRef(id="img-1", kind="image", size_bytes=size)


def audio(size: int | None = 1024) -> MediaRef:
    return MediaRef(id="aud-1", kind="audio", size_bytes=size,
                    mime_type="audio/ogg", filename="voice.ogg")


@pytest.fixture()
def graph():
    g = AsyncMock()
    g.ainvoke = AsyncMock(return_value={"answer": "Respuesta OK", "messages": []})
    return g


@pytest.fixture()
def vision_on():
    with patch(VISION_MODEL, "vision-model"), \
         patch(SPECIALIZATION, new_callable=AsyncMock, return_value=""):
        yield


RESOLVE_STAFF = "app.channels.turn.resolve_staff"
UNDER_CONTROL = "app.channels.turn.is_under_human_control"
RECORD_MESSAGE = "app.channels.turn.record_message"


@pytest.fixture()
def under_human_control():
    with patch(UNDER_CONTROL, new_callable=AsyncMock, return_value=True), \
         patch(RECORD_MESSAGE, new_callable=AsyncMock) as record:
        yield record


# ── Text ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_text_reaches_the_graph_and_the_answer_is_sent(graph):
    adapter = FakeAdapter()
    await run_turn(adapter, make_inbound(text="hola"), graph)

    graph.ainvoke.assert_awaited_once()
    assert graph.ainvoke.call_args[0][0]["messages"][0].content == "hola"
    assert adapter.sent == ["Respuesta OK"]


@pytest.mark.asyncio
async def test_thread_id_identifies_tenant_user_and_channel(graph):
    await run_turn(FakeAdapter(), make_inbound(text="hola"), graph)

    expected = "tenant:demo:user:42:channel:fake"
    assert graph.ainvoke.call_args[0][0]["thread_id"] == expected
    assert graph.ainvoke.call_args[1]["config"]["configurable"]["thread_id"] == expected


@pytest.mark.asyncio
async def test_nothing_to_answer_stops_before_the_graph(graph):
    adapter = FakeAdapter()
    await run_turn(adapter, make_inbound(), graph)

    graph.ainvoke.assert_not_awaited()
    assert adapter.sent == []


# ── Document number redaction ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_document_number_is_redacted_before_reaching_the_graph(graph):
    await run_turn(FakeAdapter(), make_inbound(text="mi cédula es 12345678"), graph)

    content = graph.ainvoke.call_args[0][0]["messages"][0].content
    assert "12345678" not in content
    assert "[documento]" in content


@pytest.mark.asyncio
async def test_ordinary_message_reaches_the_graph_unmodified(graph):
    await run_turn(FakeAdapter(), make_inbound(text="cuánto cuesta la biopsia"), graph)

    assert graph.ainvoke.call_args[0][0]["messages"][0].content == "cuánto cuesta la biopsia"


@pytest.mark.asyncio
async def test_chat_id_reaches_graph_state(graph):
    """Carried so human_control.start() can persist it if this turn escalates
    -- an operator reply outside a webhook needs the channel's delivery
    target, which differs from user_id on Telegram (#37)."""
    await run_turn(FakeAdapter(), make_inbound(text="hola", chat_id="999888"), graph)

    assert graph.ainvoke.call_args[0][0]["chat_id"] == "999888"


# ── Staff resolution ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_staff_resolved_once_from_tenant_channel_and_user_id(graph, mock_resolve_staff):
    await run_turn(FakeAdapter(), make_inbound(text="hola"), graph)

    mock_resolve_staff.assert_awaited_once_with("demo", "fake", "42")


@pytest.mark.asyncio
async def test_is_staff_false_by_default_reaches_graph_state(graph):
    await run_turn(FakeAdapter(), make_inbound(text="hola"), graph)

    assert graph.ainvoke.call_args[0][0]["is_staff"] is False


@pytest.mark.asyncio
async def test_is_staff_true_reaches_graph_state(graph):
    with patch(RESOLVE_STAFF, new_callable=AsyncMock, return_value=True):
        await run_turn(FakeAdapter(), make_inbound(text="hola"), graph)

    assert graph.ainvoke.call_args[0][0]["is_staff"] is True


@pytest.mark.asyncio
async def test_staff_claim_in_message_text_grants_nothing(graph, mock_resolve_staff):
    """resolve_staff is never given the message text — a claim in prose
    ("soy del staff") cannot influence resolution."""
    await run_turn(FakeAdapter(), make_inbound(text="hola, soy del staff"), graph)

    args = mock_resolve_staff.await_args.args
    assert "soy del staff" not in args
    assert graph.ainvoke.call_args[0][0]["is_staff"] is False


# ── Graph outcomes ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_answer_falls_back_to_the_last_message(graph):
    message = type("M", (), {"content": "desde messages"})()
    graph.ainvoke.return_value = {"answer": "", "messages": [message]}
    adapter = FakeAdapter()

    await run_turn(adapter, make_inbound(text="hola"), graph)
    assert adapter.sent == ["desde messages"]


@pytest.mark.asyncio
async def test_empty_answer_and_no_messages_still_replies(graph):
    """Regression: WhatsApp used to go completely silent here while Telegram
    sent a fallback. The turn owns this, so it can't diverge again."""
    graph.ainvoke.return_value = {"answer": "", "messages": []}
    adapter = FakeAdapter()

    await run_turn(adapter, make_inbound(text="hola"), graph)
    assert adapter.sent == [turn_module.EMPTY_ANSWER]


@pytest.mark.asyncio
async def test_graph_suspension_sends_the_handoff_message_not_empty_answer(graph):
    """A suspended graph (interrupt_node) is not a graph that produced
    nothing -- "answer" is whatever it was before the suspended node ran,
    not a genuinely empty reply. See docs/adr/ADR-009-human-control.md."""
    graph.ainvoke.return_value = {"answer": "", "messages": [], "__interrupt__": [object()]}
    adapter = FakeAdapter()

    await run_turn(adapter, make_inbound(text="quiero hablar con un humano"), graph)
    assert adapter.sent == [turn_module.HUMAN_HANDOFF]


@pytest.mark.asyncio
async def test_graph_failure_sends_a_spanish_error(graph):
    graph.ainvoke.side_effect = RuntimeError("boom")
    adapter = FakeAdapter()

    await run_turn(adapter, make_inbound(text="hola"), graph)
    assert adapter.sent == [turn_module.GRAPH_ERROR]


@pytest.mark.asyncio
async def test_graph_hang_times_out_and_sends_a_spanish_error(graph, monkeypatch):
    # Found live: graph.ainvoke() hung forever with no exception -- a stuck
    # coroutine must not leave the user staring at "typing..." indefinitely.
    monkeypatch.setattr(turn_module, "_GRAPH_TIMEOUT_SECONDS", 0.05)

    async def _hang(*args, **kwargs):
        await asyncio.sleep(10)

    graph.ainvoke.side_effect = _hang
    adapter = FakeAdapter()

    await run_turn(adapter, make_inbound(text="hola"), graph)
    assert adapter.sent == [turn_module.GRAPH_ERROR]


@pytest.mark.asyncio
async def test_missing_graph_sends_service_unavailable():
    adapter = FakeAdapter()
    await run_turn(adapter, make_inbound(text="hola"), None)
    assert adapter.sent == [turn_module.SERVICE_UNAVAILABLE]


# ── Human control ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_under_human_control_text_is_recorded_not_sent_graph_not_invoked(graph, under_human_control):
    adapter = FakeAdapter()
    await run_turn(adapter, make_inbound(text="sigo esperando"), graph)

    assert adapter.sent == []
    graph.ainvoke.assert_not_awaited()
    under_human_control.assert_awaited_once_with(
        "demo", "tenant:demo:user:42:channel:fake", "user", "sigo esperando"
    )


@pytest.mark.asyncio
async def test_under_human_control_document_number_is_masked_before_recording(graph, under_human_control):
    await run_turn(FakeAdapter(), make_inbound(text="mi cédula es 12345678"), graph)

    content = under_human_control.await_args.args[3]
    assert "12345678" not in content
    assert "[documento]" in content


@pytest.mark.asyncio
async def test_under_human_control_audio_is_transcribed_and_recorded_not_sent(graph, under_human_control):
    adapter = FakeAdapter()
    with patch(TRANSCRIBE, new_callable=AsyncMock, return_value="tengo dudas sobre el resultado"):
        await run_turn(adapter, make_inbound(media=[audio()]), graph)

    assert adapter.sent == []
    graph.ainvoke.assert_not_awaited()
    assert len(adapter.fetched) == 1  # media still fetched -- operator reads text, not a placeholder
    under_human_control.assert_awaited_once_with(
        "demo", "tenant:demo:user:42:channel:fake", "user", "tengo dudas sobre el resultado"
    )


@pytest.mark.asyncio
async def test_under_human_control_image_is_extracted_and_recorded_not_sent(graph, under_human_control, vision_on):
    adapter = FakeAdapter()
    with patch(EXTRACT, new_callable=AsyncMock, return_value="¿Cuánto cuesta un examen de IGRA?"):
        await run_turn(adapter, make_inbound(media=[image()]), graph)

    assert adapter.sent == []
    graph.ainvoke.assert_not_awaited()
    assert len(adapter.fetched) == 1
    under_human_control.assert_awaited_once_with(
        "demo", "tenant:demo:user:42:channel:fake", "user", "¿Cuánto cuesta un examen de IGRA?"
    )


@pytest.mark.asyncio
async def test_under_human_control_unsupported_document_sends_nothing(graph, under_human_control):
    adapter = FakeAdapter()
    document = MediaRef(id="doc-1", kind="document", mime_type="application/pdf")

    await run_turn(adapter, make_inbound(media=[document]), graph)

    assert adapter.sent == []
    graph.ainvoke.assert_not_awaited()
    under_human_control.assert_not_awaited()  # nothing resolved to text -- nothing to record


@pytest.mark.asyncio
async def test_under_human_control_oversized_audio_sends_nothing(graph, under_human_control):
    adapter = FakeAdapter()
    await run_turn(adapter, make_inbound(media=[audio(11 * 1024 * 1024)]), graph)

    assert adapter.sent == []
    assert adapter.fetched == []
    graph.ainvoke.assert_not_awaited()
    under_human_control.assert_not_awaited()


@pytest.mark.asyncio
async def test_under_human_control_failed_transcription_sends_nothing(graph, under_human_control):
    adapter = FakeAdapter()
    with patch(TRANSCRIBE, new_callable=AsyncMock, side_effect=RuntimeError("whisper down")):
        await run_turn(adapter, make_inbound(media=[audio()]), graph)

    assert adapter.sent == []
    graph.ainvoke.assert_not_awaited()
    under_human_control.assert_not_awaited()


@pytest.mark.asyncio
async def test_under_human_control_uncertain_vision_sends_nothing(graph, under_human_control, vision_on):
    adapter = FakeAdapter()
    with patch(EXTRACT, new_callable=AsyncMock, return_value=turn_module.VISION_UNCERTAIN):
        await run_turn(adapter, make_inbound(media=[image()]), graph)

    assert adapter.sent == []
    graph.ainvoke.assert_not_awaited()
    under_human_control.assert_not_awaited()


@pytest.mark.asyncio
async def test_not_under_human_control_replies_as_normal(graph):
    """Regression guard for the default (mock_under_human_control returns
    False): everything above must not fire when a thread isn't escalated."""
    adapter = FakeAdapter()
    await run_turn(adapter, make_inbound(text="hola"), graph)

    assert adapter.sent == ["Respuesta OK"]
    graph.ainvoke.assert_awaited_once()


# ── Best-effort side channels ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_acknowledge_runs_before_any_media_fetch(graph, vision_on):
    adapter = FakeAdapter()
    order: list[str] = []

    async def record_ack(inbound):
        order.append("ack")

    async def record_fetch(ref):
        order.append("fetch")
        return b"img"

    adapter.acknowledge = record_ack
    adapter.fetch_media = record_fetch

    with patch(EXTRACT, new_callable=AsyncMock, return_value="¿Cuánto cuesta X?"):
        await run_turn(adapter, make_inbound(media=[image()]), graph)

    assert order == ["ack", "fetch"]


@pytest.mark.asyncio
async def test_acknowledge_failure_does_not_stop_the_turn(graph):
    adapter = FakeAdapter()
    adapter.acknowledge_error = RuntimeError("typing indicator down")

    await run_turn(adapter, make_inbound(text="hola"), graph)
    assert adapter.sent == ["Respuesta OK"]


@pytest.mark.asyncio
async def test_send_failure_never_escapes_the_turn(graph):
    """The turn runs after the webhook already returned 200 — an exception
    here would be invisible to the platform and silent to the user."""
    adapter = FakeAdapter()
    adapter.send_error = RuntimeError("network down")

    await run_turn(adapter, make_inbound(text="hola"), graph)  # must not raise


# ── Audio ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audio_is_transcribed_and_reaches_the_graph(graph):
    adapter = FakeAdapter()
    with patch(TRANSCRIBE, new_callable=AsyncMock, return_value="cuanto cuesta la biopsia"):
        await run_turn(adapter, make_inbound(media=[audio()]), graph)

    assert graph.ainvoke.call_args[0][0]["messages"][0].content == "cuanto cuesta la biopsia"


@pytest.mark.asyncio
async def test_audio_uses_the_filename_and_mime_the_adapter_supplied(graph):
    with patch(TRANSCRIBE, new_callable=AsyncMock, return_value="hola") as transcribe:
        ref = MediaRef(id="a", kind="audio", size_bytes=10,
                       mime_type="audio/mpeg", filename="audio.mp3")
        await run_turn(FakeAdapter(), make_inbound(media=[ref]), graph)

    assert transcribe.await_args[0][1:] == ("audio.mp3", "audio/mpeg")


@pytest.mark.asyncio
async def test_oversized_audio_is_rejected_before_download(graph):
    adapter = FakeAdapter()
    await run_turn(adapter, make_inbound(media=[audio(11 * 1024 * 1024)]), graph)

    assert adapter.fetched == []
    assert adapter.sent == [turn_module.AUDIO_TOO_LARGE]
    graph.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_media_size_is_not_treated_as_oversized(graph):
    adapter = FakeAdapter()
    with patch(TRANSCRIBE, new_callable=AsyncMock, return_value="hola"):
        await run_turn(adapter, make_inbound(media=[audio(None)]), graph)

    assert len(adapter.fetched) == 1
    graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_stt_not_configured_tells_the_user_the_feature_is_off(graph):
    from app.services.stt import STTNotConfiguredError

    adapter = FakeAdapter()
    with patch(TRANSCRIBE, new_callable=AsyncMock, side_effect=STTNotConfiguredError()):
        await run_turn(adapter, make_inbound(media=[audio()]), graph)

    assert adapter.sent == [turn_module.STT_DISABLED]
    graph.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_stt_failure_asks_the_user_to_type(graph):
    adapter = FakeAdapter()
    with patch(TRANSCRIBE, new_callable=AsyncMock, side_effect=RuntimeError("whisper down")):
        await run_turn(adapter, make_inbound(media=[audio()]), graph)

    assert adapter.sent == [turn_module.STT_FAILED]


@pytest.mark.asyncio
async def test_empty_transcription_asks_the_user_to_repeat(graph):
    adapter = FakeAdapter()
    with patch(TRANSCRIBE, new_callable=AsyncMock, return_value=""):
        await run_turn(adapter, make_inbound(media=[audio()]), graph)

    assert adapter.sent == [turn_module.STT_EMPTY]
    graph.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_download_failure_on_audio_asks_the_user_to_type(graph):
    adapter = FakeAdapter(media=RuntimeError("download failed"))
    with patch(TRANSCRIBE, new_callable=AsyncMock):
        await run_turn(adapter, make_inbound(media=[audio()]), graph)

    assert adapter.sent == [turn_module.STT_FAILED]


@pytest.mark.asyncio
async def test_media_too_large_raised_by_fetch_sends_the_audio_size_message(graph):
    """A late-discovered size (unknown until the adapter's fetch, e.g.
    WhatsApp) must reach the user worded identically to the upfront gate."""
    adapter = FakeAdapter(media=MediaTooLarge())
    await run_turn(adapter, make_inbound(media=[audio()]), graph)

    assert adapter.sent == [turn_module.AUDIO_TOO_LARGE]
    graph.ainvoke.assert_not_awaited()


# ── Images ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extracted_question_reaches_the_graph(graph, vision_on):
    adapter = FakeAdapter()
    with patch(EXTRACT, new_callable=AsyncMock, return_value="¿Cuánto cuesta un examen de IGRA?"):
        await run_turn(adapter, make_inbound(media=[image()], caption="hola"), graph)

    assert graph.ainvoke.call_args[0][0]["messages"][0].content == "¿Cuánto cuesta un examen de IGRA?"


@pytest.mark.asyncio
async def test_caption_and_tenant_context_are_passed_to_extraction(graph):
    with patch(VISION_MODEL, "vision-model"), \
         patch(SPECIALIZATION, new_callable=AsyncMock, return_value="laboratorio"), \
         patch(EXTRACT, new_callable=AsyncMock, return_value="¿Cuánto cuesta X?") as extract:
        await run_turn(FakeAdapter(), make_inbound(media=[image()], caption="mirá esto"), graph)

    assert extract.await_args[0][1] == "mirá esto"
    assert extract.await_args[1] == {"tenant_slug": "demo", "specialization_context": "laboratorio"}


@pytest.mark.asyncio
async def test_document_number_in_caption_is_redacted_before_vision_call(graph, vision_on):
    """Found in /code-review: a caption is free-form user text same as any
    other message, but it used to reach the third-party vision API
    unredacted -- only the graph-bound text was masked."""
    with patch(EXTRACT, new_callable=AsyncMock, return_value="¿Cuánto cuesta X?") as extract:
        await run_turn(
            FakeAdapter(),
            make_inbound(media=[image()], caption="mi cédula es 12345678"),
            graph,
        )

    assert "12345678" not in extract.await_args[0][1]


@pytest.mark.asyncio
async def test_vision_disabled_sends_a_notice(graph):
    adapter = FakeAdapter()
    with patch(VISION_MODEL, ""):
        await run_turn(adapter, make_inbound(media=[image()]), graph)

    assert adapter.sent == [turn_module.VISION_DISABLED]
    assert adapter.fetched == []
    graph.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_oversized_image_is_rejected_before_download(graph, vision_on):
    adapter = FakeAdapter()
    await run_turn(adapter, make_inbound(media=[image(11 * 1024 * 1024)]), graph)

    assert adapter.fetched == []
    assert adapter.sent == [turn_module.IMAGE_TOO_LARGE]


@pytest.mark.asyncio
async def test_uncertain_extraction_asks_the_user_to_type(graph, vision_on):
    """Never forward a guessed procedure name into the RAG pipeline — a misread
    exam becomes a confidently wrong price downstream."""
    adapter = FakeAdapter()
    with patch(EXTRACT, new_callable=AsyncMock, return_value=turn_module.VISION_UNCERTAIN):
        await run_turn(adapter, make_inbound(media=[image()]), graph)

    graph.ainvoke.assert_not_awaited()
    assert adapter.sent == [turn_module.VISION_UNSURE]


@pytest.mark.asyncio
async def test_extraction_failure_on_a_single_image_sends_an_error(graph, vision_on):
    adapter = FakeAdapter()
    with patch(EXTRACT, new_callable=AsyncMock, side_effect=RuntimeError("vision 500")):
        await run_turn(adapter, make_inbound(media=[image()]), graph)

    graph.ainvoke.assert_not_awaited()
    assert adapter.sent == [turn_module.VISION_FAILED]


# ── Batched images (Telegram albums arrive here as N refs) ────────────────────

@pytest.mark.asyncio
async def test_batched_images_are_combined_into_one_turn(graph, vision_on):
    queries = iter(["¿Cuánto cuesta un examen de IGRA?", "¿Cuánto cuesta una biopsia?"])

    async def extract(img, caption, tenant_slug="", specialization_context=""):
        return next(queries)

    with patch(EXTRACT, side_effect=extract):
        await run_turn(FakeAdapter(), make_inbound(media=[image(), image()]), graph)

    graph.ainvoke.assert_awaited_once()
    content = graph.ainvoke.call_args[0][0]["messages"][0].content
    assert "- ¿Cuánto cuesta un examen de IGRA?" in content
    assert "- ¿Cuánto cuesta una biopsia?" in content


@pytest.mark.asyncio
async def test_multi_sample_query_is_spliced_in_not_re_bulleted(graph, vision_on):
    """A photo listing several items already comes back as a formatted
    multi-line answer — wrapping it in one more "- " would nest its header and
    sub-bullets inside a single outer bullet."""
    multi = ("Veo varios ítems distintos en la imagen, ¿cuánto cuesta cada uno?\n"
             "- Epiplón\n- Líquido peritoneal")
    queries = iter(["¿Cuánto cuesta un examen de IGRA?", multi])

    async def extract(img, caption, tenant_slug="", specialization_context=""):
        return next(queries)

    with patch(EXTRACT, side_effect=extract):
        await run_turn(FakeAdapter(), make_inbound(media=[image(), image()]), graph)

    content = graph.ainvoke.call_args[0][0]["messages"][0].content
    assert "- ¿Cuánto cuesta un examen de IGRA?" in content
    assert "- Veo varios ítems" not in content
    assert "- Epiplón" in content


@pytest.mark.asyncio
async def test_one_failing_image_does_not_fail_the_batch(graph, vision_on):
    results = iter([RuntimeError("vision 500"), "¿Cuánto cuesta una biopsia?"])

    async def extract(img, caption, tenant_slug="", specialization_context=""):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    adapter = FakeAdapter()
    with patch(EXTRACT, side_effect=extract):
        await run_turn(adapter, make_inbound(media=[image(), image()]), graph)

    graph.ainvoke.assert_awaited_once()
    assert graph.ainvoke.call_args[0][0]["messages"][0].content == "¿Cuánto cuesta una biopsia?"
    assert adapter.sent == ["Respuesta OK"]


@pytest.mark.asyncio
async def test_batch_where_every_image_is_uncertain_sends_one_message(graph, vision_on):
    adapter = FakeAdapter()
    with patch(EXTRACT, new_callable=AsyncMock, return_value=turn_module.VISION_UNCERTAIN):
        await run_turn(adapter, make_inbound(media=[image(), image()]), graph)

    graph.ainvoke.assert_not_awaited()
    assert adapter.sent == [turn_module.VISION_UNSURE]


@pytest.mark.asyncio
async def test_oversized_images_are_skipped_not_fatal_to_the_batch(graph, vision_on):
    adapter = FakeAdapter()
    with patch(EXTRACT, new_callable=AsyncMock, return_value="¿Cuánto cuesta X?"):
        await run_turn(
            adapter,
            make_inbound(media=[image(11 * 1024 * 1024), image()]),
            graph,
        )

    assert len(adapter.fetched) == 1
    graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_media_too_large_raised_by_fetch_sends_the_image_size_message(graph, vision_on):
    """A late-discovered size (unknown until the adapter's fetch, e.g.
    WhatsApp) must reach the user worded identically to the upfront gate."""
    adapter = FakeAdapter(media=MediaTooLarge())
    await run_turn(adapter, make_inbound(media=[image()]), graph)

    assert adapter.sent == [turn_module.IMAGE_TOO_LARGE]
    graph.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_too_large_on_one_batched_image_is_skipped_not_fatal(graph, vision_on):
    fetched: list[MediaRef] = []

    async def fetch(ref):
        fetched.append(ref)
        if len(fetched) == 1:
            raise MediaTooLarge()
        return b"img"

    adapter = FakeAdapter()
    adapter.fetch_media = fetch
    with patch(EXTRACT, new_callable=AsyncMock, return_value="¿Cuánto cuesta X?"):
        await run_turn(adapter, make_inbound(media=[image(), image()]), graph)

    assert len(fetched) == 2
    graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_tenant_specialization_is_looked_up_once_per_batch(graph):
    with patch(VISION_MODEL, "vision-model"), \
         patch(EXTRACT, new_callable=AsyncMock, return_value="¿Cuánto cuesta X?"), \
         patch(SPECIALIZATION, new_callable=AsyncMock, return_value="") as lookup:
        await run_turn(FakeAdapter(), make_inbound(media=[image(), image(), image()]), graph)

    lookup.assert_awaited_once()


# ── Documents ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_documents_get_an_explanation_instead_of_silence(graph):
    """Regression: Telegram used to ignore PDFs entirely while WhatsApp
    explained itself. Both channels can receive one."""
    adapter = FakeAdapter()
    document = MediaRef(id="doc-1", kind="document", mime_type="application/pdf")

    await run_turn(adapter, make_inbound(media=[document]), graph)

    assert adapter.sent == [turn_module.DOCUMENT_UNSUPPORTED]
    assert adapter.fetched == []
    graph.ainvoke.assert_not_awaited()
