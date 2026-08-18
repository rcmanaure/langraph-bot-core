"""The inbound turn: one pass from a received message to a delivered reply.

Channel-agnostic by design. Everything here was previously written once per
channel and drifted — the empty-answer fallback existed only on Telegram, the
PDF reply only on WhatsApp, and the two error strings differed by a sentence.
If behaviour needs to differ per channel and the platform doesn't force it,
that is a defect here, not a feature of the channel.

What genuinely varies stays behind ChannelAdapter (app/channels/base.py).
"""
import logging

from langchain_core.messages import HumanMessage

from app.channels.base import ChannelAdapter, Inbound, MediaRef, MediaTooLarge
from app.channels.silent import SilentAdapter
from app.config import MAX_MEDIA_BYTES, settings
from app.messages import HUMAN_HANDOFF
from app.services.human_control import is_under_human_control, record_message
from app.services.redaction import redact_document_numbers
from app.services.staff import resolve_staff
from app.services.tenant_context import get_tenant_specialization
from app.services.vision import VISION_UNCERTAIN, extract_procedure_query

logger = logging.getLogger(__name__)

# Every user-facing string of the turn, in one place. Previously duplicated
# across telegram.py and whatsapp.py, where two of them had already diverged.
AUDIO_TOO_LARGE = "Archivo de voz demasiado grande (máx 10MB)."
IMAGE_TOO_LARGE = "Imagen demasiado grande (máx 10MB)."
STT_DISABLED = "La transcripción de audio no está habilitada."
STT_FAILED = "No pude procesar su nota de voz. ¿Puede escribirme su consulta?"
STT_EMPTY = "No escuché nada en el audio. ¿Puede repetirlo o escribirme?"
VISION_DISABLED = "El análisis de imágenes no está habilitado."
VISION_FAILED = "No pude procesar la imagen. Por favor intente de nuevo."
VISION_UNSURE = (
    "No pude leer con seguridad el examen en la imagen. Intente con una foto "
    "más clara: buena luz, enfocada, y que se vea toda la hoja. O si prefiere, "
    "puede escribirme el nombre del examen o procedimiento."
)
DOCUMENT_UNSUPPORTED = (
    "Por ahora no puedo leer documentos PDF. ¿Puede enviarme una foto del examen?"
)
SERVICE_UNAVAILABLE = "Lo siento, el servicio no está disponible. Por favor intente de nuevo más tarde."
GRAPH_ERROR = "Lo siento, ocurrió un error. Por favor intente de nuevo."
EMPTY_ANSWER = "Lo siento, no pude generar una respuesta."
# Sent for every cause of an escalation (see docs/adr/ADR-009-human-control.md)
# — the user doesn't need to know whether they asked for a person or the bot
# ran out of things to say. Shared with generate.py's automatic-escalation
# path (app/messages.py) rather than defined twice.


async def run_turn(adapter: ChannelAdapter, inbound: Inbound, graph) -> None:
    """Acknowledge, resolve the message to text, invoke the graph, reply.

    Never raises: this runs as a background task after the webhook route has
    already returned 200, so an escaping exception would be invisible to the
    platform and silent to the user.
    """
    # Ack first, unconditionally -- it's identical whether or not the thread
    # is under human control (SilentAdapter delegates it unchanged), so it
    # must not wait on a DB round trip to reach the user.
    await _acknowledge(adapter, inbound)

    under_control = await is_under_human_control(inbound.tenant_slug, inbound.channel, inbound.user_id)
    if under_control:
        # The bot goes silent for the rest of this turn -- media is still
        # fetched/transcribed/extracted so the operator reads text, not a
        # placeholder, but nothing reaches the user. See ADR-009.
        adapter = SilentAdapter(adapter)

    text = await _resolve_text(adapter, inbound)
    if text is None:
        return  # nothing to answer, or the user was already told why
    # Masked before the text becomes part of persisted graph state or a
    # conversation audit record — see app/services/redaction.py.
    text = redact_document_numbers(text)

    if under_control:
        await record_message(inbound.tenant_slug, inbound.thread_id, "user", text)
        return  # the graph is never invoked while a thread is under human control

    await _reply(adapter, inbound, text, graph)


async def _acknowledge(adapter: ChannelAdapter, inbound: Inbound) -> None:
    """Feedback before any download/STT/vision work — those take seconds and
    the user should see a signal right away. Best-effort by definition."""
    try:
        await adapter.acknowledge(inbound)
    except Exception as exc:
        logger.warning("turn_acknowledge_failed channel=%s err=%s", inbound.channel, exc)


async def _resolve_text(adapter: ChannelAdapter, inbound: Inbound) -> str | None:
    """The message as text the graph can answer. None means the turn stops —
    either there was nothing to answer, or the user has been told why."""
    if inbound.text:
        return inbound.text
    if not inbound.media:
        return None

    kind = inbound.media[0].kind
    if kind == "document":
        await _send(adapter, inbound, DOCUMENT_UNSUPPORTED)
        return None
    if kind == "audio":
        return await _transcribe(adapter, inbound)
    return await _extract_images(adapter, inbound)


def _oversized(ref: MediaRef) -> bool:
    """An unknown size is not oversized — the channel didn't tell us, and
    refusing on that basis would reject legitimate media."""
    return ref.size_bytes is not None and ref.size_bytes > MAX_MEDIA_BYTES


async def _transcribe(adapter: ChannelAdapter, inbound: Inbound) -> str | None:
    from app.services.stt import STTNotConfiguredError, transcribe

    ref = inbound.media[0]
    if _oversized(ref):
        await _send(adapter, inbound, AUDIO_TOO_LARGE)
        return None

    try:
        audio = await adapter.fetch_media(ref)
        text = await transcribe(
            audio, ref.filename or "audio.ogg", ref.mime_type or "audio/ogg"
        )
    except MediaTooLarge:
        await _send(adapter, inbound, AUDIO_TOO_LARGE)
        return None
    except STTNotConfiguredError:
        logger.error(
            "turn_stt_not_configured channel=%s tenant=%s user=%s",
            inbound.channel, inbound.tenant_slug, inbound.user_id,
        )
        await _send(adapter, inbound, STT_DISABLED)
        return None
    except Exception as exc:
        logger.warning("turn_stt_failed channel=%s user=%s err=%s",
                       inbound.channel, inbound.user_id, exc)
        await _send(adapter, inbound, STT_FAILED)
        return None

    if not text:
        await _send(adapter, inbound, STT_EMPTY)
        return None
    return text


async def _extract_images(adapter: ChannelAdapter, inbound: Inbound) -> str | None:
    if not settings.openai_vision_model:
        await _send(adapter, inbound, VISION_DISABLED)
        return None

    refs = [ref for ref in inbound.media if not _oversized(ref)]
    if not refs:
        await _send(adapter, inbound, IMAGE_TOO_LARGE)
        return None
    single = len(refs) == 1

    # Looked up once for the whole batch, not per image — the tenant can't
    # change mid-batch, so this avoids N redundant DB round-trips.
    specialization = await get_tenant_specialization(inbound.tenant_slug)

    # Masked before it reaches the vision API, not just before it reaches
    # persisted state -- found in /code-review: a caption is free-form user
    # text same as any other message, and this path used to send it to a
    # third-party model unredacted.
    caption = redact_document_numbers(inbound.caption)

    queries: list[str] = []
    for ref in refs:
        try:
            img_bytes = await adapter.fetch_media(ref)
            query = await extract_procedure_query(
                img_bytes, caption,
                tenant_slug=inbound.tenant_slug,
                specialization_context=specialization,
            )
        except MediaTooLarge:
            if single:
                await _send(adapter, inbound, IMAGE_TOO_LARGE)
                return None
            continue  # one oversized photo must not fail a whole batch
        except Exception as exc:
            logger.warning("turn_vision_failed channel=%s user=%s err=%s",
                           inbound.channel, inbound.user_id, exc)
            if single:
                await _send(adapter, inbound, VISION_FAILED)
                return None
            continue  # one bad photo must not fail a whole batch
        if VISION_UNCERTAIN not in query:
            queries.append(query)

    if not queries:
        # Don't guess and forward an uncertain read into the RAG pipeline — a
        # wrong procedure name there looks exactly like a confident, correct
        # answer downstream. Ask the user to type it instead.
        logger.warning("turn_vision_uncertain channel=%s tenant=%s count=%d",
                       inbound.channel, inbound.tenant_slug, len(refs))
        await _send(adapter, inbound, VISION_UNSURE)
        return None

    logger.warning("turn_vision_extracted channel=%s tenant=%s count=%d",
                   inbound.channel, inbound.tenant_slug, len(queries))
    if len(queries) == 1:
        return queries[0]
    # A query that already spans multiple lines (vision.py's multi-sample
    # combined_question — one photo listing several distinct items) is
    # spliced in as-is, not re-bulleted, or its own header and sub-bullets
    # would nest inside a single outer "- " bullet.
    return "\n".join(q if "\n" in q else f"- {q}" for q in queries)


async def _reply(adapter: ChannelAdapter, inbound: Inbound, text: str, graph) -> None:
    if graph is None:
        logger.error("turn_graph_not_initialized thread=%s", inbound.thread_id)
        await _send(adapter, inbound, SERVICE_UNAVAILABLE)
        return

    # Resolved once per turn, here — the only place the graph is invoked —
    # from tenant/channel/user id alone. Never from anything in `text`: a
    # message claiming staff status in prose grants nothing (see ADR-006).
    is_staff = await resolve_staff(inbound.tenant_slug, inbound.channel, inbound.user_id)

    try:
        result = await graph.ainvoke(
            {
                "tenant_id": inbound.tenant_slug,
                "thread_id": inbound.thread_id,
                "messages": [HumanMessage(content=text)],
                "answer": "",
                "is_staff": is_staff,
            },
            config={"configurable": {"thread_id": inbound.thread_id}},
        )
        if result.get("__interrupt__"):
            # The graph suspended (interrupt_node) rather than finishing —
            # "answer" is whatever it was before the suspended node ran, not
            # a genuinely empty reply. Distinguish it from EMPTY_ANSWER.
            answer = HUMAN_HANDOFF
        else:
            answer = result.get("answer") or ""
            if not answer and result.get("messages"):
                answer = result["messages"][-1].content
            if not answer:
                answer = EMPTY_ANSWER
    except Exception:
        logger.exception("turn_graph_failed thread=%s", inbound.thread_id)
        answer = GRAPH_ERROR

    await _send(adapter, inbound, answer)


async def _send(adapter: ChannelAdapter, inbound: Inbound, text: str) -> None:
    """A delivery failure is logged, never raised — the turn runs after the
    webhook already returned 200, so there is nobody left to report it to."""
    try:
        await adapter.send(inbound, text)
    except Exception as exc:
        logger.warning("turn_send_failed channel=%s chat=%s err=%s",
                       inbound.channel, inbound.chat_id, exc, exc_info=True)
