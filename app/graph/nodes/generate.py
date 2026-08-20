import logging

from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langgraph.runtime import Runtime
from sqlalchemy import text

from app.config import settings
from app.db import AsyncSessionLocal
from app.graph.nodes.retrieve import (
    GREETING_PREFIX_RE,
    LOCATION_RE,
    is_bare_rejection,
    last_human_text,
)
from app.messages import HUMAN_HANDOFF
from app.models.tenant import DEFAULT_TONE_DESCRIPTION
from app.services import human_control
from app.services.llm import get_chat_llm
from app.services.rag import cap_chunks_to_tokens, token_counter
from app.state import AgentState

logger = logging.getLogger(__name__)

# Fixed floor injected into every generation prompt ahead of the tenant's own
# tone text — the tenant's {tone_description} only colours the voice inside
# this block, it can never remove or contradict it (see ADR-007).
_REGISTER_FLOOR = """\
Registro (fijo — el tono del negocio no lo cambia):
- Trate al usuario siempre de "usted", nunca de "tú" ni de "vos".
- Sin emojis.
- Sin diminutivos.
- Sin frase de apertura de relleno ("¡Claro!", "¡Por supuesto!") — responda directo.
- Nunca se dirija al usuario por su nombre de pila."""

# Item layout/emphasis/list rules shared by both paths. The total-length cap
# is NOT here — it's RAG-only (see _FORMAT_HINT below). A total-length cap on
# the catalog path directly contradicts "list every item, omit nothing" the
# moment the catalog has more than 4-5 lines' worth of items.
_ITEM_FORMAT_RULES = """
Formato (OBLIGATORIO — compatible WhatsApp/Telegram):
- Tono: {tone_description}.
- *negrita* con asteriscos simples para códigos y nombres de ítems.
- _cursiva_ con guiones bajos para notas o aclaraciones breves.
- Listas con guión (- item). Sin tablas, sin encabezados Markdown (##).
- Por ítem: - *CÓDIGO* Nombre: $precio"""

_FORMAT_HINT = _REGISTER_FLOOR + _ITEM_FORMAT_RULES + """
- BREVE: máximo 4-5 líneas en total. Sin párrafos largos."""

_CATALOG_FORMAT_HINT = _REGISTER_FLOOR + _ITEM_FORMAT_RULES

# Staff variant of the floor (#27) — same core (usted, no emoji, no
# diminutives, no first-name address) but terser: no courtesy/filler at all,
# not just no opener, since a staff conversation is operational rather than
# a customer exchange. Patients get _REGISTER_FLOOR unchanged.
_REGISTER_FLOOR_STAFF = """\
Registro para staff (fijo — el tono del negocio no lo cambia):
- Trate al staff siempre de "usted", nunca de "tú" ni de "vos".
- Sin emojis.
- Sin diminutivos.
- Directo a la respuesta operativa — sin cortesías ni frases de relleno.
- Nunca se dirija al staff por su nombre de pila."""

_FORMAT_HINT_STAFF = _REGISTER_FLOOR_STAFF + _ITEM_FORMAT_RULES + """
- BREVE: máximo 4-5 líneas en total. Sin párrafos largos."""

_CATALOG_FORMAT_HINT_STAFF = _REGISTER_FLOOR_STAFF + _ITEM_FORMAT_RULES

def _build_results_note_rule(turnaround: str | None) -> str:
    """Applies whenever a price is quoted (RAG or catalog) — one line at the
    end of the reply naming the tenant's own results-turnaround time.
    Per-tenant column, not a hardcoded default: a tenant that never set one
    gets no invented turnaround, just an empty line here (see #46)."""
    if not turnaround:
        return ""
    return f'Si menciona algún precio, agregue al final una nota breve: "_Resultados: {turnaround}._"\n'

_RAG_SYSTEM = """\
Es un asistente de {expertise}. Su tono es {tone_description}.{specialization_block}{greeting_note}
Para precios y disponibilidad use ÚNICAMENTE el contexto proporcionado — NUNCA invente un producto
o precio que no esté ahí. Si arriba se le dio contexto de especialización, puede usar ese
conocimiento de dominio para interpretar jerga, sinónimos o abreviaturas del usuario, pero la
respuesta final SIEMPRE respeta el formato breve indicado más abajo, sin importar cuán detallado
sea ese conocimiento.

{match_instruction}

REGLAS (en orden de prioridad):
1. AMBIGÜEDAD: Si lo que pide el usuario puede referirse a varios ítems distintos, haga UNA sola pregunta breve y amable de aclaración. No asuma. EXCEPCIÓN: si el usuario ya incluyó en su mensaje el detalle que distingue entre los ítems similares (ej. pidió "con anexos" o mencionó explícitamente lo que un ítem incluye y otro no — "trompas y ovarios" especifica CON anexos), eso NO es ambigüedad — el usuario ya eligió, responda directo con ESE ítem, no pregunte de nuevo algo que ya contestó.
2. Si el contexto trae varios ítems cuyo nombre coincide con lo que pide el usuario, muéstrelos TODOS, sin filtrar por categoría o tipo.
3. {negative_confirmation_rule}
4. NO invente precios ni servicios.
{results_note_rule}{format_hint}
Contexto:
{context}
"""

_CATALOG_SYSTEM = """\
Es un asistente de {expertise}.{greeting_note}
Liste TODOS los ítems del catálogo a continuación, organizados por sección.
No omita ningún ítem. Use los nombres y precios exactos del catálogo.{contact_hint}
{results_note_rule}{format_hint}
Catálogo:
{context}
"""

_OFF_TOPIC_MSG = "Lo siento, no puedo ayudarle con eso. Soy un asistente especializado en {expertise}."

# Vertical-neutral fallback — no tenant's business facts belong here (see
# #46). Every real tenant is expected to set its own `greeting_message`
# column instead; this only covers a tenant that hasn't.
_GREETING_MSG = "Gracias por comunicarse con nosotros. ¿Cómo podemos ayudarle?"

_FALLBACK = "Lo siento, no pude procesar su consulta en este momento. Por favor intente de nuevo."

# Mixed message ("Hola, cuanto cuesta X") normally reaches "rag"/"catalog"
# (GREETING_PREFIX_RE is defined in retrieve.py -- triage.py uses the same
# signal to keep a mixed message OUT of "greeting"), answered normally but
# with no acknowledgement of the greeting at all. Deliberately does NOT pull
# in the full canned _GREETING_MSG/greeting_message here -- repeating the
# fixed address/phone/maps block on every priced question would read as MORE
# robotic, not less (see conversation). Instead the model opens with its own
# short natural acknowledgement.
_GREETING_ACK_NOTE = (
    "\nEl usuario abrió su mensaje con un saludo. Antes de responder, abra su respuesta con un "
    "saludo breve y natural (una frase corta, sin repetir información de contacto, horario ni "
    "ubicación) y luego responda la consulta."
)


# The match decision — exact vs. approximate — used to be computed here and
# then handed to the model as a per-chunk bracketed label ([COINCIDENCIA
# EXACTA] / [APROXIMACIÓN...]) plus several rules telling it not to
# recompute, question, or leak those labels. That's bookkeeping the code
# already did; the outcome now reaches the model as a single instruction
# instead (see ADR-007), and the labels never enter the context at all.
_MATCH_CONFIRMED_INSTRUCTION = (
    "El sistema ya verificó que el contexto contiene una coincidencia confiable para lo que pide "
    "el usuario: responda directo, con el precio y el nombre EXACTO del ítem tal como aparece en "
    "el contexto."
)

_MATCH_UNCONFIRMED_INSTRUCTION = (
    "El sistema determinó que el contexto NO contiene una coincidencia confiable para lo que pide "
    'el usuario, solo algo relacionado. Preséntelo de forma natural y pregunte: "¿Es lo que '
    'necesita?". NUNCA dé el precio como si fuera seguro hasta que el usuario confirme. Cuando '
    "confirme, dé el precio con el nombre EXACTO del ítem tal como aparece en el contexto — nunca "
    "lo renombre para que suene igual a lo que pidió el usuario — y mantenga una aclaración breve "
    "de que es lo más cercano disponible.\n"
    "EXCEPCIÓN — varios ítems (regla 2): si el contexto trae VARIOS ítems que coinciden por nombre, "
    'esto NO es una aproximación incierta de un solo ítem — no pregunte genérico "¿Es lo que '
    'necesita?"; si no es obvio cuál busca, pregunte si necesita todos o uno en particular. Si el '
    'usuario ya confirmó (p. ej. "sí") sin especificar cuál, NO repita la lista de precios que ya '
    'dio — confirme en una línea breve (p. ej. "Perfecto, esos son los tres.") sin re-listar, salvo '
    "que el usuario pida verla de nuevo."
)


# Negative-confirmation rule (#27) — shared lead, one place to fix its
# wording (see ADR-007's "expressed once" rationale). Only the tail differs:
# the patient version refers the requester to the tenant's contact; a staff
# member IS the business, so nothing points them at a contact line for their
# own workplace.
_NEGATIVE_CONFIRMATION_LEAD = (
    "CONFIRMACIÓN NEGATIVA: Si el usuario responde que la aproximación NO es lo que busca, o si "
    "definitivamente no hay nada relacionado, diga en una línea que no lo ofrecemos"
)

_NEGATIVE_CONFIRMATION_RULE = _NEGATIVE_CONFIRMATION_LEAD + " y eleve al contacto: {contact_hint}"

_NEGATIVE_CONFIRMATION_RULE_STAFF = _NEGATIVE_CONFIRMATION_LEAD + "."

# Found live: _MATCH_UNCONFIRMED_INSTRUCTION is written for a price/item
# approximation ("no dé el precio como seguro, pregunte si es lo que
# necesita") -- retrieve.py's location boost (see LOCATION_RE) guarantees
# the tenant's address/contact chunk reaches this prompt, but that chunk's
# own retrieval similarity is naturally low (it's scored against a generic
# probe, not the user's exact words), so _has_confirmed_match() often reads
# "unconfirmed" even when the address itself is exact, fixed, tenant-owned
# data with no approximation to hedge. Reproduced live: 11/12 regenerations
# against the same real context answered with the exact address, 1/12
# hedged into a vague restatement (once even switching to English) because
# the model followed the price-approximation framing literally. An address
# either IS or ISN'T in the context -- there's no "closest match" the way
# there is for "biopsia de pulmón" vs. a similarly-named item, so this
# exemption only fires for a location-intent message, never for an actual
# price/item lookup where the hedge is the correct behavior.
_LOCATION_CONFIDENCE_NOTE = (
    "\nExcepción — datos de contacto/ubicación: si el contexto incluye la dirección, teléfono o "
    "forma de contacto del negocio, indíquelos de forma directa y seguros — NO son una aproximación "
    "de un ítem, son datos fijos del negocio, así que la instrucción de arriba sobre no confirmar "
    "precios/preguntar \"¿Es lo que necesita?\" no aplica a ellos."
)


def _has_confirmed_match(chunks: list[dict]) -> bool:
    """The match decision, computed in code from the retrieval similarity —
    never handed to the model to recompute or narrate. Driven by the
    top-ranked chunk specifically (retrieve.py already reranks by relevance,
    so chunks[0] IS the primary match) rather than the max similarity across
    the whole list — a max would let one unrelated but numerically-similar
    later chunk falsely confirm a weak top match. True when no chunk carries
    a similarity score at all (e.g. chunks built from a raw catalog dump)."""
    for c in chunks:
        if "similarity" in c:
            return c["similarity"] >= settings.exact_match_threshold
    return True


async def _load_tenant(slug: str) -> dict:
    # Cross-reference note (found in /code-review): app/services/tenant_context.py's
    # get_tenant_specialization() runs its own independent SELECT for just
    # specialization_context (used by telegram.py/whatsapp.py before vision
    # calls). If this WHERE clause changes (e.g. an `active = true` filter),
    # check that function too — both read the same tenants row.
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            text(
                "SELECT expertise_area, tone_description, contact_url, specialization_context, "
                "greeting_message, results_turnaround FROM tenants WHERE slug = :s"
            ),
            {"s": slug},
        )).first()
    if not row:
        return {
            "expertise": "este negocio",
            "tone_description": DEFAULT_TONE_DESCRIPTION,
            "contact_hint": "",
            "specialization_context": "",
            "greeting_message": None,
            "results_turnaround": None,
        }
    expertise = row.expertise_area or "este negocio"
    contact_hint = (f"\nSi necesita más ayuda, contacte: {row.contact_url}" if row.contact_url else "")
    return {
        "expertise": expertise,
        "greeting_message": row.greeting_message,
        "tone_description": row.tone_description or DEFAULT_TONE_DESCRIPTION,
        "contact_hint": contact_hint,
        "specialization_context": row.specialization_context or "",
        "results_turnaround": row.results_turnaround,
    }


def _build_specialization_block(specialization: str) -> str:
    return f"\nContexto de especialización:\n{specialization}\n" if specialization else ""


async def generate(state: AgentState, runtime: Runtime | None = None) -> dict:
    chunks = list(state.get("retrieved_chunks") or [])
    decision = state.get("triage_decision", "rag")
    is_catalog = decision == "catalog"
    is_staff = bool(state.get("is_staff"))
    tenant_ctx = await _load_tenant(state["tenant_id"])
    if is_staff:
        # A staff member IS the business — no referral line pointing them at
        # their own workplace's contact (#27).
        tenant_ctx["contact_hint"] = ""

    logger.info("generate_called decision=%s chunks=%d", decision, len(chunks))

    if decision == "off_topic":
        content = _OFF_TOPIC_MSG.format(**tenant_ctx)
        msg = AIMessage(content=content)
        # Explicit False, not omitted -- an unrelated message closes out any
        # approximation offered last turn (see AgentState.awaiting_confirmation).
        return {"answer": content, "messages": [msg], "awaiting_confirmation": False}

    if decision == "greeting":
        # Static reply — a bare greeting has no question to answer, so this
        # skips the LLM call entirely (not just the retrieve/rerank stage the
        # router already skips for this decision). Cuts a ~40s round trip
        # (retrieve+rerank+triage+generate, all sequential) down to just the
        # triage call needed to classify the message in the first place.
        content = tenant_ctx.get("greeting_message") or _GREETING_MSG
        msg = AIMessage(content=content)
        return {"answer": content, "messages": [msg], "awaiting_confirmation": False}

    if decision == "canned":
        # Verbatim tenant-authored reply, matched in triage.py before any
        # model call (#50) -- same zero-LLM short-circuit shape as the
        # greeting branch above.
        content = state.get("canned_answer", "")
        msg = AIMessage(content=content)
        return {"answer": content, "messages": [msg], "awaiting_confirmation": False}

    if is_catalog and not chunks:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                text(
                    "SELECT content FROM document_chunks "
                    "WHERE namespace = :ns AND embedding IS NOT NULL "
                    "ORDER BY id LIMIT 100"
                ),
                {"ns": state["tenant_id"]},
            )
            chunks = [{"content": r.content} for r in result.fetchall()]
            chunks = cap_chunks_to_tokens(chunks, settings.retrieval_max_tokens)

    confirmed = _has_confirmed_match(chunks)

    # Two escalation signals, neither the model's opinion (see ADR-009).
    # Catalog and staff never escalate on either signal — a catalog dump may
    # carry no similarity at all, and a staff member IS the business, nobody
    # to hand them to. Computed fresh from this turn's own chunks/message,
    # never stored in state, so a prior turn's escalation can't leak into
    # this one (the *rejection* signal below does read one bit of state, but
    # only the guard that expires after exactly one turn).
    escalate = False
    if not is_catalog and not is_staff:
        # Signal 1: nothing in the retrieved pool is close to what was
        # asked. Reads the MAXIMUM similarity across the pool — the
        # opposite of _has_confirmed_match's chunks[0], deliberately: that
        # guards against a false confirmation, this guards against a false
        # escalation, and chunk order is fused/reranked so chunks[0] is
        # frequently not the highest-similarity chunk. An empty pool never
        # escalates either: it means the tenant was never indexed, an
        # operational fault, not a conversation.
        similarities = [c["similarity"] for c in chunks if "similarity" in c]
        if similarities and max(similarities) < settings.handoff_threshold:
            escalate = True
            logger.info(
                "generate_escalating reason=similarity tenant=%s max_similarity=%.3f threshold=%.3f",
                state["tenant_id"], max(similarities), settings.handoff_threshold,
            )
        elif not confirmed and is_bare_rejection(last_human_text(state) or "") and state.get("awaiting_confirmation"):
            # Signal 2: the user rejected the approximation the bot offered
            # LAST turn -- only when that turn actually offered one, or
            # every "no" would escalate, including one answering a reply
            # that was confidently correct. Retrieval stays anchored to the
            # original question on a bare rejection (see retrieve.py), so
            # this fires alongside a still-unconfirmed match, never instead
            # of the similarity signal above.
            escalate = True
            logger.info("generate_escalating reason=rejection tenant=%s", state["tenant_id"])
        if escalate:
            # Pointing at a contact URL while queueing a person for the
            # same question is two answers to one question (#27's staff
            # variant already suppresses this for the same reason).
            tenant_ctx["contact_hint"] = ""

    context = "Sin contexto disponible." if not chunks else "\n\n---\n\n".join(c["content"] for c in chunks)
    greeting_note = (
        _GREETING_ACK_NOTE
        if not is_staff and GREETING_PREFIX_RE.match(last_human_text(state) or "")
        else ""
    )
    template = _CATALOG_SYSTEM if is_catalog else _RAG_SYSTEM
    if is_catalog:
        hint_template = _CATALOG_FORMAT_HINT_STAFF if is_staff else _CATALOG_FORMAT_HINT
    else:
        hint_template = _FORMAT_HINT_STAFF if is_staff else _FORMAT_HINT
    format_hint = hint_template.format(tone_description=tenant_ctx["tone_description"])
    match_instruction = _MATCH_CONFIRMED_INSTRUCTION if confirmed else _MATCH_UNCONFIRMED_INSTRUCTION
    if not confirmed and LOCATION_RE.search(last_human_text(state) or ""):
        match_instruction += _LOCATION_CONFIDENCE_NOTE
    negative_confirmation_rule = (
        _NEGATIVE_CONFIRMATION_RULE_STAFF if is_staff
        else _NEGATIVE_CONFIRMATION_RULE.format(contact_hint=tenant_ctx["contact_hint"])
    )
    # Defensive .get(), not a bare {specialization_context} placeholder relying on
    # **tenant_ctx always containing the key — existing tests mock _load_tenant()'s
    # return dict directly and don't include this key; a missing key would KeyError.
    specialization = tenant_ctx.get("specialization_context", "") or ""
    specialization_block = _build_specialization_block(specialization) if not is_catalog else ""
    if specialization:
        logger.debug("generate_specialization_applied tenant=%s len=%d", state["tenant_id"], len(specialization))
    results_note_rule = _build_results_note_rule(tenant_ctx.get("results_turnaround"))
    system = template.format(
        context=context, format_hint=format_hint, match_instruction=match_instruction,
        negative_confirmation_rule=negative_confirmation_rule, results_note_rule=results_note_rule,
        specialization_block=specialization_block, greeting_note=greeting_note, **tenant_ctx,
    )

    trimmed = trim_messages(
        state["messages"],
        max_tokens=settings.history_max_tokens,
        strategy="last",
        token_counter=token_counter,
        allow_partial=False,
        include_system=True,
    )

    llm = get_chat_llm()
    logger.info("generate_llm_model=%s", llm.model_name)
    try:
        response = await llm.ainvoke([SystemMessage(content=system)] + trimmed)
        logger.info("generate_response=%s", response.content[:80])
    except Exception as exc:
        if not settings.openai_fallback_model:
            raise
        logger.warning("generate_primary_failed=%s retrying with fallback", exc)
        response = await get_chat_llm(fallback=True).ainvoke(
            [SystemMessage(content=system)] + trimmed
        )

    content = response.content
    if escalate:
        # The bot answers, then escalates — closes with the same handover
        # line the reactive suspend path sends, never the model's own words:
        # this is a decision computed in code, not something it narrates.
        # model_copy (not a fresh AIMessage) keeps response_metadata/
        # usage_metadata/id intact — this is the message a token-accounting
        # or tracing consumer reads for the turn that just escalated.
        content = f"{content.rstrip()}\n\n{HUMAN_HANDOFF}"
        response = response.model_copy(update={"content": content})
        await human_control.start(state["tenant_id"], state["thread_id"], state.get("chat_id", ""))

    # True only for a genuine unconfirmed-RAG offer that didn't itself
    # escalate -- catalog/staff turns and confirmed matches all close the
    # question out, same as escalating does.
    awaiting_confirmation = not is_catalog and not is_staff and not escalate and not confirmed
    return {"answer": content, "messages": [response], "awaiting_confirmation": awaiting_confirmation}
