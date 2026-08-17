import logging

from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langgraph.runtime import Runtime
from sqlalchemy import text

from app.config import settings
from app.db import AsyncSessionLocal
from app.models.tenant import DEFAULT_TONE_DESCRIPTION
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

_RAG_SYSTEM = """\
Es un asistente de {expertise}. Su tono es {tone_description}.{specialization_block}
Para precios y disponibilidad use ÚNICAMENTE el contexto proporcionado — NUNCA invente un producto
o precio que no esté ahí. Si arriba se le dio contexto de especialización, puede usar ese
conocimiento de dominio para interpretar jerga, sinónimos o abreviaturas del usuario, pero la
respuesta final SIEMPRE respeta el formato breve indicado más abajo, sin importar cuán detallado
sea ese conocimiento.

{match_instruction}

REGLAS (en orden de prioridad):
1. AMBIGÜEDAD: Si lo que pide el usuario puede referirse a varios ítems distintos, haga UNA sola pregunta breve y amable de aclaración. No asuma. EXCEPCIÓN: si el usuario ya incluyó en su mensaje el detalle que distingue entre los ítems similares (ej. pidió "con anexos" o mencionó explícitamente lo que un ítem incluye y otro no — "trompas y ovarios" especifica CON anexos), eso NO es ambigüedad — el usuario ya eligió, responda directo con ESE ítem, no pregunte de nuevo algo que ya contestó.
2. Si el contexto trae varios ítems cuyo nombre coincide con lo que pide el usuario, muéstrelos TODOS, sin filtrar por categoría o tipo.
3. CONFIRMACIÓN NEGATIVA: Si el usuario responde que la aproximación NO es lo que busca, o si definitivamente no hay nada relacionado, diga en una línea que no lo ofrecemos y eleve al contacto: {contact_hint}
- NO invente precios ni servicios.
{format_hint}
Contexto:
{context}
"""

_CATALOG_SYSTEM = """\
Es un asistente de {expertise}.
Liste TODOS los ítems del catálogo a continuación, organizados por sección.
No omita ningún ítem. Use los nombres y precios exactos del catálogo.{contact_hint}
{format_hint}
Catálogo:
{context}
"""

_OFF_TOPIC_MSG = "Lo siento, no puedo ayudarle con eso. Soy un asistente especializado en {expertise}."

_GREETING_MSG = "Hola, gracias por escribirnos. Somos especialistas en {expertise}. ¿En qué podemos ayudarle?"

_FALLBACK = "Lo siento, no pude procesar su consulta en este momento. Por favor intente de nuevo."


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
    "de que es lo más cercano disponible."
)


def _has_confirmed_match(chunks: list[dict]) -> bool:
    """The match decision, computed in code from the retrieval similarity —
    never handed to the model to recompute or narrate. True when the best
    scored chunk clears the exact-match threshold, or when no chunk carries
    a similarity score at all (e.g. chunks built from a raw catalog dump)."""
    scored = [c["similarity"] for c in chunks if "similarity" in c]
    if not scored:
        return True
    return max(scored) >= settings.exact_match_threshold


async def _load_tenant(slug: str) -> dict:
    # Cross-reference note (found in /code-review): app/services/tenant_context.py's
    # get_tenant_specialization() runs its own independent SELECT for just
    # specialization_context (used by telegram.py/whatsapp.py before vision
    # calls). If this WHERE clause changes (e.g. an `active = true` filter),
    # check that function too — both read the same tenants row.
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            text(
                "SELECT expertise_area, tone_description, contact_url, specialization_context "
                "FROM tenants WHERE slug = :s"
            ),
            {"s": slug},
        )).first()
    if not row:
        return {
            "expertise": "este negocio",
            "tone_description": DEFAULT_TONE_DESCRIPTION,
            "contact_hint": "",
            "specialization_context": "",
        }
    expertise = row.expertise_area or "este negocio"
    contact_hint = (f"\nSi necesita más ayuda, contacte: {row.contact_url}" if row.contact_url else "")
    return {
        "expertise": expertise,
        "tone_description": row.tone_description or DEFAULT_TONE_DESCRIPTION,
        "contact_hint": contact_hint,
        "specialization_context": row.specialization_context or "",
    }


def _build_specialization_block(specialization: str) -> str:
    return f"\nContexto de especialización:\n{specialization}\n" if specialization else ""


async def generate(state: AgentState, runtime: Runtime | None = None) -> dict:
    chunks = list(state.get("retrieved_chunks") or [])
    decision = state.get("triage_decision", "rag")
    is_catalog = decision == "catalog"
    tenant_ctx = await _load_tenant(state["tenant_id"])

    logger.info("generate_called decision=%s chunks=%d", decision, len(chunks))

    if decision == "off_topic":
        content = _OFF_TOPIC_MSG.format(**tenant_ctx)
        msg = AIMessage(content=content)
        return {"answer": content, "messages": [msg]}

    if decision == "greeting":
        # Static reply — a bare greeting has no question to answer, so this
        # skips the LLM call entirely (not just the retrieve/rerank stage the
        # router already skips for this decision). Cuts a ~40s round trip
        # (retrieve+rerank+triage+generate, all sequential) down to just the
        # triage call needed to classify the message in the first place.
        content = _GREETING_MSG.format(**tenant_ctx)
        msg = AIMessage(content=content)
        return {"answer": content, "messages": [msg]}

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

    context = "Sin contexto disponible." if not chunks else "\n\n---\n\n".join(c["content"] for c in chunks)
    template = _CATALOG_SYSTEM if is_catalog else _RAG_SYSTEM
    hint_template = _CATALOG_FORMAT_HINT if is_catalog else _FORMAT_HINT
    format_hint = hint_template.format(tone_description=tenant_ctx["tone_description"])
    match_instruction = (
        _MATCH_CONFIRMED_INSTRUCTION if _has_confirmed_match(chunks) else _MATCH_UNCONFIRMED_INSTRUCTION
    )
    # Defensive .get(), not a bare {specialization_context} placeholder relying on
    # **tenant_ctx always containing the key — existing tests mock _load_tenant()'s
    # return dict directly and don't include this key; a missing key would KeyError.
    specialization = tenant_ctx.get("specialization_context", "") or ""
    specialization_block = _build_specialization_block(specialization) if not is_catalog else ""
    if specialization:
        logger.debug("generate_specialization_applied tenant=%s len=%d", state["tenant_id"], len(specialization))
    system = template.format(
        context=context, format_hint=format_hint, match_instruction=match_instruction,
        specialization_block=specialization_block, **tenant_ctx,
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

    return {"answer": response.content, "messages": [response]}
