import logging

from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langgraph.runtime import Runtime
from sqlalchemy import text

from app.config import settings
from app.db import AsyncSessionLocal
from app.graph.thread import profile_namespace
from app.models.tenant import DEFAULT_TONE_DESCRIPTION
from app.services.llm import get_chat_llm
from app.services.rag import cap_chunks_to_tokens, token_counter
from app.state import AgentState

logger = logging.getLogger(__name__)

_FORMAT_HINT = """
Formato (OBLIGATORIO — compatible WhatsApp/Telegram):
- Tono: {tone_description}.
- Abre con una frase breve y natural acorde a ese tono si corresponde (confirmación directa, sin relleno).
- BREVE: máximo 4-5 líneas en total. Sin párrafos largos.
- *negrita* con asteriscos simples para códigos y nombres de ítems.
- _cursiva_ con guiones bajos para notas o aclaraciones breves.
- Listas con guión (- item). Sin tablas, sin encabezados Markdown (##).
- Por ítem: - *CÓDIGO* Nombre: $precio"""

_RAG_SYSTEM = """\
Eres un asistente de {expertise}. Eres {tone_description}.{name_hint}{specialization_hint}
Usa ÚNICAMENTE el contexto proporcionado. NO uses conocimiento propio fuera de ese contexto.

Cada ítem del contexto ya viene etiquetado por el sistema de búsqueda como [COINCIDENCIA EXACTA] o
[APROXIMACIÓN (confianza baja)] — es una clasificación ya calculada, NO la recalcules ni la
cuestiones aunque el nombre te "suene" parecido:
- [COINCIDENCIA EXACTA]: trátalo como tal si además el nombre corresponde a lo que pide el usuario.
- [APROXIMACIÓN (confianza baja)]: SIEMPRE es aproximación, sin excepción. Nunca afirmes un precio
  directo en este caso — primero confirma con el usuario.
- Estas etiquetas son solo para tu clasificación interna — NUNCA las escribas literalmente en tu
  respuesta al usuario (nada de "[COINCIDENCIA EXACTA]" ni "[APROXIMACIÓN...]" en el texto final).

REGLAS (en orden de prioridad):
1. AMBIGÜEDAD: Si lo que pide el usuario puede referirse a varios ítems distintos, haz UNA sola pregunta breve y amable de aclaración. No asumas.
2. COINCIDENCIA EXACTA (etiquetado [COINCIDENCIA EXACTA] Y el nombre corresponde): Muestra TODOS los ítems del contexto cuyo nombre coincida con lo que el usuario menciona, sin filtrar por categoría o tipo.
3. APROXIMACIÓN — primera vez (etiquetado [APROXIMACIÓN] O el nombre no corresponde exactamente): Si el ítem exacto no está en el contexto pero hay algo relacionado, preséntalo de forma natural y pregunta: "¿Eso es lo que necesitas?" NO eleves al contacto todavía — espera la confirmación del usuario. NUNCA dés el precio como si fuera seguro.
4. APROXIMACIÓN — el usuario CONFIRMA que sí: Da el precio, pero SIEMPRE con el nombre EXACTO del ítem tal como aparece en el contexto — nunca lo renombres para que suene igual a lo que pidió el usuario. Mantén una aclaración breve de que es lo más cercano disponible (ej. "el precio de *Citología de otros sitios*, que es lo más cercano que tenemos, es..."). La confirmación del usuario valida que quiere ESE ítem, no que el ítem sea una coincidencia exacta.
5. CONFIRMACIÓN NEGATIVA: Si el usuario responde que la aproximación NO es lo que busca, o si definitivamente no hay nada relacionado, di en una línea que no lo ofrecemos y eleva al contacto: {contact_hint}
- NO inventes precios ni servicios.
{format_hint}
Contexto:
{context}
"""

_CATALOG_SYSTEM = """\
Eres un asistente de {expertise}.{name_hint}
Lista TODOS los ítems del catálogo a continuación, organizados por sección.
No omitas ningún ítem. Usa los nombres y precios exactos del catálogo.{contact_hint}
{format_hint}
Catálogo:
{context}
"""

_OFF_TOPIC_MSG = "Lo siento, no puedo ayudarte con eso. Soy un asistente especializado en {expertise}."

_GREETING_MSG = "¡Hola! 👋 Gracias por escribirnos. Somos especialistas en {expertise}. ¿En qué podemos ayudarte hoy?"

_FALLBACK = "Lo siento, no pude procesar tu consulta en este momento. Por favor intenta de nuevo."


def _match_tag(similarity: float) -> str:
    is_exact = similarity >= settings.exact_match_threshold
    return "COINCIDENCIA EXACTA" if is_exact else "APROXIMACIÓN (confianza baja)"


async def _load_tenant(slug: str) -> dict:
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
    contact_hint = (f"\nSi necesitas más ayuda, contacta: {row.contact_url}" if row.contact_url else "")
    return {
        "expertise": expertise,
        "tone_description": row.tone_description or DEFAULT_TONE_DESCRIPTION,
        "contact_hint": contact_hint,
        "specialization_context": row.specialization_context or "",
    }


def _build_specialization_hint(specialization: str) -> str:
    return f"\nContexto de especialización: {specialization}." if specialization else ""


async def _load_name_hint(state: AgentState, runtime: Runtime | None) -> str:
    if runtime is None or runtime.store is None:
        return ""
    try:
        item = await runtime.store.aget(profile_namespace(state), "profile")
    except Exception:
        return ""
    name = (item.value.get("display_name") if item else None) or ""
    return f" El usuario se llama {name}, salúdalo por su nombre si es natural." if name else ""


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

    if not chunks:
        context = "Sin contexto disponible."
    elif is_catalog:
        context = "\n\n---\n\n".join(c["content"] for c in chunks)
    else:
        context = "\n\n---\n\n".join(
            f"{c['content']} [{_match_tag(c['similarity'])}]"
            if "similarity" in c else c["content"]
            for c in chunks
        )
    template = _CATALOG_SYSTEM if is_catalog else _RAG_SYSTEM
    format_hint = _FORMAT_HINT.format(tone_description=tenant_ctx["tone_description"])
    name_hint = await _load_name_hint(state, runtime)
    # Defensive .get(), not a bare {specialization_context} placeholder relying on
    # **tenant_ctx always containing the key — existing tests mock _load_tenant()'s
    # return dict directly and don't include this key; a missing key would KeyError.
    specialization = tenant_ctx.get("specialization_context", "") or ""
    specialization_hint = _build_specialization_hint(specialization) if not is_catalog else ""
    if specialization:
        logger.debug("generate_specialization_applied tenant=%s len=%d", state["tenant_id"], len(specialization))
    system = template.format(
        context=context, format_hint=format_hint, name_hint=name_hint,
        specialization_hint=specialization_hint, **tenant_ctx,
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
