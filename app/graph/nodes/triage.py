import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages

from app.config import settings
from app.schemas.triage import TriageDecision
from app.services.llm import get_triage_llm
from app.services.rag import token_counter
from app.state import AgentState

logger = logging.getLogger(__name__)

_TRIAGE_PROMPT = """\
Classify the user's latest message into ONE category:
- "greeting": ONLY a greeting, thanks, farewell, or social pleasantry with NO question about products/services/prices at all
- "rag": ANY question about a product, service, price, exam, procedure, study, biopsy, analysis, cost, or anything the business might offer — even if vague
- "catalog": explicitly wants a FULL list/catalog/ALL products or services
- "human": explicitly asks to speak with a human, operator, or agent
- "off_topic": ONLY if completely unrelated (politics, weather, sports, jokes, coding questions)

IMPORTANT: Medical terms, body parts, lab tests, procedures, and prices are ALWAYS "rag" —
never "greeting", even if the message opens with "hola" first.
Examples of "greeting": "hola", "buenas", "gracias", "buen día", "hasta luego"
Examples of "rag": "biopsia de pulmon", "cuanto cuesta", "riñon", "análisis de sangre", "histología"
Examples of "off_topic": "quien ganó el partido", "como programo en python", "chiste"

When in doubt between rag/off_topic → "rag". Default is "rag".
Reply ONLY with JSON: {"decision": "<category>"}
"""

# Whole-message-only match — a pure greeting/farewell/thanks with nothing
# else. Mirrors the prompt's own "IMPORTANT" rule: any extra content (a
# question, a medical term) must still reach the LLM as "rag", so this never
# matches a partial prefix like "hola, cuanto cuesta" — the trailing
# `[.!¡?¿,]*\s*$` anchor requires nothing follows the greeting phrase itself.
# Short-circuits the common high-frequency, near-zero-ambiguity case (bare
# "hola"/"gracias"/"chao") without paying for an LLM call on every message.
_GREETING_RE = re.compile(
    r"^\s*"
    r"(?:hola+|holis|buen[oa]s?(?:\s+(?:d[ií]as?|tardes|noches))?|"
    r"gracias|muchas\s+gracias|mil\s+gracias|de\s+nada|"
    r"hasta\s+luego|nos\s+vemos|chao|adi[oó]s|"
    r"mucho\s+gusto|un\s+gusto)"
    r"\s*[.!¡?¿,]*\s*$",
    re.IGNORECASE,
)


def _last_human_text(state: AgentState) -> str | None:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else None
    return None


async def triage(state: AgentState) -> dict:
    last_human = _last_human_text(state)
    if last_human is None:
        return {"triage_decision": "rag"}

    if _GREETING_RE.match(last_human):
        logger.info("triage_regex_greeting_shortcut")
        return {"triage_decision": "greeting"}

    trimmed = trim_messages(
        state["messages"],
        max_tokens=settings.history_max_tokens,
        strategy="last",
        token_counter=token_counter,
        allow_partial=False,
        include_system=True,
    )

    llm = get_triage_llm()
    payload = [SystemMessage(content=_TRIAGE_PROMPT)] + trimmed

    # Primary: structured output (function calling)
    try:
        result: TriageDecision = await llm.with_structured_output(TriageDecision).ainvoke(payload)
        return {"triage_decision": result.decision}
    except Exception as exc:
        logger.warning("triage_structured_failed=%s falling back to json parse", exc)

    # Fallback: raw LLM + JSON parse (strip markdown fences if present)
    try:
        resp = await llm.ainvoke(payload)
        content = resp.content.strip()
        content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
        content = re.sub(r"\s*```$", "", content).strip()
        decision = json.loads(content)["decision"]
        td = TriageDecision(decision=decision)  # validate enum
        return {"triage_decision": td.decision}
    except Exception:
        logger.warning("triage_json_fallback_failed defaulting to rag")
        return {"triage_decision": "rag"}
