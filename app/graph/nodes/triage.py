import asyncio
import json
import logging
import re

from langchain_core.messages import SystemMessage, trim_messages

from app.config import settings
from app.graph.nodes.retrieve import is_bare_rejection, last_human_text
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
    r"mucho\s+gusto|un\s+gusto|"
    r"hi+|hey+|hello+|thanks|thank\s+you|bye)"
    r"\s*[.!¡?¿,]*\s*$",
    re.IGNORECASE,
)


async def triage(state: AgentState) -> dict:
    last_human = last_human_text(state)
    if last_human is None:
        return {"triage_decision": "rag"}

    # A bare rejection answering last turn's approximation offer must reach
    # generate()'s signal-2 escalation check, which needs retrieve() to have
    # run first -- off_topic/greeting skip retrieve() entirely (see
    # builder.py's _route_triage), so classifying it as either would silently
    # defeat the whole rejection-escalation feature (see ADR-009 / #38).
    # Checked ahead of both the regex shortcut and the LLM call: a bare "no"
    # would otherwise never match the greeting regex, but the LLM could still
    # call it off_topic.
    if is_bare_rejection(last_human) and state.get("awaiting_confirmation"):
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

    # Structured (function-calling) and raw+JSON-parse both fired at once,
    # not sequentially -- triage_model (amazon/nova-micro-v1) runs 7-10s per
    # call (live-measured, see docs/model-upgrade-baseline.md), so a
    # sequential retry doubled worst-case latency to ~20s. Running both
    # concurrently caps it at ~10s (the slower of the two) with the same
    # two-tier accuracy this already had -- structured preferred, JSON parse
    # as fallback, "rag" as last resort. Costs one extra OpenRouter call on
    # every turn instead of only on structured failures; nova-micro is cheap
    # enough ($0.035/$0.14 per 1M tokens) that this isn't a real cost trade.
    structured_result, raw_response = await asyncio.gather(
        llm.with_structured_output(TriageDecision).ainvoke(payload),
        llm.ainvoke(payload),
        return_exceptions=True,
    )

    # isinstance(TriageDecision), not "not an exception" -- with_structured_
    # output() returns None (no exception) when the model skips the tool
    # call entirely, a documented langchain_core behavior distinct from a
    # parse/validation failure (see langchain-ai/langchain#36349).
    if isinstance(structured_result, TriageDecision):
        return {"triage_decision": structured_result.decision}
    logger.warning("triage_structured_failed=%s falling back to json parse", structured_result)

    if not isinstance(raw_response, BaseException):
        try:
            content = raw_response.content.strip()
            content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
            content = re.sub(r"\s*```$", "", content).strip()
            decision = json.loads(content)["decision"]
            td = TriageDecision(decision=decision)  # validate enum
            return {"triage_decision": td.decision}
        except Exception:
            pass

    logger.warning("triage_json_fallback_failed defaulting to rag")
    return {"triage_decision": "rag"}
