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
- "greeting": ONLY a greeting, thanks, farewell, or social pleasantry with NO question at all
- "rag": the default for ANYTHING a real customer of THIS business might plausibly ask — not
  just prices/exams/procedures, but ALSO practical/logistical questions about the business
  itself: location, address, how to get there, hours, phone/contact, payment methods,
  insurance/convenios, delivery of results, appointments, parking, or anything else about how
  to interact with this business. If the message is about the business in ANY way and isn't
  clearly one of the other categories below, it is "rag" — the examples below are
  illustrations, not an exhaustive list of what counts.
- "catalog": explicitly asks to see the FULL list/catalog/menu of ALL products or services —
  not a specific question about one aspect of the business (that is "rag", even if that aspect
  isn't explicitly listed as an example anywhere in this prompt).
- "human": explicitly asks to speak with a human, operator, or agent
- "off_topic": the business genuinely has nothing to do with this message — general knowledge,
  entertainment, other businesses, or topics with no plausible connection to this business at
  all (politics, weather, sports, jokes, coding questions). This is the rare exception, not a
  fallback for "I'm not sure which specific category this practical question fits."

IMPORTANT: Medical terms, body parts, lab tests, procedures, and prices are ALWAYS "rag" —
never "greeting", even if the message opens with "hola" first. A question about the business's
location, hours, contact info, payment methods, or any other practical/logistical topic is
ALSO "rag" — these are real questions about the business, never "off_topic", and asking about
ONE of them is not the same as asking for the FULL catalog.
Examples of "greeting": "hola", "buenas", "gracias", "buen día", "hasta luego"
Examples of "rag": "biopsia de pulmon", "cuanto cuesta", "riñon", "análisis de sangre", "histología",
"donde estan ubicados", "cual es la direccion", "que horario tienen", "como los contacto",
"que metodos de pago aceptan", "hacen envio de resultados", "tienen convenio con seguros"
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

# Found live (#42): once the LLM misclassifies a location question as
# off_topic ONE time, that (wrong) exchange sits in the conversation history
# forever, and the model then pattern-matches its own prior refusal on every
# later location question in the same thread -- reproduced 8/8 off_topic
# with the real poisoned history, vs 8/8 rag with a fresh/short one. No
# amount of prompt tuning fixes a classifier conditioning on its own past
# mistakes, so this -- like _GREETING_RE above -- routes a known-safe
# category deterministically instead of asking the LLM at all. Substring
# match (not whole-message anchored like greeting): "hola, donde quedan" or
# "y donde se encuentran ubicados" must still hit this before either the
# greeting regex or the LLM sees them.
_LOCATION_RE = re.compile(
    r"d[oó]nde\s+(es|qued[ao]n?|est[aá]n?(?:\s+ubicados?)?|se\s+encuentran|"
    r"puedo\s+encontrarlos?)|"
    r"direcci[oó]n|"
    r"c[oó]mo\s+(llego|llegar|puedo\s+llegar)|"
    r"ubicaci[oó]n|"
    r"en\s+qu[eé]\s+(parte|zona|sector|lugar)",
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

    if _LOCATION_RE.search(last_human):
        logger.info("triage_regex_location_shortcut")
        return {"triage_decision": "rag"}

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
