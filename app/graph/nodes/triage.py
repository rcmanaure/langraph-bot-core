import json
import logging
import re

from langchain_core.messages import SystemMessage, trim_messages

from app.config import settings
from app.graph.nodes.retrieve import LOCATION_RE, is_bare_rejection, last_human_text
from app.schemas.triage import TriageDecision
from app.services.llm import get_triage_llm
from app.services.prompt_pack import get_rag_examples
from app.services.rag import token_counter
from app.services.tenant_context import get_tenant_vertical
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
Examples of "greeting": "hola", "buenas", "gracias", "buen día", "hasta luego", "qué tal",
"quihubo", "épale", "aló", "qué más" (LATAM informal openers, not just "hola")
Examples of "rag": "biopsia de pulmon", "cuanto cuesta", "riñon", "análisis de sangre", "histología",
"donde estan ubicados", "cual es la direccion", "que horario tienen", "como los contacto",
"que metodos de pago aceptan", "hacen envio de resultados", "tienen convenio con seguros"
Examples of "off_topic": "quien ganó el partido", "como programo en python", "chiste"

When in doubt between rag/off_topic → "rag". Default is "rag".
Reply ONLY with JSON: {"decision": "<category>"}
"""


def _build_triage_prompt(extra_rag_examples: list[str]) -> str:
    """Appends pack-supplied vocabulary (app/services/prompt_pack.py) after
    the fixed classification prompt — vocabulary only, never a new rule or
    instruction (see ADR-007/ADR-011): the appended block can only ever be a
    comma-separated example list, so a pack row can't smuggle in an
    instruction that changes how triage classifies or how generate.py's
    register floor behaves. No-op (returns _TRIAGE_PROMPT unchanged) for a
    tenant with no vertical/pack, so this is byte-identical to before the
    pack mechanism existed — see tests/test_evals.py, which imports
    _TRIAGE_PROMPT directly and must keep working unmodified."""
    if not extra_rag_examples:
        return _TRIAGE_PROMPT
    extra = ", ".join(extra_rag_examples)
    return _TRIAGE_PROMPT + f'\nVocabulario adicional de "rag" para este rubro: {extra}\n'

# Shared by both regexes below. Widened 2026-08-19 to cover common LATAM
# openers beyond hola/buenas (qué tal, quihubo/quiubo, épale, aló, qué
# onda, qué más -- México/Colombia/Venezuela/general usage; see
# GREETING_PREFIX_RE in retrieve.py for the same widening and rationale).
# buen[oa]s? -> buen(?:[oa]s?)? so a bare "buen día" (no trailing s) matches.
_GREETING_PHRASE = (
    r"(?:hola+|holis|saludos?|qu[eé]\s+tal|qu[eé]\s+h[uú]bo|quih[uú]bo|quiub[oa]?|"
    r"qu[eé]\s+m[aá]s|qu[eé]\s+onda|[eé]pale|epa|al[oó]|"
    r"buen(?:[oa]s?)?(?:\s+(?:d[ií]as?|tardes|noches))?|"
    r"gracias|muchas\s+gracias|mil\s+gracias|de\s+nada|"
    r"hasta\s+luego|nos\s+vemos|chao|adi[oó]s|"
    r"mucho\s+gusto|un\s+gusto|"
    r"hi+|hey+|hello+|thanks|thank\s+you|bye)"
)

# Whole-message-only match — a pure greeting/farewell/thanks with nothing
# else. Mirrors the prompt's own "IMPORTANT" rule: any extra content (a
# question, a medical term) must still reach the LLM as "rag", so this never
# matches a partial prefix like "hola, cuanto cuesta" — the trailing
# `\s*[.!¡?¿,]*\s*$` anchor requires nothing follows the greeting phrase itself.
# Short-circuits the common high-frequency, near-zero-ambiguity case (bare
# "hola"/"gracias"/"chao") without paying for an LLM call on every message.
_GREETING_RE = re.compile(rf"^\s*{_GREETING_PHRASE}\s*[.!¡?¿,]*\s*$", re.IGNORECASE)

# Chained variant of the same phrase set ("hola buenas", "hola, gracias") —
# one or more greeting phrases back to back, still nothing else. Used below
# to tell an honestly-pure multi-word greeting apart from a real mixed
# message ("hola cuanto cuesta X") when validating the LLM's own "greeting"
# verdict, since _GREETING_RE's single-phrase anchor doesn't cover it.
_GREETING_CHAIN_RE = re.compile(rf"^\s*(?:{_GREETING_PHRASE}\s*[.!¡?¿,]*\s*)+$", re.IGNORECASE)

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
#
# Defined in retrieve.py, not here -- retrieve() reuses the same signal to
# guarantee the tenant's address chunk survives retrieval for a location
# question (see LOCATION_RE's docstring there), and retrieve.py is already
# the module triage.py imports from (is_bare_rejection/last_human_text
# below), so defining it there instead of here avoids a circular import.


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

    if LOCATION_RE.search(last_human):
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

    vertical = await get_tenant_vertical(state["tenant_id"])
    extra_rag_examples = await get_rag_examples(vertical)

    llm = get_triage_llm()
    payload = [SystemMessage(content=_build_triage_prompt(extra_rag_examples))] + trimmed

    # ONE model call, both tiers. This used to fire the structured
    # (function-calling) call and a plain raw call concurrently via
    # asyncio.gather, purely so the JSON-parse fallback would have the
    # model's own text in hand without paying a second round trip
    # sequentially -- but that doubled the call count and the prompt tokens
    # on every non-shortcut turn. include_raw=True returns
    # {"raw": AIMessage, "parsed": ..., "parsing_error": ...} from a single
    # call, which is exactly the two-tier input the code below already
    # wanted: structured preferred, JSON parse of the raw text as fallback,
    # "rag" as last resort.
    try:
        result = await llm.with_structured_output(TriageDecision, include_raw=True).ainvoke(payload)
    except Exception as exc:
        logger.warning("triage_call_failed=%s defaulting to rag", exc)
        return {"triage_decision": "rag"}

    structured_result = result.get("parsed")
    raw_response = result.get("raw")

    # isinstance(TriageDecision), not "is not None" -- with_structured_
    # output() yields parsed=None (no exception) when the model skips the
    # tool call entirely, a documented langchain_core behavior distinct from
    # a parse/validation failure (see langchain-ai/langchain#36349). In that
    # case raw.content still holds the model's text, which the prompt asked
    # to be bare JSON -- hence the fallback below.
    if isinstance(structured_result, TriageDecision):
        decision = structured_result.decision
    else:
        logger.warning("triage_structured_failed=%s falling back to json parse", result.get("parsing_error"))
        decision = None
        if raw_response is not None:
            try:
                content = raw_response.content.strip()
                content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
                content = re.sub(r"\s*```$", "", content).strip()
                parsed = json.loads(content)["decision"]
                td = TriageDecision(decision=parsed)  # validate enum
                decision = td.decision
            except Exception:
                pass
        if decision is None:
            logger.warning("triage_json_fallback_failed defaulting to rag")
            decision = "rag"

    # Found live (2026-08-19): nova-micro misclassified a real mixed message
    # ("hola cuanto cuesta el examen de utero?") as pure "greeting" in ~20%
    # of a 10-sample check, despite the prompt's explicit "medical terms
    # ALWAYS rag, never greeting, even if the message opens with hola" rule
    # -- silently dropping the user's actual question, since generate()'s
    # greeting branch never looks at retrieved content. Same "LLM ignores
    # its own instruction on the easy pattern" failure as LOCATION_RE's
    # off_topic case above -- deterministic override, not more prompt tuning.
    # Gated on _GREETING_CHAIN_RE (not just any "greeting" verdict) so an
    # honest multi-word greeting like "hola buenas" stays "greeting" instead
    # of being punished for not fitting the single-phrase _GREETING_RE.
    if decision == "greeting" and not _GREETING_CHAIN_RE.match(last_human):
        logger.info("triage_greeting_override_mixed_message")
        decision = "rag"

    return {"triage_decision": decision}
