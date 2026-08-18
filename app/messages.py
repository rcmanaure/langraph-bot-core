"""User-facing strings shared across layers -- both the graph
(app/graph/nodes/generate.py, on an automatic escalation) and the channel
turn (app/channels/turn.py, on a reactive suspend) need to say the same
thing when a thread moves into human control. See
docs/adr/ADR-009-human-control.md.
"""

HUMAN_HANDOFF = "En breve lo va a atender una persona. Por favor, espere un momento."

# Sent only to a thread the scheduler auto-expires (app/scheduler.py) --
# reactively suspended, past the TTL, and no operator ever claimed it (see
# #39). A claimed thread never hits this: the first operator message stops
# the clock.
HUMAN_CONTROL_EXPIRED = (
    "Lo sentimos, no pudimos conectarlo con una persona a tiempo. "
    "¿En qué más podemos ayudarle?"
)
