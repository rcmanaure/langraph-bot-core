"""User-facing strings shared across layers -- both the graph
(app/graph/nodes/generate.py, on an automatic escalation) and the channel
turn (app/channels/turn.py, on a reactive suspend) need to say the same
thing when a thread moves into human control. See
docs/adr/ADR-009-human-control.md.
"""

HUMAN_HANDOFF = "En breve lo va a atender una persona. Por favor, espere un momento."
