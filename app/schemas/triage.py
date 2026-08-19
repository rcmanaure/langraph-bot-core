from typing import Literal

from pydantic import BaseModel


class TriageDecision(BaseModel):
    """Classification of the user's latest message into one triage category."""

    decision: Literal["rag", "catalog", "human", "off_topic", "greeting"]
