from pydantic import BaseModel


class ProfileExtraction(BaseModel):
    new_topic: str | None = None
