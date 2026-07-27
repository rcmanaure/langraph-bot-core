from pydantic import BaseModel


class RewrittenQuery(BaseModel):
    query: str
