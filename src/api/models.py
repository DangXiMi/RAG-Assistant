from pydantic import BaseModel
from typing import Any, Optional

class QueryRequest(BaseModel):
    question: str
    user_id: str = "default"
    mode: str 

class SourceRef(BaseModel):
    """Structured citation for one retrieved chunk."""
    ref: int
    source: str
    location: str = ""
    label: str
    page: Optional[int] = None
    sheet: Optional[str] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None

# Define what the response looks like
class QueryResponse(BaseModel):
    answer: str
    # Plain citation labels, e.g. "refund_policy.pdf, p.3". The list[str]
    # contract is unchanged; only the content is now human-readable.
    sources: list[str]
    contexts: Optional[list[str]] = None
    # Additive fields: older clients that ignore them keep working.
    source_refs: Optional[list[SourceRef]] = None
    retrieved: Optional[int] = None
    answered: Optional[bool] = None