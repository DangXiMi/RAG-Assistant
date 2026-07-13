from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    question: str
    user_id: str = "default"
    mode: str 

# Define what the response looks like
class QueryResponse(BaseModel):
    answer: str
    sources: list
    context: Optional[list[str]] = None