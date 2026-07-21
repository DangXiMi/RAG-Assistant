from fastapi import (APIRouter, UploadFile, File, HTTPException, Request)
from src.api.models import QueryResponse, QueryRequest

router = APIRouter() 

@router.post("/api/v1/query", response_model=QueryResponse)
async def handle_rag_query(payload: QueryRequest, request: Request):
    try:
        pipeline = request.app.state.pipeline
        
        questions = payload.question
        user_id = payload.user_id
        mode = payload.mode

        retrievers = pipeline["retrievers"][mode]
        generator = pipeline["generator"]
        
        answer = generator.run(query = questions, retriever = retrievers, top_k=3)
        
        return QueryResponse(answer = answer["answer"],
                             sources = answer["sources"],
                             contexts = answer["contexts"]
                             )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
