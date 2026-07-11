import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
import uvicorn
from contextlib import asynccontextmanager

from src.ingestion.data_pipeline import load_pipeline
from app_comp import QueryRequest, QueryResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application...")
    global pipeline
    pipeline = load_pipeline()
    print("Load pipeline successful")
    
    yield
    
    print("Shutting down application...")
    
app = FastAPI(title="RAG Backend Services", lifespan=lifespan)
    


@app.post("/api/v1/query", response_model=QueryResponse)
async def handle_rag_query(payload: QueryRequest):
    try:
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
    
@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)