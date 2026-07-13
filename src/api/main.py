import sys 
from pathlib import Path 
project_root = Path(__file__).parent.parent.parent 
sys.path.insert(0, str(project_root))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import uvicorn

from src.api.dependencies import QueryRequest, QueryResponse

from src.ingestion.data_pipeline import load_pipeline

from src.api.routes import ingest, status, query



@asynccontextmanager
async def lifespan(app: FastAPI):

    print(
        "Starting application..."
    )
 
    app.state.pipeline = load_pipeline()

    print(
        "Pipeline loaded"
    )


    yield


    print(
        "Shutdown"
    )



app = FastAPI(
    title="RAG Backend Services",
    lifespan=lifespan,
)



@app.get("/api/v1/health")
async def health():

    return {
        "status": "ok"
    }



app.include_router(
    ingest.router
)

app.include_router(
    status.router
)

app.include_router(
    query.router
)

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )