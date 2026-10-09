"""
FastAPI backend exposing the RAG pipeline as a REST API.

Usage:
    uvicorn app:app --reload
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_chain import RAGPipeline

load_dotenv()

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once on startup and keep it for the app's lifetime.

    If the vector index hasn't been built yet (python ingest.py has not been
    run), RAGPipeline() raises FileNotFoundError and `pipeline` is left as
    None so the app still starts; /query then returns a 503 until the index
    exists.
    """
    global pipeline
    try:
        pipeline = RAGPipeline()
    except FileNotFoundError:
        pipeline = None
    yield

app = FastAPI(title="RAG Document Q&A Chatbot", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    """Report service status and whether the vector index has been loaded."""
    return {"status": "ok", "index_loaded": pipeline is not None}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Answer a question against the ingested documents.

    Returns a 503 if the vector index hasn't been built yet (see `lifespan`),
    otherwise delegates to RAGPipeline.answer for retrieval + generation and
    returns the answer along with its source documents.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Vector index not found. Run `python ingest.py` first.",
        )
    result = pipeline.answer(request.question)
    return QueryResponse(**result)
