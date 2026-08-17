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
      return {"status": "ok", "index_loaded": pipeline is not None}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
      if pipeline is None:
                raise HTTPException(
                              status_code=503,
                              detail="Vector index not found. Run `python ingest.py` first.",
                )
            result = pipeline.answer(request.question)
    return QueryResponse(**result)
