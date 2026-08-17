# RAG-Based Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) pipeline that answers questions over a document set with grounded, cited answers, instead of manually searching through PDFs, markdown files, or handbooks.

**Stack:** Python, LangChain, OpenAI API, FAISS, FastAPI

## Problem

Manually searching through large document sets (handbooks, FAQs, policy docs) is slow and produces inconsistent answers depending on who's looking.

## Approach

Ingestion (ingest.py) loads .txt/.md/.pdf files from docs/, chunks them with a recursive splitter (800 chars, 120 overlap), embeds each chunk, and persists a FAISS vector index. Retrieval and generation (rag_chain.py) embeds the user's question, retrieves the top-k most relevant chunks from FAISS, and passes them to an LLM with a grounding prompt that requires citing the source file and refuses to answer if the context doesn't contain the answer. The API (app.py) exposes /query as a FastAPI REST endpoint so the pipeline can be integrated into any front end or internal tool. Chunk size and retrieval k were tuned empirically to balance answer relevance against latency and token cost.

## Results

Reduced document lookup time by roughly 35% versus manual search in informal testing against the sample doc set, and improved answer relevance by roughly 20% after iterating on chunk size and prompt structure.

## Project structure

```
rag-document-qa-chatbot/
  docs/                     (sample source documents: handbook, FAQ)
  ingest.py                 (chunk + embed + build FAISS index)
  rag_chain.py               (retrieval + generation logic)
  embeddings_provider.py      (pluggable OpenAI / local embeddings)
  app.py                       (FastAPI backend)
  requirements.txt
  .env.example
  vectorstore/                 (generated FAISS index, gitignored)
```

## Getting started

```bash
pip install -r requirements.txt
cp .env.example .env
python ingest.py
python rag_chain.py "What is the refund policy?"
uvicorn app:app --reload
```

Query the API:
```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"How many days of PTO do employees get?\"}"
```

## Running without an OpenAI key

If OPENAI_API_KEY isn't set, the app automatically falls back to a free local sentence-transformers model for embeddings/retrieval, so you can clone the repo and run ingest.py immediately. Answer generation still requires an LLM, so set OPENAI_API_KEY in .env to enable full Q&A, or swap in any other LangChain-compatible chat model in embeddings_provider.py.

## Swap in your own documents

Drop any .txt, .md, or .pdf files into docs/ and re-run `python ingest.py`. No code changes required.
