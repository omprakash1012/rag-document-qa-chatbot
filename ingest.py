"""
Ingestion pipeline: loads documents from docs/, chunks them, embeds them,
and persists a FAISS vector index to disk.

Usage:
    python ingest.py --docs-dir docs --index-dir vectorstore
    """
import argparse
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS

from embeddings_provider import get_embeddings

LOADER_MAP = {
      ".txt": TextLoader,
      ".md": TextLoader,
      ".pdf": PyPDFLoader,
}


def load_documents(docs_dir: str):
      documents = []
      for ext, loader_cls in LOADER_MAP.items():
                loader = DirectoryLoader(
                              docs_dir, glob=f"**/*{ext}", loader_cls=loader_cls, show_progress=False
                )
                documents.extend(loader.load())
            return documents


def chunk_documents(documents, chunk_size=800, chunk_overlap=120):
      splitter = RecursiveCharacterTextSplitter(
          chunk_size=chunk_size,
          chunk_overlap=chunk_overlap,
          separators=["\n\n", "\n", ". ", " ", ""],
)
    return splitter.split_documents(documents)


def build_index(docs_dir: str, index_dir: str):
      print(f"Loading documents from {docs_dir}...")
    documents = load_documents(docs_dir)
    if not documents:
              raise ValueError(f"No .txt/.md/.pdf files found in {docs_dir}")
          print(f"Loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    print("Embedding chunks and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"Saved FAISS index -> {index_dir}")


if __name__ == "__main__":
      parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--index-dir", default="vectorstore")
    args = parser.parse_args()
    build_index(args.docs_dir, args.index_dir)
