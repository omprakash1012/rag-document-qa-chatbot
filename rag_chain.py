"""
Core RAG logic: loads the FAISS index, retrieves relevant chunks for a
query, and generates a grounded answer with citations back to source docs.
"""
import os

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from embeddings_provider import get_embeddings, get_llm

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the
provided context. If the answer isn't in the context, say you don't know —
do not make anything up.

Context:
{context}

Question: {question}

Answer concisely, and cite the source filename(s) you used in square brackets, e.g. [handbook.md].
"""


class RAGPipeline:
    def __init__(self, index_dir: str = "vectorstore", k: int = 4):
        if not os.path.exists(index_dir):
            raise FileNotFoundError(
                f"No index found at {index_dir}. Run `python ingest.py` first."
            )
        embeddings = get_embeddings(index_dir)
        self.vectorstore = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def retrieve(self, query: str):
        docs = self.retriever.invoke(query)
        return docs

    def _format_context(self, docs):
        parts = []
        for d in docs:
            src = os.path.basename(d.metadata.get("source", "unknown"))
            parts.append(f"[{src}]\n{d.page_content}")
        return "\n\n---\n\n".join(parts)

    def answer(self, query: str):
        docs = self.retrieve(query)
        context = self._format_context(docs)
        sources = sorted({os.path.basename(d.metadata.get("source", "")) for d in docs})

        try:
            llm = get_llm()
        except RuntimeError:
            # No OPENAI_API_KEY configured, so there's no LLM to generate a
            # natural-language answer. Retrieval still works end-to-end
            # without any paid API, so fall back to returning the top
            # retrieved excerpt instead of a generated answer.
            excerpt = docs[0].page_content.strip() if docs else ""
            return {"answer": excerpt, "sources": sources, "mode": "retrieval-only"}

        chain = self.prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": query})
        return {"answer": response, "sources": sources, "mode": "generated"}


if __name__ == "__main__":
    import sys

    rag = RAGPipeline()
    question = sys.argv[1] if len(sys.argv) > 1 else "What is this document collection about?"
    result = rag.answer(question)
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"Sources: {', '.join(result['sources'])}")
    print(f"Mode: {result['mode']}")
