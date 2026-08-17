import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import rag_chain
from local_tfidf_embeddings import TfidfEmbeddings
from rag_chain import RAGPipeline


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    """Build a tiny real FAISS index using the TF-IDF embeddings, and force
    RAGPipeline to load it with the *same* embeddings backend. This keeps
    the test deterministic and network-free regardless of whether an
    OpenAI key or HuggingFace download is available in the environment
    running the tests."""
    vectorizer_path = str(tmp_path / "tfidf.pkl")
    docs = [
        Document(
            page_content="Customers can request a full refund within 30 days of purchase.",
            metadata={"source": "product_faq.md"},
        ),
        Document(
            page_content="Full-time employees accrue 15 days of PTO per year.",
            metadata={"source": "employee_handbook.md"},
        ),
    ]
    build_embeddings = TfidfEmbeddings(n_components=2, vectorizer_path=vectorizer_path)
    vectorstore = FAISS.from_documents(docs, build_embeddings)
    vectorstore.save_local(str(tmp_path))

    def fake_get_embeddings(index_dir=None):
        return TfidfEmbeddings(n_components=2, vectorizer_path=vectorizer_path)

    monkeypatch.setattr(rag_chain, "get_embeddings", fake_get_embeddings)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return str(tmp_path)


def test_retrieve_returns_the_relevant_document(index_dir):
    rag = RAGPipeline(index_dir=index_dir, k=1)
    docs = rag.retrieve("What is the refund policy?")

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "product_faq.md"


def test_answer_falls_back_to_retrieval_only_without_api_key(index_dir):
    rag = RAGPipeline(index_dir=index_dir, k=1)
    result = rag.answer("How many days of PTO do employees get?")

    assert result["mode"] == "retrieval-only"
    assert result["sources"] == ["employee_handbook.md"]
    assert "PTO" in result["answer"]


def test_format_context_includes_source_filenames(index_dir):
    rag = RAGPipeline(index_dir=index_dir, k=2)
    docs = rag.retrieve("refund")
    context = rag._format_context(docs)

    assert "[product_faq.md]" in context or "[employee_handbook.md]" in context


def test_missing_index_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        RAGPipeline(index_dir=str(tmp_path / "does-not-exist"))
