from langchain_core.documents import Document

from ingest import chunk_documents


def test_chunk_documents_splits_long_text():
    long_text = "Sentence one. " * 100
    docs = [Document(page_content=long_text, metadata={"source": "long.md"})]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "long.md" for c in chunks)
    # reassembled chunks should still contain the original content
    assert "Sentence one." in chunks[0].page_content


def test_chunk_documents_keeps_short_text_as_one_chunk():
    docs = [Document(page_content="Short text.", metadata={"source": "short.md"})]
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=120)

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."


def test_chunk_documents_respects_chunk_size_bound():
    long_text = "word " * 500
    docs = [Document(page_content=long_text, metadata={"source": "long.md"})]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)

    # each chunk should be roughly at or under the requested size
    assert all(len(c.page_content) <= 220 for c in chunks)
