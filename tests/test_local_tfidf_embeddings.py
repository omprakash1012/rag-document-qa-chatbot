from local_tfidf_embeddings import TfidfEmbeddings


def test_embed_documents_returns_one_vector_per_text():
    texts = [
        "refund policy allows returns within 30 days",
        "employees get vacation days and sick leave",
        "api rate limits apply to the free tier",
    ]
    emb = TfidfEmbeddings(n_components=2)
    vectors = emb.embed_documents(texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == len(vectors[0]) for v in vectors)


def test_embed_query_matches_document_vector_dimension():
    texts = [
        "refund policy allows returns within 30 days",
        "employees get vacation days and sick leave",
    ]
    emb = TfidfEmbeddings(n_components=2)
    doc_vectors = emb.embed_documents(texts)
    query_vector = emb.embed_query("what is the refund policy")

    assert len(query_vector) == len(doc_vectors[0])


def test_embed_query_loads_persisted_vectorizer(tmp_path):
    texts = [
        "refund policy allows returns within 30 days",
        "employees get vacation days and sick leave",
    ]
    path = str(tmp_path / "tfidf.pkl")

    fit_emb = TfidfEmbeddings(n_components=2, vectorizer_path=path)
    fit_emb.embed_documents(texts)

    # a fresh instance with no fitted state should load from disk
    fresh_emb = TfidfEmbeddings(n_components=2, vectorizer_path=path)
    vector = fresh_emb.embed_query("refund")

    assert len(vector) == 2


def test_embed_query_without_fit_or_saved_path_raises():
    emb = TfidfEmbeddings(n_components=2, vectorizer_path=None)
    try:
        emb.embed_query("anything")
        assert False, "expected an error when no vectorizer has been fit or saved"
    except (TypeError, FileNotFoundError):
        pass
