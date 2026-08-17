"""
Pluggable embeddings provider.

Three tiers, tried in order:
1. OpenAI embeddings when OPENAI_API_KEY is set (production path, matches
   the resume's OpenAI API + LangChain stack).
2. A free local sentence-transformers model, downloaded from HuggingFace
   Hub on first use.
3. A fully local TF-IDF + SVD embedding (see local_tfidf_embeddings.py) that
   needs no network access at all, for offline or network-restricted
   environments where tier 2's model download isn't possible.

This means the project can always be cloned and run end-to-end, with the
best available option for whatever environment it's running in.
"""
import os


def get_embeddings(index_dir: str = "vectorstore"):
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"HuggingFace embeddings unavailable ({e.__class__.__name__}), "
              f"falling back to local TF-IDF embeddings.")

    from local_tfidf_embeddings import TfidfEmbeddings
    return TfidfEmbeddings(vectorizer_path=os.path.join(index_dir, "tfidf.pkl"))


def get_llm(temperature: float = 0.0):
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    raise RuntimeError(
        "No OPENAI_API_KEY set. Set it in a .env file to enable answer generation. "
        "Retrieval (chunk search) works without it, but generation needs an LLM."
    )
