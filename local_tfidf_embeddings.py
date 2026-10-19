"""
Fully local embeddings fallback (no network access, no model download).

Used when neither an OpenAI API key nor a downloadable HuggingFace model
is available (e.g. offline environments, network-restricted CI/sandboxes).
Fits a TF-IDF vectorizer over the ingested chunks and reduces it to a dense
vector with truncated SVD (LSA), so it can drop into any FAISS-based
LangChain pipeline that expects an `embed_documents` / `embed_query`
interface. It's weaker semantically than a neural embedding model, but it's
dependency-light (scikit-learn only) and keeps the pipeline runnable
end-to-end anywhere.
"""
import os
import pickle

from langchain_core.embeddings import Embeddings
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by TF-IDF + truncated SVD."""

    def __init__(self, n_components: int = 128, vectorizer_path: str | None = None):
        self.n_components = n_components
        self.vectorizer_path = vectorizer_path
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None

    def _fit(self, texts: list[str]):
        """Fit the TF-IDF vectorizer and SVD reducer on a batch of texts.

        Builds a TF-IDF matrix over `texts`, then reduces it to a dense
        vector space via truncated SVD. If `vectorizer_path` is set, the
        fitted vectorizer and SVD model are pickled there so a later
        process can call `embed_query` without re-fitting.

        Args:
            texts: Document chunks to fit the vectorizer on.

        Returns:
            A dense (n_texts, n_components) array of embeddings.
        """
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = self.vectorizer.fit_transform(texts)

        n_comp = max(2, min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        vectors = self.svd.fit_transform(tfidf)

        if self.vectorizer_path:
            os.makedirs(os.path.dirname(self.vectorizer_path), exist_ok=True)
            with open(self.vectorizer_path, "wb") as f:
                pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd}, f)

        return vectors

    def _load(self):
        """Load a previously fitted vectorizer and SVD model from disk.

        Reads the pickle written by `_fit` at `vectorizer_path` and
        restores `self.vectorizer` and `self.svd` so `embed_query`
        can run in a fresh process without re-fitting on the corpus.
        """
        with open(self.vectorizer_path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.svd = data["svd"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Fit on and embed a batch of documents (LangChain interface).

        Args:
            texts: Document chunks to embed.

        Returns:
            One embedding vector (as a list of floats) per input text.
        """
        vectors = self._fit(texts)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string using the fitted model.

        Loads the vectorizer/SVD from `vectorizer_path` first if this
        instance hasn't fit or loaded one yet (e.g. a fresh process that
        only needs to embed queries against an index built earlier).

        Args:
            text: The query string to embed.

        Returns:
            The query's embedding vector as a list of floats.
        """
        if self.vectorizer is None or self.svd is None:
            self._load()
        tfidf = self.vectorizer.transform([text])
        vector = self.svd.transform(tfidf)
        return vector[0].tolist()
