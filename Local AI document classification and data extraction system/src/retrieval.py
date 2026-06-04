"""
Retrieval Engine: Semantic search over documents using local embeddings.
Uses SentenceTransformers (all-MiniLM-L6-v2) + FAISS for vector search.
Falls back to TF-IDF cosine similarity if SentenceTransformers unavailable.
"""

import os
import re
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# ─── EMBEDDING BACKEND ─────────────────────────────────────────────────────────

class EmbeddingBackend:
    """Abstract embedding backend with fallback."""

    def __init__(self):
        self.backend_name = None
        self._model = None
        self._vectorizer = None
        self._tfidf_matrix = None
        self._init_backend()

    def _init_backend(self):
        # Try SentenceTransformers first
        try:
            from sentence_transformers import SentenceTransformer
            print("  Loading SentenceTransformers (all-MiniLM-L6-v2)...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self.backend_name = "sentence-transformers"
            print("  ✓ SentenceTransformers loaded")
            return
        except Exception as e:
            logging.warning(f"SentenceTransformers unavailable: {e}")

        # Fallback: TF-IDF via scikit-learn
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self.backend_name = "tfidf"
            print("  ✓ TF-IDF fallback ready")
            return
        except Exception as e:
            logging.error(f"TF-IDF also unavailable: {e}")

        raise RuntimeError("No embedding backend available. Install sentence-transformers or scikit-learn.")

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend_name == "sentence-transformers":
            vecs = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return vecs.astype("float32")
        else:
            # TF-IDF: fit or transform
            from sklearn.feature_extraction.text import TfidfVectorizer
            if self._tfidf_matrix is None:
                # First call = fit
                mat = self._vectorizer.fit_transform(texts).toarray().astype("float32")
                self._tfidf_matrix = mat
                return mat
            else:
                # Query call = transform only
                return self._vectorizer.transform(texts).toarray().astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        if self.backend_name == "sentence-transformers":
            vec = self._model.encode([query], show_progress_bar=False, convert_to_numpy=True)
            return vec.astype("float32")
        else:
            # For TF-IDF, fit includes query so vocab is shared; we re-fit with all texts + query
            # Instead, we just transform with the already-fitted vectorizer
            return self._vectorizer.transform([query]).toarray().astype("float32")

    def encode_query_bm25(self, query: str, doc_texts: List[str]) -> np.ndarray:
        """BM25-style keyword overlap scoring (pure numpy, no deps)."""
        query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
        scores = []
        for text in doc_texts:
            words = set(re.sub(r'[^\w\s]', '', text.lower()).split())
            overlap = len(query_words & words)
            scores.append(float(overlap))
        return np.array(scores, dtype="float32")


# ─── FAISS INDEX ───────────────────────────────────────────────────────────────

class RetrievalEngine:
    """Build a vector index from documents and run semantic search queries."""

    def __init__(self):
        self.embedder = EmbeddingBackend()
        self.index = None
        self.doc_names: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_classes: Dict[str, str] = {}

    def build_index(self, documents: Dict[str, str], classifications: Dict[str, dict]):
        """
        Build FAISS (or cosine) index from {filename: text} mapping.
        classifications: the output from processor.process_documents()
        """
        self.doc_names = list(documents.keys())
        self.doc_texts = list(documents.values())
        self.doc_classes = {k: v.get("class", "Unknown") for k, v in classifications.items()}

        print(f"\n  Building index for {len(self.doc_texts)} documents...")
        embeddings = self.embedder.encode(self.doc_texts)

        if self.embedder.backend_name == "sentence-transformers":
            try:
                import faiss
                dim = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalise)
                # Normalise for cosine similarity
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings)
                self._index_type = "faiss"
                print(f"  ✓ FAISS index built (dim={dim})")
                return
            except ImportError:
                logging.warning("FAISS not available, falling back to numpy cosine.")

        # Numpy fallback
        self._np_embeddings = embeddings
        self._index_type = "numpy"
        print("  ✓ Numpy cosine index built")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Search for query. Returns list of (filename, score, doc_class).
        Uses FAISS/cosine when available, falls back to BM25 keyword overlap.
        """
        if self.index is None and not hasattr(self, '_np_embeddings'):
            raise RuntimeError("Index not built. Call build_index() first.")

        if self._index_type == "faiss":
            import faiss
            q_emb = self.embedder.encode_query(query)
            faiss.normalize_L2(q_emb)
            scores, indices = self.index.search(q_emb, min(top_k, len(self.doc_names)))
            results = [
                (self.doc_names[i], float(scores[0][j]), self.doc_classes.get(self.doc_names[i], "?"))
                for j, i in enumerate(indices[0]) if i >= 0
            ]
        else:
            # Try TF-IDF cosine first
            q_emb = self.embedder.encode_query(query)
            doc_norms = np.linalg.norm(self._np_embeddings, axis=1, keepdims=True) + 1e-10
            q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10
            doc_emb_n = self._np_embeddings / doc_norms
            q_emb_n = q_emb / q_norm
            sims = (doc_emb_n @ q_emb_n.T).flatten()

            # If all near-zero, fall back to BM25 keyword overlap
            if sims.max() < 0.01:
                sims = self.embedder.encode_query_bm25(query, self.doc_texts)

            top_idx = np.argsort(sims)[::-1][:top_k]
            results = [
                (self.doc_names[i], float(sims[i]), self.doc_classes.get(self.doc_names[i], "?"))
                for i in top_idx
            ]

        return results
