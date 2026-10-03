from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from rag_benchmark.utils.config import DEFAULT_TOP_K


@dataclass
class DenseRetriever:
    documents: list[Document]
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = DEFAULT_TOP_K
    embedding_model: Any | None = None

    model: Any = field(init=False)
    index: faiss.Index = field(init=False)

    def __post_init__(self) -> None:
        if not self.documents:
            raise ValueError("No documents provided for dense retrieval.")

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if self.embedding_model is None:
            self.model = SentenceTransformer(self.model_name)
        else:
            self.model = self.embedding_model

        embeddings = self._encode(
            [document.page_content for document in self.documents]
        )

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        faiss.normalize_L2(embeddings)

        return embeddings

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[Document]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        k = self.top_k if top_k is None else top_k

        if k <= 0:
            raise ValueError("top_k must be greater than 0.")

        k = min(k, len(self.documents))

        query_embedding = self._encode([query])

        _, indices = self.index.search(
            query_embedding,
            k,
        )

        return [
            self.documents[index]
            for index in indices[0]
            if index >= 0
        ]