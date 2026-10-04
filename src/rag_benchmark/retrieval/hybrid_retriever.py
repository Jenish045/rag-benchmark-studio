from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document

from rag_benchmark.retrieval.bm25_retriever import BM25Retriever
from rag_benchmark.retrieval.dense_retriever import DenseRetriever
from rag_benchmark.utils.config import DEFAULT_TOP_K


@dataclass
class HybridRetriever:
    dense_retriever: DenseRetriever
    bm25_retriever: BM25Retriever
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    top_k: int = DEFAULT_TOP_K

    def __post_init__(self) -> None:
        if self.dense_retriever.documents != self.bm25_retriever.documents:
            raise ValueError(
                "Dense and BM25 retrievers must use the same documents."
            )

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if self.dense_weight < 0 or self.bm25_weight < 0:
            raise ValueError("Retrieval weights cannot be negative.")

        if self.dense_weight + self.bm25_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than 0."
            )

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32)

        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:
            return np.ones_like(scores)

        return (scores - minimum) / (maximum - minimum)

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

        k = min(k, len(self.dense_retriever.documents))

        dense_documents = self.dense_retriever.documents
        bm25_documents = self.bm25_retriever.documents

        dense_embeddings = self.dense_retriever._encode([query])
        dense_scores, _ = self.dense_retriever.index.search(
            dense_embeddings,
            len(dense_documents),
        )

        bm25_scores = self.bm25_retriever.index.get_scores(
            self.bm25_retriever._tokenize(query)
        )

        dense_scores = dense_scores[0]

        dense_normalized = self._normalize_scores(dense_scores)
        bm25_normalized = self._normalize_scores(bm25_scores)

        combined_scores = (
            self.dense_weight * dense_normalized
            + self.bm25_weight * bm25_normalized
        )

        ranked_indices = np.argsort(
            combined_scores
        )[::-1][:k]

        return [
            dense_documents[index]
            for index in ranked_indices
        ]