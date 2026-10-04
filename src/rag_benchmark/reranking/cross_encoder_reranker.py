from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


@dataclass
class CrossEncoderReranker:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model: Any | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if not documents:
            raise ValueError("No documents provided for reranking.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        top_k = min(top_k, len(documents))

        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked_indices = sorted(
            range(len(documents)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        return [
            documents[index]
            for index in ranked_indices
        ]