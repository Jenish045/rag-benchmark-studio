from dataclasses import dataclass, field
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_benchmark.utils.config import DEFAULT_TOP_K


@dataclass
class BM25Retriever:
    documents: list[Document]
    top_k: int = DEFAULT_TOP_K

    tokenized_documents: list[list[str]] = field(init=False)
    index: BM25Okapi = field(init=False)

    def __post_init__(self) -> None:
        if not self.documents:
            raise ValueError("No documents provided for BM25 retrieval.")

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        self.tokenized_documents = [
            self._tokenize(document.page_content)
            for document in self.documents
        ]

        if any(not tokens for tokens in self.tokenized_documents):
            raise ValueError(
                "All documents must contain searchable text."
            )

        self.index = BM25Okapi(self.tokenized_documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not isinstance(text, str):
            raise ValueError("text must be a string.")

        return re.findall(r"\b\w+\b", text.lower())

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

        query_tokens = self._tokenize(query)

        if not query_tokens:
            raise ValueError("query must contain searchable text.")

        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:k]

        return [
            self.documents[index]
            for index in ranked_indices
        ]