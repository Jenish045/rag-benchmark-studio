from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    documents: list[Document]
    scores: list[float]
    query: str
    retriever: str