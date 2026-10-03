from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class IngestionResult:
    documents: list[Document]
    source_files: list[str]
    total_documents: int
    total_source_files: int