from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class ChunkingResult:
    chunks: list[Document]
    total_chunks: int
    source_files: list[str]