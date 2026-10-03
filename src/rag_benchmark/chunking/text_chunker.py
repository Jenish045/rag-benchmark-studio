from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_benchmark.chunking.models import ChunkingResult
from rag_benchmark.utils.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ChunkingResult:
    """
    Split documents into fixed-size overlapping chunks.
    """

    if not documents:
        raise ValueError("No documents provided for chunking.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    source_files = sorted(
        {
            document.metadata["source"]
            for document in chunks
            if "source" in document.metadata
        }
    )

    return ChunkingResult(
        chunks=chunks,
        total_chunks=len(chunks),
        source_files=source_files,
    )