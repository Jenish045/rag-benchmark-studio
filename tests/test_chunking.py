from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_benchmark.chunking.text_chunker import chunk_documents
from rag_benchmark.ingestion.document_loader import load_documents
from rag_benchmark.utils.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)


DOCUMENTS_DIR = Path("data/documents")


def test_chunk_documents():
    documents = [
        Document(
            page_content=(
                "This is a test document. "
                "It contains enough text to produce multiple chunks."
            ),
            metadata={
                "source": "test.pdf",
                "page": 0,
                "page_label": "1",
                "total_pages": 1,
            },
        )
    ]

    result = chunk_documents(
        documents,
        chunk_size=30,
        chunk_overlap=5,
    )

    assert result.total_chunks == len(result.chunks)
    assert result.total_chunks > 1
    assert result.source_files == ["test.pdf"]

    assert all(
        isinstance(chunk, Document)
        for chunk in result.chunks
    )

    for chunk in result.chunks:
        assert chunk.page_content.strip()
        assert chunk.metadata["source"] == "test.pdf"


def test_chunk_documents_uses_default_configuration():
    documents = [
        Document(
            page_content="A " * 1000,
            metadata={
                "source": "test.pdf",
                "page": 0,
                "page_label": "1",
                "total_pages": 1,
            },
        )
    ]

    result = chunk_documents(documents)

    assert result.total_chunks > 1
    assert DEFAULT_CHUNK_SIZE > 0
    assert DEFAULT_CHUNK_OVERLAP >= 0
    assert DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE


def test_chunk_documents_rejects_empty_input():
    with pytest.raises(ValueError, match="No documents provided"):
        chunk_documents([])


def test_chunk_documents_rejects_invalid_chunk_size():
    documents = [
        Document(
            page_content="Test document.",
            metadata={"source": "test.pdf"},
        )
    ]

    with pytest.raises(ValueError, match="chunk_size"):
        chunk_documents(
            documents,
            chunk_size=0,
            chunk_overlap=0,
        )


def test_chunk_documents_rejects_invalid_overlap():
    documents = [
        Document(
            page_content="Test document.",
            metadata={"source": "test.pdf"},
        )
    ]

    with pytest.raises(
        ValueError,
        match="smaller than chunk_size",
    ):
        chunk_documents(
            documents,
            chunk_size=100,
            chunk_overlap=100,
        )


def test_chunk_metadata_preserves_source_and_page():
    ingestion_result = load_documents(DOCUMENTS_DIR)

    chunking_result = chunk_documents(
        ingestion_result.documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    assert chunking_result.total_chunks > 0

    expected_sources = set(ingestion_result.source_files)
    actual_sources = set(chunking_result.source_files)

    assert actual_sources == expected_sources

    for chunk in chunking_result.chunks:
        metadata = chunk.metadata

        assert metadata["source"] in expected_sources
        assert isinstance(metadata["page"], int)
        assert metadata["page"] >= 0

        assert metadata["page_label"] == str(
            metadata["page"] + 1
        )

        assert metadata["total_pages"] >= 1
        assert chunk.page_content.strip()