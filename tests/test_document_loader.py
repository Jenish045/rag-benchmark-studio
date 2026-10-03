from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_benchmark.ingestion.document_loader import load_documents
from rag_benchmark.ingestion.models import IngestionResult


DOCUMENTS_DIR = Path("data/documents")


def test_load_documents():
    result = load_documents(DOCUMENTS_DIR)

    assert isinstance(result, IngestionResult)
    assert result.documents

    assert all(
        isinstance(document, Document)
        for document in result.documents
    )

    assert result.source_files == [
        "attention_is_all_you_need.pdf",
        "bert.pdf",
        "rag.pdf",
        "sentence_bert.pdf",
    ]

    assert result.total_source_files == 4
    assert result.total_documents == len(result.documents)


def test_document_metadata():
    result = load_documents(DOCUMENTS_DIR)

    for document in result.documents:
        assert "source" in document.metadata
        assert "page" in document.metadata
        assert "page_label" in document.metadata
        assert "total_pages" in document.metadata

        assert document.metadata["page"] >= 0
        assert document.metadata["page_label"] == str(
            document.metadata["page"] + 1
        )
        assert document.metadata["total_pages"] >= 1


def test_missing_directory():
    with pytest.raises(FileNotFoundError):
        load_documents("data/does_not_exist")


def test_file_instead_of_directory(tmp_path):
    file_path = tmp_path / "document.pdf"
    file_path.write_text("test")

    with pytest.raises(NotADirectoryError):
        load_documents(file_path)


def test_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)

def test_complete_ingestion_pipeline():
    result = load_documents(DOCUMENTS_DIR)

    assert result.total_source_files == 4
    assert result.total_documents > 0

    for source_file in result.source_files:
        source_documents = [
            document
            for document in result.documents
            if document.metadata["source"] == source_file
        ]

        assert source_documents

        total_pages = source_documents[0].metadata["total_pages"]

        assert len(source_documents) == total_pages

        page_numbers = [
            document.metadata["page"]
            for document in source_documents
        ]

        assert page_numbers == list(range(total_pages))