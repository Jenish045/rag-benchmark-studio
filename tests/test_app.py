from pathlib import Path

from langchain_core.documents import Document

from app.main import display_documents


def test_app_module_imports():
    assert Path("app/main.py").exists()


def test_document_has_expected_metadata():
    document = Document(
        page_content="Test content.",
        metadata={
            "source": "test.pdf",
            "page_label": "1",
        },
    )

    assert document.metadata["source"] == "test.pdf"
    assert document.metadata["page_label"] == "1"