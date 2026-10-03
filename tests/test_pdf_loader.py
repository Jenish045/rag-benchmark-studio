from pathlib import Path

from langchain_core.documents import Document

from rag_benchmark.ingestion.pdf_loader import load_pdf


DOCUMENTS_DIR = Path("data/documents")


def test_load_pdf():
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    assert pdf_files, "No PDF files found in data/documents"

    documents = load_pdf(pdf_files[0])

    assert documents
    assert all(isinstance(document, Document) for document in documents)
    assert all(document.page_content.strip() for document in documents)

    for document in documents:
        assert document.metadata["source"] == pdf_files[0].name
        assert isinstance(document.metadata["page"], int)
        assert document.metadata["page_label"] == str(
            document.metadata["page"] + 1
        )
        assert document.metadata["total_pages"] == len(documents)