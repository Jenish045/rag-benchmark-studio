from pathlib import Path

from langchain_core.documents import Document

from rag_benchmark.ingestion.models import IngestionResult
from rag_benchmark.ingestion.pdf_loader import load_pdf


def load_documents(directory: str | Path) -> IngestionResult:
    """
    Load all PDF documents from a directory.

    Returns an IngestionResult containing the loaded pages
    and corpus-level information.
    """

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory not found: {directory_path}"
        )

    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, got: {directory_path}"
        )

    pdf_files = sorted(directory_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {directory_path}"
        )

    documents: list[Document] = []

    for pdf_file in pdf_files:
        documents.extend(load_pdf(pdf_file))

    source_files = [pdf_file.name for pdf_file in pdf_files]

    return IngestionResult(
        documents=documents,
        source_files=source_files,
        total_documents=len(documents),
        total_source_files=len(source_files),
    )