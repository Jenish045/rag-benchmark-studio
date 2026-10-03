from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load a PDF and return one LangChain Document per page.

    The returned documents preserve the original page content and
    normalize metadata required by the RAG pipeline.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    for document in documents:
        page_number = document.metadata.get("page", 0)

        document.metadata["source"] = path.name
        document.metadata["page"] = page_number
        document.metadata["page_label"] = str(page_number + 1)
        document.metadata["total_pages"] = len(documents)

    return documents