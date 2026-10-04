from langchain_core.documents import Document


def build_context(documents: list[Document]) -> str:
    if not documents:
        raise ValueError("No documents provided for context building.")

    context_parts = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page_label")

        if page is not None:
            citation = f"{source}, page {page}"
        else:
            citation = source

        context_parts.append(
            f"[Source {index}: {citation}]\n"
            f"{document.page_content.strip()}"
        )

    return "\n\n".join(context_parts)