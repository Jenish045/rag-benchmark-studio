from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_benchmark.chunking.text_chunker import chunk_documents
from rag_benchmark.ingestion.document_loader import load_documents
from rag_benchmark.retrieval.bm25_retriever import BM25Retriever


DOCUMENTS_DIR = Path("data/documents")


def create_retriever(documents, top_k=5):
    return BM25Retriever(
        documents=documents,
        top_k=top_k,
    )


def test_bm25_retriever_builds_index():
    documents = [
        Document(
            page_content="Transformers use attention mechanisms.",
            metadata={"source": "transformers.pdf", "page": 0},
        ),
        Document(
            page_content="Convolutional networks process images.",
            metadata={"source": "cnn.pdf", "page": 0},
        ),
    ]

    retriever = create_retriever(documents)

    assert len(retriever.tokenized_documents) == 2
    assert retriever.index is not None


def test_bm25_retriever_returns_keyword_relevant_results():
    documents = [
        Document(
            page_content="Transformers use self attention mechanisms.",
            metadata={"source": "transformers.pdf", "page": 0},
        ),
        Document(
            page_content="Convolutional networks process visual features.",
            metadata={"source": "cnn.pdf", "page": 0},
        ),
    ]

    retriever = create_retriever(documents)

    results = retriever.retrieve(
        "attention mechanisms",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].metadata["source"] == "transformers.pdf"


def test_bm25_retriever_tokenization_is_case_insensitive():
    documents = [
        Document(
            page_content="Attention ATTENTION attention.",
            metadata={"source": "test.pdf", "page": 0},
        ),
        Document(
            page_content="Images contain visual information.",
            metadata={"source": "images.pdf", "page": 0},
        ),
    ]

    retriever = create_retriever(documents)

    results = retriever.retrieve(
        "attention",
        top_k=1,
    )

    assert results[0].metadata["source"] == "test.pdf"


def test_bm25_retriever_preserves_metadata():
    documents = [
        Document(
            page_content="Attention models relationships between tokens.",
            metadata={
                "source": "attention.pdf",
                "page": 3,
                "page_label": "4",
            },
        ),
        Document(
            page_content="Images contain pixels and visual patterns.",
            metadata={
                "source": "vision.pdf",
                "page": 1,
                "page_label": "2",
            },
        ),
    ]

    retriever = create_retriever(documents)

    results = retriever.retrieve(
        "attention",
        top_k=1,
    )

    assert results[0].metadata["source"] == "attention.pdf"
    assert results[0].metadata["page"] == 3
    assert results[0].metadata["page_label"] == "4"


def test_bm25_retriever_rejects_invalid_query():
    documents = [
        Document(
            page_content="Test document.",
            metadata={"source": "test.pdf"},
        )
    ]

    retriever = create_retriever(documents)

    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        retriever.retrieve("")


def test_bm25_retriever_rejects_invalid_top_k():
    documents = [
        Document(
            page_content="Test document.",
            metadata={"source": "test.pdf"},
        )
    ]

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        create_retriever(
            documents,
            top_k=0,
        )


def test_bm25_retriever_limits_results_to_available_documents():
    documents = [
        Document(
            page_content="Document one about embeddings.",
            metadata={"source": "one.pdf"},
        ),
        Document(
            page_content="Document two about embeddings.",
            metadata={"source": "two.pdf"},
        ),
    ]

    retriever = create_retriever(documents)

    results = retriever.retrieve(
        "embeddings",
        top_k=10,
    )

    assert len(results) == 2


def test_bm25_retriever_rejects_empty_documents():
    with pytest.raises(
        ValueError,
        match="No documents provided",
    ):
        BM25Retriever(documents=[])


def test_bm25_retriever_rejects_documents_without_text():
    documents = [
        Document(
            page_content="",
            metadata={"source": "empty.pdf"},
        )
    ]

    with pytest.raises(
        ValueError,
        match="searchable text",
    ):
        BM25Retriever(documents=documents)


def test_bm25_retriever_on_real_corpus():
    ingestion_result = load_documents(DOCUMENTS_DIR)

    chunking_result = chunk_documents(
        ingestion_result.documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    retriever = create_retriever(
        chunking_result.chunks,
    )

    results = retriever.retrieve(
        "attention mechanism",
        top_k=5,
    )

    assert results
    assert len(results) <= 5

    for document in results:
        assert document.page_content.strip()
        assert "source" in document.metadata
        assert "page" in document.metadata