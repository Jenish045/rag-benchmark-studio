import numpy as np
import pytest
from langchain_core.documents import Document

from rag_benchmark.retrieval.bm25_retriever import BM25Retriever
from rag_benchmark.retrieval.dense_retriever import DenseRetriever
from rag_benchmark.retrieval.hybrid_retriever import HybridRetriever


class FakeEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True):
        vectors = []

        for text in texts:
            text = text.lower()

            vectors.append(
                [
                    float("transformer" in text or "attention" in text),
                    float("convolution" in text or "image" in text),
                    float("embedding" in text or "vector" in text),
                ]
            )

        return np.array(vectors, dtype=np.float32)


def create_retrievers(documents):
    embedding_model = FakeEmbeddingModel()

    dense = DenseRetriever(
        documents=documents,
        top_k=5,
        embedding_model=embedding_model,
    )

    bm25 = BM25Retriever(
        documents=documents,
        top_k=5,
    )

    return dense, bm25


def test_hybrid_retriever_builds():
    documents = [
        Document(
            page_content="Transformers use attention mechanisms.",
            metadata={"source": "transformers.pdf"},
        ),
        Document(
            page_content="Convolutional networks process images.",
            metadata={"source": "cnn.pdf"},
        ),
    ]

    dense, bm25 = create_retrievers(documents)

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    assert retriever.dense_weight == 0.5
    assert retriever.bm25_weight == 0.5


def test_hybrid_retriever_returns_results():
    documents = [
        Document(
            page_content="Transformers use self attention mechanisms.",
            metadata={"source": "transformers.pdf"},
        ),
        Document(
            page_content="Convolutional networks process visual features.",
            metadata={"source": "cnn.pdf"},
        ),
    ]

    dense, bm25 = create_retrievers(documents)

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    results = retriever.retrieve(
        "attention mechanisms",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].metadata["source"] == "transformers.pdf"


def test_hybrid_retriever_combines_retrieval_signals():
    documents = [
        Document(
            page_content="Transformers use attention mechanisms.",
            metadata={"source": "transformers.pdf"},
        ),
        Document(
            page_content="Attention is useful for language models.",
            metadata={"source": "language.pdf"},
        ),
        Document(
            page_content="Images contain visual features.",
            metadata={"source": "vision.pdf"},
        ),
    ]

    dense, bm25 = create_retrievers(documents)

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        dense_weight=0.5,
        bm25_weight=0.5,
    )

    results = retriever.retrieve(
        "attention",
        top_k=2,
    )

    assert len(results) == 2
    assert all(
        document.metadata["source"] != "vision.pdf"
        for document in results
    )


def test_hybrid_retriever_preserves_metadata():
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

    dense, bm25 = create_retrievers(documents)

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    results = retriever.retrieve(
        "attention",
        top_k=1,
    )

    assert results[0].metadata["source"] == "attention.pdf"
    assert results[0].metadata["page"] == 3
    assert results[0].metadata["page_label"] == "4"


def test_hybrid_retriever_rejects_invalid_query():
    documents = [
        Document(
            page_content="Test document.",
            metadata={"source": "test.pdf"},
        )
    ]

    dense, bm25 = create_retrievers(documents)

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        retriever.retrieve("")


def test_hybrid_retriever_rejects_invalid_top_k():
    documents = [
        Document(
            page_content="Test document.",
            metadata={"source": "test.pdf"},
        )
    ]

    dense, bm25 = create_retrievers(documents)

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        retriever.retrieve(
            "test",
            top_k=0,
        )


def test_hybrid_retriever_rejects_negative_weights():
    documents = [
        Document(
            page_content="Test document.",
            metadata={"source": "test.pdf"},
        )
    ]

    dense, bm25 = create_retrievers(documents)

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            dense_weight=-0.1,
        )


def test_hybrid_retriever_rejects_zero_weights():
    documents = [
        Document(
            page_content="Test document.",
            metadata={"source": "test.pdf"},
        )
    ]

    dense, bm25 = create_retrievers(documents)

    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            dense_weight=0,
            bm25_weight=0,
        )


def test_hybrid_retriever_rejects_different_documents():
    dense_documents = [
        Document(
            page_content="Attention mechanisms.",
            metadata={"source": "one.pdf"},
        )
    ]

    bm25_documents = [
        Document(
            page_content="Different document.",
            metadata={"source": "two.pdf"},
        )
    ]

    dense = DenseRetriever(
        documents=dense_documents,
        embedding_model=FakeEmbeddingModel(),
    )

    bm25 = BM25Retriever(
        documents=bm25_documents,
    )

    with pytest.raises(
        ValueError,
        match="same documents",
    ):
        HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
        )


def test_score_normalization():
    scores = np.array([1.0, 2.0, 3.0])

    normalized = HybridRetriever._normalize_scores(scores)

    assert np.allclose(
        normalized,
        np.array([0.0, 0.5, 1.0]),
    )


def test_constant_scores_are_handled():
    scores = np.array([5.0, 5.0, 5.0])

    normalized = HybridRetriever._normalize_scores(scores)

    assert np.allclose(
        normalized,
        np.ones(3),
    )