import numpy as np
import pytest
from langchain_core.documents import Document

from rag_benchmark.reranking.cross_encoder_reranker import (
    CrossEncoderReranker,
)


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []

        for query, document in pairs:
            query_words = set(query.lower().split())
            document_words = set(document.lower().split())

            overlap = len(query_words & document_words)

            scores.append(float(overlap))

        return np.array(scores, dtype=np.float32)


def create_reranker():
    return CrossEncoderReranker(
        model=FakeCrossEncoder(),
    )


def test_cross_encoder_reranker_initializes():
    reranker = create_reranker()

    assert reranker.model is not None


def test_reranker_returns_top_k_results():
    documents = [
        Document(
            page_content="Images contain visual features.",
            metadata={"source": "vision.pdf"},
        ),
        Document(
            page_content="Transformers use attention mechanisms.",
            metadata={"source": "transformers.pdf"},
        ),
        Document(
            page_content="Attention mechanisms improve language models.",
            metadata={"source": "language.pdf"},
        ),
    ]

    reranker = create_reranker()

    results = reranker.rerank(
        "attention mechanisms",
        documents,
        top_k=2,
    )

    assert len(results) == 2


def test_reranker_orders_documents_by_relevance():
    documents = [
        Document(
            page_content="Images contain visual features.",
            metadata={"source": "vision.pdf"},
        ),
        Document(
            page_content="Transformers use attention mechanisms.",
            metadata={"source": "transformers.pdf"},
        ),
        Document(
            page_content="Attention mechanisms improve language models.",
            metadata={"source": "language.pdf"},
        ),
    ]

    reranker = create_reranker()

    results = reranker.rerank(
        "attention mechanisms",
        documents,
        top_k=1,
    )

    assert results[0].metadata["source"] in {
        "transformers.pdf",
        "language.pdf",
    }


def test_reranker_preserves_metadata():
    documents = [
        Document(
            page_content="Attention mechanisms model relationships.",
            metadata={
                "source": "attention.pdf",
                "page": 4,
                "page_label": "5",
            },
        ),
        Document(
            page_content="Images contain visual patterns.",
            metadata={
                "source": "vision.pdf",
                "page": 2,
                "page_label": "3",
            },
        ),
    ]

    reranker = create_reranker()

    results = reranker.rerank(
        "attention",
        documents,
        top_k=1,
    )

    assert results[0].metadata["source"] == "attention.pdf"
    assert results[0].metadata["page"] == 4
    assert results[0].metadata["page_label"] == "5"


def test_reranker_limits_top_k_to_available_documents():
    documents = [
        Document(
            page_content="Attention mechanisms.",
            metadata={"source": "one.pdf"},
        ),
        Document(
            page_content="Embedding vectors.",
            metadata={"source": "two.pdf"},
        ),
    ]

    reranker = create_reranker()

    results = reranker.rerank(
        "attention",
        documents,
        top_k=10,
    )

    assert len(results) == 2


def test_reranker_rejects_invalid_query():
    documents = [
        Document(
            page_content="Attention mechanisms.",
            metadata={"source": "test.pdf"},
        )
    ]

    reranker = create_reranker()

    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        reranker.rerank(
            "",
            documents,
        )


def test_reranker_rejects_empty_documents():
    reranker = create_reranker()

    with pytest.raises(
        ValueError,
        match="No documents provided",
    ):
        reranker.rerank(
            "attention",
            [],
        )


def test_reranker_rejects_invalid_top_k():
    documents = [
        Document(
            page_content="Attention mechanisms.",
            metadata={"source": "test.pdf"},
        )
    ]

    reranker = create_reranker()

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        reranker.rerank(
            "attention",
            documents,
            top_k=0,
        )


def test_reranker_uses_query_document_pairs():
    class TrackingModel:
        def __init__(self):
            self.received_pairs = None

        def predict(self, pairs):
            self.received_pairs = pairs
            return np.array([1.0] * len(pairs))

    model = TrackingModel()

    reranker = CrossEncoderReranker(
        model=model,
    )

    documents = [
        Document(
            page_content="Attention mechanisms.",
            metadata={"source": "test.pdf"},
        )
    ]

    reranker.rerank(
        "What is attention?",
        documents,
    )

    assert model.received_pairs == [
        ("What is attention?", "Attention mechanisms.")
    ]


def test_reranked_retriever_composes_retrieval_and_reranking():
    from rag_benchmark.reranking.cross_encoder_reranker import RerankedRetriever

    class DummyRetriever:
        def retrieve(self, query, top_k=5):
            return [
                Document(
                    page_content="Images and pixels.",
                    metadata={"source": "vision.pdf"},
                ),
                Document(
                    page_content="Attention mechanisms.",
                    metadata={"source": "attention.pdf"},
                ),
            ]

    reranker = create_reranker()
    pipeline = RerankedRetriever(
        base_retriever=DummyRetriever(),
        reranker=reranker,
        initial_top_k=5,
    )

    results = pipeline.retrieve("attention", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["source"] == "attention.pdf"