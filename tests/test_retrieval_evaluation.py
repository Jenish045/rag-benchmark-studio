import pytest
from langchain_core.documents import Document

from rag_benchmark.evaluation.models import RetrievalQuery
from rag_benchmark.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
)
from rag_benchmark.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def document(source: str) -> Document:
    return Document(
        page_content=f"Content from {source}",
        metadata={"source": source},
    )


def test_recall_at_k():
    retrieved = [
        document("a.pdf"),
        document("b.pdf"),
        document("c.pdf"),
    ]

    score = recall_at_k(
        retrieved,
        ["a.pdf", "c.pdf"],
        k=3,
    )

    assert score == 1.0


def test_recall_at_k_partial():
    retrieved = [
        document("a.pdf"),
        document("b.pdf"),
        document("c.pdf"),
    ]

    score = recall_at_k(
        retrieved,
        ["a.pdf", "d.pdf"],
        k=3,
    )

    assert score == 0.5


def test_recall_at_k_respects_k():
    retrieved = [
        document("a.pdf"),
        document("b.pdf"),
        document("c.pdf"),
    ]

    score = recall_at_k(
        retrieved,
        ["c.pdf"],
        k=2,
    )

    assert score == 0.0


def test_hit_rate_at_k():
    retrieved = [
        document("wrong.pdf"),
        document("correct.pdf"),
        document("other.pdf"),
    ]

    score = hit_rate_at_k(
        retrieved,
        ["correct.pdf"],
        k=2,
    )

    assert score == 1.0


def test_hit_rate_at_k_miss():
    retrieved = [
        document("wrong.pdf"),
        document("other.pdf"),
    ]

    score = hit_rate_at_k(
        retrieved,
        ["correct.pdf"],
        k=2,
    )

    assert score == 0.0


def test_reciprocal_rank_first_result():
    retrieved = [
        document("correct.pdf"),
        document("wrong.pdf"),
    ]

    score = reciprocal_rank(
        retrieved,
        ["correct.pdf"],
    )

    assert score == 1.0


def test_reciprocal_rank_third_result():
    retrieved = [
        document("wrong1.pdf"),
        document("wrong2.pdf"),
        document("correct.pdf"),
    ]

    score = reciprocal_rank(
        retrieved,
        ["correct.pdf"],
    )

    assert score == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_for_miss():
    retrieved = [
        document("wrong.pdf"),
    ]

    score = reciprocal_rank(
        retrieved,
        ["correct.pdf"],
    )

    assert score == 0.0


def test_mean_reciprocal_rank():
    rankings = [
        [
            document("correct1.pdf"),
            document("wrong.pdf"),
        ],
        [
            document("wrong.pdf"),
            document("correct2.pdf"),
        ],
    ]

    score = mean_reciprocal_rank(
        rankings,
        [
            ["correct1.pdf"],
            ["correct2.pdf"],
        ],
    )

    assert score == pytest.approx(0.75)


class FakeRetriever:
    def __init__(self):
        self.documents = {
            "attention": [
                document("attention_is_all_you_need.pdf"),
                document("bert.pdf"),
            ],
            "embeddings": [
                document("sentence_bert.pdf"),
                document("bert.pdf"),
            ],
        }

    def retrieve(self, query, top_k=5):
        return self.documents[query][:top_k]


def test_retrieval_evaluator():
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever(),
    )

    queries = [
        RetrievalQuery(
            query="attention",
            relevant_sources=[
                "attention_is_all_you_need.pdf",
            ],
        ),
        RetrievalQuery(
            query="embeddings",
            relevant_sources=[
                "sentence_bert.pdf",
            ],
        ),
    ]

    metrics = evaluator.evaluate(
        queries,
        k=2,
    )

    assert metrics.recall_at_k == 1.0
    assert metrics.hit_rate_at_k == 1.0
    assert metrics.mrr == 1.0


def test_retrieval_evaluator_handles_later_relevant_result():
    class LaterResultRetriever:
        def retrieve(self, query, top_k=5):
            return [
                document("wrong.pdf"),
                document("correct.pdf"),
            ]

    evaluator = RetrievalEvaluator(
        retriever=LaterResultRetriever(),
    )

    queries = [
        RetrievalQuery(
            query="test",
            relevant_sources=["correct.pdf"],
        )
    ]

    metrics = evaluator.evaluate(
        queries,
        k=2,
    )

    assert metrics.recall_at_k == 1.0
    assert metrics.hit_rate_at_k == 1.0
    assert metrics.mrr == 0.5


def test_retrieval_evaluator_rejects_empty_queries():
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever(),
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        evaluator.evaluate([])


def test_retrieval_evaluator_rejects_invalid_k():
    evaluator = RetrievalEvaluator(
        retriever=FakeRetriever(),
    )

    queries = [
        RetrievalQuery(
            query="attention",
            relevant_sources=[
                "attention_is_all_you_need.pdf"
            ],
        )
    ]

    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        evaluator.evaluate(
            queries,
            k=0,
        )


def test_retrieval_query_model():
    query = RetrievalQuery(
        query="What is attention?",
        relevant_sources=[
            "attention_is_all_you_need.pdf",
        ],
    )

    assert query.query == "What is attention?"
    assert query.relevant_sources == [
        "attention_is_all_you_need.pdf"
    ]