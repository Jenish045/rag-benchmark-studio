import pytest
from langchain_core.documents import Document

from rag_benchmark.benchmarking.benchmark_engine import (
    BenchmarkEngine,
)
from rag_benchmark.benchmarking.models import (
    BenchmarkReport,
    BenchmarkResult,
)
from rag_benchmark.benchmarking.result_utils import (
    report_to_records,
)
from rag_benchmark.evaluation.models import RetrievalQuery


def document(source: str) -> Document:
    return Document(
        page_content=f"Content from {source}",
        metadata={"source": source},
    )


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=5):
        return self.documents[:top_k]


def evaluation_queries():
    return [
        RetrievalQuery(
            query="attention",
            relevant_sources=["attention.pdf"],
        ),
        RetrievalQuery(
            query="embeddings",
            relevant_sources=["embeddings.pdf"],
        ),
    ]


def test_benchmark_engine_runs_multiple_pipelines():
    dense = FakeRetriever(
        [
            document("attention.pdf"),
            document("wrong.pdf"),
        ]
    )

    bm25 = FakeRetriever(
        [
            document("embeddings.pdf"),
            document("wrong.pdf"),
        ]
    )

    engine = BenchmarkEngine(
        pipelines={
            "Dense": dense,
            "BM25": bm25,
        }
    )

    report = engine.run(
        evaluation_queries(),
        k=2,
    )

    assert isinstance(report, BenchmarkReport)
    assert len(report.results) == 2


def test_benchmark_engine_preserves_pipeline_names():
    retriever = FakeRetriever(
        [
            document("attention.pdf"),
        ]
    )

    engine = BenchmarkEngine(
        pipelines={
            "Dense Retrieval": retriever,
            "BM25 Retrieval": retriever,
            "Hybrid Retrieval": retriever,
        }
    )

    report = engine.run(
        evaluation_queries()[:1],
        k=1,
    )

    names = [
        result.pipeline
        for result in report.results
    ]

    assert names == [
        "Dense Retrieval",
        "BM25 Retrieval",
        "Hybrid Retrieval",
    ]


def test_benchmark_engine_calculates_metrics():
    retriever = FakeRetriever(
        [
            document("attention.pdf"),
            document("wrong.pdf"),
        ]
    )

    engine = BenchmarkEngine(
        pipelines={"Test": retriever},
    )

    report = engine.run(
        [
            RetrievalQuery(
                query="attention",
                relevant_sources=["attention.pdf"],
            )
        ],
        k=2,
    )

    result = report.results[0]

    assert result.recall_at_k == 1.0
    assert result.hit_rate_at_k == 1.0
    assert result.mrr == 1.0


def test_benchmark_engine_handles_poor_retrieval():
    retriever = FakeRetriever(
        [
            document("wrong.pdf"),
            document("attention.pdf"),
        ]
    )

    engine = BenchmarkEngine(
        pipelines={"Test": retriever},
    )

    report = engine.run(
        [
            RetrievalQuery(
                query="attention",
                relevant_sources=["attention.pdf"],
            )
        ],
        k=2,
    )

    result = report.results[0]

    assert result.recall_at_k == 1.0
    assert result.hit_rate_at_k == 1.0
    assert result.mrr == 0.5


def test_benchmark_engine_rejects_empty_pipelines():
    with pytest.raises(
        ValueError,
        match="pipelines cannot be empty",
    ):
        BenchmarkEngine(
            pipelines={},
        )


def test_benchmark_engine_rejects_empty_queries():
    retriever = FakeRetriever(
        [document("test.pdf")]
    )

    engine = BenchmarkEngine(
        pipelines={"Test": retriever},
    )

    with pytest.raises(
        ValueError,
        match="evaluation_queries cannot be empty",
    ):
        engine.run([])


def test_benchmark_engine_rejects_invalid_k():
    retriever = FakeRetriever(
        [document("test.pdf")]
    )

    engine = BenchmarkEngine(
        pipelines={"Test": retriever},
    )

    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        engine.run(
            evaluation_queries(),
            k=0,
        )


def test_report_to_records():
    report = BenchmarkReport(
        results=[
            BenchmarkResult(
                pipeline="Dense",
                recall_at_k=0.8,
                hit_rate_at_k=0.9,
                mrr=0.7,
                latency_ms=12.34,
            ),
            BenchmarkResult(
                pipeline="BM25",
                recall_at_k=0.7,
                hit_rate_at_k=0.8,
                mrr=0.6,
                latency_ms=5.67,
            ),
        ]
    )

    records = report_to_records(report)

    assert records == [
        {
            "Pipeline": "Dense",
            "Recall@K": 0.8,
            "Hit Rate@K": 0.9,
            "MRR": 0.7,
            "Latency (ms)": 12.34,
        },
        {
            "Pipeline": "BM25",
            "Recall@K": 0.7,
            "Hit Rate@K": 0.8,
            "MRR": 0.6,
            "Latency (ms)": 5.67,
        },
    ]


def test_benchmark_report_preserves_result_order():
    report = BenchmarkReport(
        results=[
            BenchmarkResult(
                pipeline="First",
                recall_at_k=0.5,
                hit_rate_at_k=0.5,
                mrr=0.5,
            ),
            BenchmarkResult(
                pipeline="Second",
                recall_at_k=0.8,
                hit_rate_at_k=0.8,
                mrr=0.8,
            ),
        ]
    )

    records = report_to_records(report)

    assert records[0]["Pipeline"] == "First"
    assert records[1]["Pipeline"] == "Second"


def test_benchmark_engine_measures_latency():
    retriever = FakeRetriever(
        [
            document("attention.pdf"),
        ]
    )

    engine = BenchmarkEngine(
        pipelines={"Test": retriever},
    )

    report = engine.run(
        evaluation_queries()[:1],
        k=1,
    )

    assert report.results[0].latency_ms >= 0.0