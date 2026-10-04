from dataclasses import dataclass
from typing import Any

from rag_benchmark.benchmarking.models import (
    BenchmarkReport,
    BenchmarkResult,
)
from rag_benchmark.evaluation.models import RetrievalQuery
from rag_benchmark.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
)


@dataclass
class BenchmarkEngine:
    pipelines: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.pipelines:
            raise ValueError("pipelines cannot be empty.")

    def run(
        self,
        evaluation_queries: list[RetrievalQuery],
        k: int = 5,
    ) -> BenchmarkReport:
        if not evaluation_queries:
            raise ValueError(
                "evaluation_queries cannot be empty."
            )

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        results = []

        for name, retriever in self.pipelines.items():
            evaluator = RetrievalEvaluator(
                retriever=retriever,
            )

            metrics = evaluator.evaluate(
                evaluation_queries,
                k=k,
            )

            results.append(
                BenchmarkResult(
                    pipeline=name,
                    recall_at_k=metrics.recall_at_k,
                    hit_rate_at_k=metrics.hit_rate_at_k,
                    mrr=metrics.mrr,
                    latency_ms=metrics.latency_ms,
                )
            )

        return BenchmarkReport(
            results=results,
        )