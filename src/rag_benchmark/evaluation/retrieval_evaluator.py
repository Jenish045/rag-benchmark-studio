from dataclasses import dataclass
import time
from typing import Any

from langchain_core.documents import Document

from rag_benchmark.evaluation.models import (
    RetrievalMetrics,
    RetrievalQuery,
)
from rag_benchmark.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    recall_at_k,
)


@dataclass
class RetrievalEvaluator:
    retriever: Any

    def evaluate(
        self,
        evaluation_queries: list[RetrievalQuery],
        k: int = 5,
    ) -> RetrievalMetrics:
        if not evaluation_queries:
            raise ValueError(
                "evaluation_queries cannot be empty."
            )

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        rankings: list[list[Document]] = []
        relevant_sources_list: list[list[str]] = []

        recall_scores = []
        hit_scores = []
        latencies_ms = []

        for evaluation_query in evaluation_queries:
            start_time = time.perf_counter()
            documents = self.retriever.retrieve(
                evaluation_query.query,
                top_k=k,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            latencies_ms.append(elapsed_ms)

            rankings.append(documents)
            relevant_sources_list.append(
                evaluation_query.relevant_sources
            )

            recall_scores.append(
                recall_at_k(
                    documents,
                    evaluation_query.relevant_sources,
                    k,
                )
            )

            hit_scores.append(
                hit_rate_at_k(
                    documents,
                    evaluation_query.relevant_sources,
                    k,
                )
            )

        return RetrievalMetrics(
            recall_at_k=sum(recall_scores) / len(recall_scores),
            hit_rate_at_k=sum(hit_scores) / len(hit_scores),
            mrr=mean_reciprocal_rank(
                rankings,
                relevant_sources_list,
            ),
            latency_ms=sum(latencies_ms) / len(latencies_ms),
        )