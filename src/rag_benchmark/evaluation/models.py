from dataclasses import dataclass


@dataclass
class RetrievalQuery:
    query: str
    relevant_sources: list[str]


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    latency_ms: float = 0.0