from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    pipeline: str
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    latency_ms: float = 0.0


@dataclass
class BenchmarkReport:
    results: list[BenchmarkResult]