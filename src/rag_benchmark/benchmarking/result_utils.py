from rag_benchmark.benchmarking.models import BenchmarkReport


def report_to_records(
    report: BenchmarkReport,
) -> list[dict[str, float | str]]:
    return [
        {
            "Pipeline": result.pipeline,
            "Recall@K": result.recall_at_k,
            "Hit Rate@K": result.hit_rate_at_k,
            "MRR": result.mrr,
            "Latency (ms)": round(result.latency_ms, 2),
        }
        for result in report.results
    ]