from typing import Any
import pandas as pd

from rag_benchmark.benchmarking.benchmark_engine import BenchmarkEngine
from rag_benchmark.benchmarking.result_utils import report_to_records
from rag_benchmark.chunking.text_chunker import chunk_documents
from rag_benchmark.evaluation.models import RetrievalQuery
from rag_benchmark.retrieval.bm25_retriever import BM25Retriever
from rag_benchmark.retrieval.dense_retriever import DenseRetriever
from rag_benchmark.retrieval.hybrid_retriever import HybridRetriever


def run_top_k_experiment(
    pipelines: dict[str, Any],
    evaluation_queries: list[RetrievalQuery],
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """
    Evaluate pipelines across different top-k values to measure recall scaling.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    rows = []
    engine = BenchmarkEngine(pipelines=pipelines)

    for k in k_values:
        report = engine.run(evaluation_queries, k=k)
        records = report_to_records(report)
        for record in records:
            record["K"] = k
            rows.append(record)

    return pd.DataFrame(rows)


def run_chunking_experiment(
    raw_documents: list[Any],
    evaluation_queries: list[RetrievalQuery],
    chunk_configs: list[dict[str, int]] | None = None,
    k: int = 5,
    embedding_model: Any | None = None,
) -> pd.DataFrame:
    """
    Evaluate retrieval quality across different chunk configurations.
    """
    if chunk_configs is None:
        chunk_configs = [
            {"chunk_size": 250, "chunk_overlap": 50},
            {"chunk_size": 500, "chunk_overlap": 100},
            {"chunk_size": 1000, "chunk_overlap": 200},
        ]

    rows = []

    for config in chunk_configs:
        size = config["chunk_size"]
        overlap = config["chunk_overlap"]

        chunk_result = chunk_documents(
            raw_documents,
            chunk_size=size,
            chunk_overlap=overlap,
        )

        dense = DenseRetriever(
            documents=chunk_result.chunks,
            embedding_model=embedding_model,
        )
        bm25 = BM25Retriever(
            documents=chunk_result.chunks,
        )
        hybrid = HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
        )

        engine = BenchmarkEngine(pipelines={"Hybrid": hybrid})
        report = engine.run(evaluation_queries, k=k)
        records = report_to_records(report)

        for record in records:
            record["Chunk Size"] = size
            record["Chunk Overlap"] = overlap
            record["Total Chunks"] = chunk_result.total_chunks
            rows.append(record)

    return pd.DataFrame(rows)
