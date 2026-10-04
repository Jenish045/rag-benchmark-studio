import numpy as np
import pandas as pd
import pytest
from langchain_core.documents import Document

from rag_benchmark.benchmarking.experiments import (
    run_chunking_experiment,
    run_top_k_experiment,
)
from rag_benchmark.evaluation.models import RetrievalQuery


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=5):
        return self.documents[:top_k]


class FakeEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True):
        return np.ones((len(texts), 3), dtype=np.float32)


def test_run_top_k_experiment():
    documents = [
        Document(
            page_content="Transformers attention.",
            metadata={"source": "attention.pdf"},
        ),
        Document(
            page_content="Convolutional networks.",
            metadata={"source": "vision.pdf"},
        ),
    ]

    pipelines = {
        "TestPipeline": FakeRetriever(documents),
    }

    queries = [
        RetrievalQuery(
            query="attention",
            relevant_sources=["attention.pdf"],
        )
    ]

    df = run_top_k_experiment(
        pipelines=pipelines,
        evaluation_queries=queries,
        k_values=[1, 2],
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "K" in df.columns
    assert "Recall@K" in df.columns
    assert "Latency (ms)" in df.columns
    assert list(df["K"]) == [1, 2]


def test_run_chunking_experiment():
    raw_documents = [
        Document(
            page_content="Sentence one about transformers. Sentence two about attention. " * 10,
            metadata={"source": "test.pdf", "page": 1},
        )
    ]

    queries = [
        RetrievalQuery(
            query="transformers",
            relevant_sources=["test.pdf"],
        )
    ]

    configs = [
        {"chunk_size": 100, "chunk_overlap": 20},
        {"chunk_size": 200, "chunk_overlap": 40},
    ]

    df = run_chunking_experiment(
        raw_documents=raw_documents,
        evaluation_queries=queries,
        chunk_configs=configs,
        k=2,
        embedding_model=FakeEmbeddingModel(),
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Chunk Size" in df.columns
    assert "Total Chunks" in df.columns
    assert "Recall@K" in df.columns
    assert df["Chunk Size"].iloc[0] == 100
    assert df["Chunk Size"].iloc[1] == 200
