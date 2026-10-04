import json
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_benchmark.benchmarking.benchmark_engine import BenchmarkEngine
from rag_benchmark.benchmarking.result_utils import report_to_records
from rag_benchmark.chunking.text_chunker import chunk_documents
from rag_benchmark.evaluation.models import RetrievalQuery
from rag_benchmark.generation.generator import RAGGenerator
from rag_benchmark.ingestion.document_loader import load_documents
from rag_benchmark.reranking.cross_encoder_reranker import (
    CrossEncoderReranker,
    RerankedRetriever,
)
from rag_benchmark.retrieval.bm25_retriever import BM25Retriever
from rag_benchmark.retrieval.dense_retriever import DenseRetriever
from rag_benchmark.retrieval.hybrid_retriever import HybridRetriever
from rag_benchmark.utils.config import LLM_API_KEY


DOCUMENTS_DIR = Path("data/documents")

@st.cache_resource
def load_pipeline():
    ingestion_result = load_documents(DOCUMENTS_DIR)

    chunking_result = chunk_documents(
        ingestion_result.documents,
    )

    documents = chunking_result.chunks

    dense = DenseRetriever(
        documents=documents,
    )

    bm25 = BM25Retriever(
        documents=documents,
    )

    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    reranker = CrossEncoderReranker()

    hybrid_rerank = RerankedRetriever(
        base_retriever=hybrid,
        reranker=reranker,
    )

    return {
        "Dense": dense,
        "BM25": bm25,
        "Hybrid": hybrid,
        "Hybrid + Rerank": hybrid_rerank,
    }


def display_documents(documents):
    if not documents:
        st.info("No documents retrieved.")
        return

    for index, document in enumerate(
        documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get(
            "page_label",
            "?",
        )

        with st.expander(
            f"{index}. {source} — Page {page}"
        ):
            st.write(document.page_content)


def main():
    st.set_page_config(
        page_title="RAG Benchmark Studio",
        page_icon="R",
        layout="wide",
    )

    st.title("RAG Benchmark Studio")

    st.caption(
        "Experiment-driven evaluation of retrieval pipelines."
    )

    pipelines = load_pipeline()

    chat_tab, comparison_tab, benchmark_tab, context_tab = (
        st.tabs(
            [
                "Chat",
                "Retrieval Comparison",
                "Benchmark",
                "Retrieved Context",
            ]
        )
    )

    with chat_tab:
        st.subheader("Ask a question")

        query = st.text_input(
            "Question",
            placeholder="What is the attention mechanism?",
        )

        pipeline_name = st.selectbox(
            "Retrieval pipeline",
            list(pipelines.keys()),
        )

        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=10,
            value=5,
        )

        if st.button("Ask"):
            if not query.strip():
                st.warning("Enter a question first.")
            else:
                retriever = pipelines[pipeline_name]

                if LLM_API_KEY:
                    try:
                        generator = RAGGenerator(retriever=retriever)
                        result = generator.generate(query, top_k=top_k)

                        st.subheader("Answer")
                        st.write(result.answer)

                        if result.citations:
                            st.subheader("Sources")
                            for citation in result.citations:
                                st.write(f"- {citation}")
                    except Exception as exc:
                        st.error(f"Generation error: {exc}")
                else:
                    st.info(
                        "LLM API key not configured. Displaying retrieved documents only."
                    )

                results = retriever.retrieve(
                    query,
                    top_k=top_k,
                )

                st.subheader("Retrieved Documents")
                display_documents(results)

    with comparison_tab:
        st.subheader("Retrieval Comparison")

        query = st.text_input(
            "Comparison query",
            placeholder="What are embeddings?",
            key="comparison_query",
        )

        top_k = st.slider(
            "Comparison Top-K",
            min_value=1,
            max_value=10,
            value=5,
            key="comparison_k",
        )

        if st.button("Compare Retrieval"):
            if not query.strip():
                st.warning("Enter a question first.")
            else:
                for name, retriever in pipelines.items():
                    st.markdown(f"### {name}")

                    results = retriever.retrieve(
                        query,
                        top_k=top_k,
                    )

                    display_documents(results)

    with benchmark_tab:
        st.subheader("Benchmark")

        st.write(
            "Run the configured retrieval pipelines "
            "against the evaluation questions."
        )

        queries_file = Path("data/evaluation/benchmark_queries.json")
        if queries_file.exists():
            with open(queries_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            evaluation_queries = [
                RetrievalQuery(
                    query=item["query"],
                    relevant_sources=item["relevant_sources"],
                )
                for item in data
            ]
        else:
            evaluation_queries = [
                RetrievalQuery(
                    query="What is attention?",
                    relevant_sources=[
                        "attention_is_all_you_need.pdf"
                    ],
                ),
                RetrievalQuery(
                    query="What is BERT?",
                    relevant_sources=[
                        "bert.pdf"
                    ],
                ),
                RetrievalQuery(
                    query="What are sentence embeddings?",
                    relevant_sources=[
                        "sentence_bert.pdf"
                    ],
                ),
                RetrievalQuery(
                    query="What is retrieval augmented generation?",
                    relevant_sources=[
                        "rag.pdf"
                    ],
                ),
            ]

        if st.button("Run Benchmark"):
            engine = BenchmarkEngine(
                pipelines=pipelines,
            )

            report = engine.run(
                evaluation_queries,
                k=5,
            )

            records = report_to_records(report)

            dataframe = pd.DataFrame(records)

            st.dataframe(
                dataframe,
                use_container_width=True,
            )

            chart_columns = [
                col
                for col in ["Recall@K", "Hit Rate@K", "MRR"]
                if col in dataframe.columns
            ]

            chart_data = dataframe.set_index("Pipeline")[chart_columns]

            st.bar_chart(chart_data)

    with context_tab:
        st.subheader("Inspect Retrieved Context")

        query = st.text_input(
            "Context query",
            placeholder="What is retrieval augmented generation?",
            key="context_query",
        )

        pipeline_name = st.selectbox(
            "Context pipeline",
            list(pipelines.keys()),
            key="context_pipeline",
        )

        top_k = st.slider(
            "Context Top-K",
            min_value=1,
            max_value=10,
            value=5,
            key="context_k",
        )

        if st.button("Inspect Context"):
            if not query.strip():
                st.warning("Enter a question first.")
            else:
                results = pipelines[pipeline_name].retrieve(
                    query,
                    top_k=top_k,
                )

                for index, document in enumerate(
                    results,
                    start=1,
                ):
                    source = document.metadata.get(
                        "source",
                        "unknown",
                    )

                    page = document.metadata.get(
                        "page_label",
                        "?",
                    )

                    st.markdown(
                        f"**Chunk {index} — "
                        f"{source}, page {page}**"
                    )

                    st.code(
                        document.page_content,
                        language="text",
                    )


if __name__ == "__main__":
    main()