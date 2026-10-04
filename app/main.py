import json
from pathlib import Path
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from rag_benchmark.benchmarking.benchmark_engine import BenchmarkEngine
from rag_benchmark.benchmarking.experiments import (
    run_chunking_experiment,
    run_top_k_experiment,
)
from rag_benchmark.benchmarking.result_utils import report_to_records
from rag_benchmark.chunking.text_chunker import chunk_documents
from rag_benchmark.evaluation.models import RetrievalQuery
from rag_benchmark.generation.context_builder import build_context
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
QUERIES_FILE = Path("data/evaluation/benchmark_queries.json")


@st.cache_resource
def load_pipeline_data():
    ingestion_result = load_documents(DOCUMENTS_DIR)

    chunking_result = chunk_documents(
        ingestion_result.documents,
    )

    documents = chunking_result.chunks

    dense = DenseRetriever(
        documents=documents,
        index_path="data/indexes/dense_faiss.index",
    )
    bm25 = BM25Retriever(documents=documents)
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)
    reranker = CrossEncoderReranker()
    hybrid_rerank = RerankedRetriever(base_retriever=hybrid, reranker=reranker)

    pipelines = {
        "Dense": dense,
        "BM25": bm25,
        "Hybrid": hybrid,
        "Hybrid + Rerank": hybrid_rerank,
    }

    return {
        "pipelines": pipelines,
        "raw_documents": ingestion_result.documents,
        "chunks": documents,
        "reranker": reranker,
    }


def load_evaluation_queries() -> list[RetrievalQuery]:
    if QUERIES_FILE.exists():
        with open(QUERIES_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        return [
            RetrievalQuery(
                query=item["query"],
                relevant_sources=item["relevant_sources"],
            )
            for item in data
        ]

    return [
        RetrievalQuery(
            query="What is attention?",
            relevant_sources=["attention_is_all_you_need.pdf"],
        ),
        RetrievalQuery(
            query="What is BERT?",
            relevant_sources=["bert.pdf"],
        ),
        RetrievalQuery(
            query="What are sentence embeddings?",
            relevant_sources=["sentence_bert.pdf"],
        ),
        RetrievalQuery(
            query="What is retrieval augmented generation?",
            relevant_sources=["rag.pdf"],
        ),
    ]


def display_documents(documents):
    if not documents:
        st.info("No documents retrieved.")
        return

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page_label", "?")

        with st.expander(f"{index}. {source} — Page {page}"):
            st.write(document.page_content)


def main():
    st.set_page_config(
        page_title="RAG Benchmark Studio",
        page_icon="R",
        layout="wide",
    )

    st.title("RAG Benchmark Studio")
    st.caption("Evaluation and analysis studio for retrieval-augmented generation pipelines.")

    data = load_pipeline_data()
    pipelines = data["pipelines"]

    with st.sidebar:
        st.header("Pipeline Configuration")
        st.markdown(f"**Indexed Documents:** {len(data['raw_documents'])}")
        st.markdown(f"**Total Chunks:** {len(data['chunks'])}")
        st.markdown("**Embedding Model:** `all-MiniLM-L6-v2`")
        st.markdown("**Reranker:** `ms-marco-MiniLM-L-6-v2`")
        st.markdown("**FAISS Index:** `Persistent (Cached)`")
        if LLM_API_KEY:
            st.success("LLM: Configured (Gemini)")
        else:
            st.warning("LLM: Not Configured")

    (
        chat_tab,
        comparison_tab,
        context_tab,
        benchmark_tab,
        experiments_tab,
    ) = st.tabs(
        [
            "Chat",
            "Retrieval Comparison",
            "Retrieved Context",
            "Benchmark",
            "Experiments & Analysis",
        ]
    )

    # --------------------------------------------------
    # TAB 1: CHAT (Single Source of Truth)
    # --------------------------------------------------
    with chat_tab:
        st.subheader("Ask a Question")

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Question",
                value=st.session_state.get("active_query", ""),
                placeholder="e.g. What is the attention mechanism in transformers?",
                key="chat_query_input",
            )
        with col2:
            pipeline_name = st.selectbox(
                "Retrieval pipeline",
                list(pipelines.keys()),
                index=list(pipelines.keys()).index(
                    st.session_state.get("active_pipeline", "Hybrid")
                ),
            )

        top_k = st.slider(
            "Top-K Chunks",
            min_value=1,
            max_value=10,
            value=st.session_state.get("active_top_k", 5),
        )

        if st.button("Generate Answer", type="primary"):
            if not query.strip():
                st.warning("Please enter a question first.")
            else:
                st.session_state["active_query"] = query.strip()
                st.session_state["active_pipeline"] = pipeline_name
                st.session_state["active_top_k"] = top_k

                retriever = pipelines[pipeline_name]

                start_time = time.perf_counter()
                retrieved_docs = retriever.retrieve(query.strip(), top_k=top_k)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                st.session_state["chat_documents"] = retrieved_docs
                st.session_state["chat_latency_ms"] = latency_ms

                if LLM_API_KEY:
                    try:
                        generator = RAGGenerator(retriever=retriever)
                        gen_result = generator.generate(
                            query.strip(),
                            documents=retrieved_docs,
                        )
                        st.session_state["chat_answer"] = gen_result.answer
                        st.session_state["chat_citations"] = gen_result.citations
                        st.session_state["chat_context"] = gen_result.context
                    except Exception as exc:
                        st.error(f"Generation error: {exc}")
                        st.session_state["chat_answer"] = None
                        st.session_state["chat_citations"] = []
                else:
                    st.info("LLM API key not configured. Displaying retrieved context chunks only.")
                    st.session_state["chat_answer"] = None
                    st.session_state["chat_citations"] = []
                    st.session_state["chat_context"] = build_context(retrieved_docs)

        if st.session_state.get("chat_answer"):
            st.markdown("### Answer")
            st.success(st.session_state["chat_answer"])

            if st.session_state.get("chat_citations"):
                st.markdown("### Sources & Citations")
                for citation in st.session_state["chat_citations"]:
                    st.markdown(f"- `{citation}`")

        if st.session_state.get("chat_documents"):
            latency = st.session_state.get("chat_latency_ms", 0.0)
            st.markdown(f"### Retrieved Chunks ({len(st.session_state['chat_documents'])} chunks, {latency:.1f} ms)")
            display_documents(st.session_state["chat_documents"])

    # --------------------------------------------------
    # TAB 2: RETRIEVAL COMPARISON
    # --------------------------------------------------
    with comparison_tab:
        st.subheader("Retrieval Strategy Comparison")

        active_query = st.session_state.get("active_query", "")

        if not active_query:
            st.info("No active query. Please enter and run a question in the Chat tab first.")
        else:
            active_k = st.session_state.get("active_top_k", 5)
            st.markdown(f"**Comparing all 4 pipelines for:** *'{active_query}'* (Top-K = {active_k})")

            comparison_records = []
            cols = st.columns(len(pipelines))

            for (name, retriever), col in zip(pipelines.items(), cols):
                start_time = time.perf_counter()
                results = retriever.retrieve(active_query, top_k=active_k)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                comparison_records.append(
                    {
                        "Pipeline": name,
                        "Latency (ms)": elapsed_ms,
                        "Retrieved Chunks": len(results),
                        "Top Source": results[0].metadata.get("source", "none") if results else "none",
                        "Top Page": results[0].metadata.get("page_label", "?") if results else "?",
                    }
                )

                with col:
                    st.markdown(f"#### {name}")
                    st.metric("Latency", f"{elapsed_ms:.1f} ms")
                    for rank, doc in enumerate(results, start=1):
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page_label", "?")
                        with st.expander(f"Rank {rank}: {source} (p.{page})"):
                            st.caption(doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""))

            st.markdown("---")
            comp_df = pd.DataFrame(comparison_records)
            fig = px.bar(
                comp_df,
                x="Pipeline",
                y="Latency (ms)",
                color="Pipeline",
                title="Single-Query Retrieval Latency Comparison",
                text_auto=".1f",
            )
            st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # TAB 3: RETRIEVED CONTEXT
    # --------------------------------------------------
    with context_tab:
        st.subheader("Grounded Prompt Context")

        active_query = st.session_state.get("active_query", "")

        if not active_query:
            st.info("No active query. Please enter and run a question in the Chat tab first.")
        else:
            pipeline_name = st.session_state.get("active_pipeline", "Hybrid")
            active_k = st.session_state.get("active_top_k", 5)

            st.markdown(
                f"Showing context chunks retrieved by **{pipeline_name}** (Top-{active_k}) for: *'{active_query}'*"
            )

            chunks = st.session_state.get("chat_documents", [])
            if not chunks:
                retriever = pipelines[pipeline_name]
                chunks = retriever.retrieve(active_query, top_k=active_k)

            for index, document in enumerate(chunks, start=1):
                source = document.metadata.get("source", "unknown")
                page = document.metadata.get("page_label", "?")

                st.markdown(f"**Chunk {index} — {source}, Page {page}**")
                st.code(document.page_content, language="text")

            full_context = build_context(chunks)
            with st.expander("View Full Formatted Context String"):
                st.text(full_context)

    # --------------------------------------------------
    # TAB 4: BENCHMARK
    # --------------------------------------------------
    with benchmark_tab:
        st.subheader("Retrieval Benchmark Engine")
        st.caption("Evaluates pipelines across the standard benchmark questions independently of chat.")

        evaluation_queries = load_evaluation_queries()

        with st.expander(f"Benchmark Questions ({len(evaluation_queries)} questions)"):
            for i, eq in enumerate(evaluation_queries, start=1):
                st.markdown(f"{i}. **{eq.query}** *(Target: {', '.join(eq.relevant_sources)})*")

        k_val = st.slider("Benchmark Top-K", min_value=1, max_value=10, value=5, key="bench_k")

        if st.button("Run Benchmark", type="primary") or "benchmark_df" not in st.session_state:
            with st.spinner("Running retrieval benchmark across all pipelines..."):
                engine = BenchmarkEngine(pipelines=pipelines)
                report = engine.run(evaluation_queries, k=k_val)
                records = report_to_records(report)
                st.session_state["benchmark_df"] = pd.DataFrame(records)

        bench_df = st.session_state.get("benchmark_df")

        if bench_df is not None:
            st.dataframe(bench_df, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                quality_cols = [c for c in ["Recall@K", "Hit Rate@K", "MRR"] if c in bench_df.columns]
                melted_quality = bench_df.melt(
                    id_vars=["Pipeline"],
                    value_vars=quality_cols,
                    var_name="Metric",
                    value_name="Score",
                )
                fig_quality = px.bar(
                    melted_quality,
                    x="Pipeline",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title=f"Retrieval Quality Metrics (K={k_val})",
                )
                st.plotly_chart(fig_quality, use_container_width=True)

            with col2:
                if "Latency (ms)" in bench_df.columns:
                    fig_lat = px.bar(
                        bench_df,
                        x="Pipeline",
                        y="Latency (ms)",
                        color="Pipeline",
                        title="Average Query Latency (ms)",
                        text_auto=".1f",
                    )
                    st.plotly_chart(fig_lat, use_container_width=True)

    # --------------------------------------------------
    # TAB 5: EXPERIMENTS & ANALYSIS (Part 12)
    # --------------------------------------------------
    with experiments_tab:
        st.subheader("Experiments & Trade-Off Analysis")
        st.caption("Demonstrating empirical trade-offs across Dense, BM25, Hybrid, and Cross-Encoder Reranking.")

        st.markdown("### 1. Strategy Comparison: Quality vs. Latency")
        bench_df = st.session_state.get("benchmark_df")

        if bench_df is None:
            st.info("Run the Benchmark in the 'Benchmark' tab first, or click below to evaluate strategies.")
            if st.button("Run Baseline Strategies Evaluation"):
                engine = BenchmarkEngine(pipelines=pipelines)
                report = engine.run(load_evaluation_queries(), k=5)
                st.session_state["benchmark_df"] = pd.DataFrame(report_to_records(report))
                bench_df = st.session_state["benchmark_df"]

        if bench_df is not None and "Latency (ms)" in bench_df.columns:
            fig_tradeoff = px.scatter(
                bench_df,
                x="Latency (ms)",
                y="MRR",
                color="Pipeline",
                size=[18] * len(bench_df),
                text="Pipeline",
                title="Trade-off: Retrieval Quality (MRR) vs. Latency (ms)",
                hover_data=["Recall@K", "Hit Rate@K"],
            )
            fig_tradeoff.update_traces(textposition="top center")
            st.plotly_chart(fig_tradeoff, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.info("**BM25**: Lowest latency; relies on exact keyword tokens. Misses semantic synonyms.")
            c2.info("**Dense**: Semantic vector search. Overcomes vocabulary mismatch but higher latency than BM25.")
            c3.info("**Hybrid**: Fuses keyword and semantic signals. Provides robust multi-modal retrieval.")
            c4.info("**Hybrid + Rerank**: Cross-attention reranking yields sharpest precision at highest compute cost.")

        st.markdown("---")
        st.markdown("### 2. Top-K Scaling Experiment")
        st.caption("Analyzes how expanding candidate pool size K affects Recall and Latency.")

        if st.button("Run Top-K Sensitivity (K = 1, 3, 5, 10)"):
            with st.spinner("Evaluating K sensitivity across pipelines..."):
                top_k_df = run_top_k_experiment(
                    pipelines=pipelines,
                    evaluation_queries=load_evaluation_queries(),
                    k_values=[1, 3, 5, 10],
                )
                st.session_state["top_k_df"] = top_k_df

        top_k_df = st.session_state.get("top_k_df")
        if top_k_df is not None:
            st.dataframe(top_k_df, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig_top_k = px.line(
                    top_k_df,
                    x="K",
                    y="Recall@K",
                    color="Pipeline",
                    markers=True,
                    title="Recall@K as a function of K",
                )
                st.plotly_chart(fig_top_k, use_container_width=True)
            with col2:
                fig_top_k_lat = px.line(
                    top_k_df,
                    x="K",
                    y="Latency (ms)",
                    color="Pipeline",
                    markers=True,
                    title="Latency (ms) as a function of K",
                )
                st.plotly_chart(fig_top_k_lat, use_container_width=True)

            st.caption(
                "Takeaway: Higher K increases recall coverage but increases prompt length, LLM inference latency, and risk of distraction."
            )

        st.markdown("---")
        st.markdown("### 3. Chunk Size Impact Analysis")
        st.caption("Compares retrieval performance under different passage chunking configurations.")

        if st.button("Run Chunk Size Comparison (250 vs 500 vs 1000)"):
            with st.spinner("Evaluating chunk size configurations..."):
                chunk_df = run_chunking_experiment(
                    raw_documents=data["raw_documents"],
                    evaluation_queries=load_evaluation_queries(),
                    chunk_configs=[
                        {"chunk_size": 250, "chunk_overlap": 50},
                        {"chunk_size": 500, "chunk_overlap": 100},
                        {"chunk_size": 1000, "chunk_overlap": 200},
                    ],
                    k=5,
                )
                st.session_state["chunk_df"] = chunk_df

        chunk_df = st.session_state.get("chunk_df")
        if chunk_df is not None:
            st.dataframe(chunk_df, use_container_width=True)

            fig_chunk = px.bar(
                chunk_df,
                x="Chunk Size",
                y="Recall@K",
                color="Chunk Size",
                title="Recall@5 across Chunk Sizes (Hybrid Retrieval)",
                text_auto=".2f",
            )
            st.plotly_chart(fig_chunk, use_container_width=True)

            st.caption(
                "Takeaway: Smaller chunks offer higher granularity and less noisy context; larger chunks preserve sentence context at the expense of retrieval specificity."
            )


if __name__ == "__main__":
    main()