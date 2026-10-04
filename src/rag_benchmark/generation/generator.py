from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from rag_benchmark.generation.context_builder import build_context
from rag_benchmark.generation.models import GenerationResult
from rag_benchmark.generation.prompt_builder import build_prompt
from rag_benchmark.utils.config import LLM_API_KEY, LLM_MODEL


def get_default_llm() -> Any:
    if not LLM_API_KEY:
        raise ValueError(
            "LLM API key not found. Please set LLM_API_KEY in your environment or .env file."
        )

    is_gemini = (
        LLM_API_KEY.startswith("AQ")
        or LLM_API_KEY.startswith("AIza")
        or (LLM_MODEL and "gemini" in LLM_MODEL.lower())
        or LLM_MODEL == "Gemini API Key"
    )

    if is_gemini:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = (
            LLM_MODEL
            if LLM_MODEL and LLM_MODEL != "Gemini API Key"
            else "gemini-3.8-flash"
        )
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=LLM_API_KEY,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL or "gpt-4o-mini",
        api_key=LLM_API_KEY,
    )


@dataclass
class RAGGenerator:
    llm: Any | None = None
    retriever: Any | None = None
    reranker: Any | None = None

    def generate(
        self,
        query: str,
        documents: list[Document] | None = None,
        top_k: int | None = None,
    ) -> GenerationResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if documents is None:
            if self.retriever is None:
                raise ValueError(
                    "No retrieved documents provided for generation."
                )

            documents = self.retriever.retrieve(
                query,
                top_k=top_k,
            )

            if self.reranker is not None:
                documents = self.reranker.rerank(
                    query,
                    documents,
                    top_k=top_k or len(documents),
                )

        if not documents:
            raise ValueError(
                "No retrieved documents provided for generation."
            )

        if self.llm is None:
            self.llm = get_default_llm()

        context = build_context(documents)
        prompt = build_prompt(query, context)

        response = self.llm.invoke(prompt)

        answer = self._extract_answer(response)

        citations = [
            self._format_citation(document)
            for document in documents
        ]

        return GenerationResult(
            answer=answer,
            citations=citations,
            context=context,
        )

    @staticmethod
    def _extract_answer(response: Any) -> str:
        if isinstance(response, str):
            return response.strip()

        content = getattr(response, "content", None)

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                elif hasattr(part, "text"):
                    parts.append(part.text)
            if parts:
                return "\n".join(parts).strip()

        raise ValueError(
            "LLM response must be a string or contain string content."
        )

    @staticmethod
    def _format_citation(document: Document) -> str:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page_label")

        if page is not None:
            return f"{source}, page {page}"

        return source