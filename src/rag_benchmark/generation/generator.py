from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from rag_benchmark.generation.context_builder import build_context
from rag_benchmark.generation.models import GenerationResult
from rag_benchmark.generation.prompt_builder import build_prompt


@dataclass
class RAGGenerator:
    llm: Any

    def generate(
        self,
        query: str,
        documents: list[Document],
    ) -> GenerationResult:
        if not documents:
            raise ValueError(
                "No retrieved documents provided for generation."
            )

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