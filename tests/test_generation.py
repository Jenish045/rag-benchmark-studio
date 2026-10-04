import pytest
from langchain_core.documents import Document

from rag_benchmark.generation.context_builder import build_context
from rag_benchmark.generation.generator import RAGGenerator
from rag_benchmark.generation.models import GenerationResult
from rag_benchmark.generation.prompt_builder import build_prompt


class FakeLLM:
    def __init__(self):
        self.received_prompt = None

    def invoke(self, prompt):
        self.received_prompt = prompt
        return "Attention allows models to focus on relevant information."


def create_documents():
    return [
        Document(
            page_content="Attention allows models to focus on relevant information.",
            metadata={
                "source": "attention.pdf",
                "page": 2,
                "page_label": "3",
            },
        ),
        Document(
            page_content="Transformers use attention mechanisms.",
            metadata={
                "source": "transformers.pdf",
                "page": 4,
                "page_label": "5",
            },
        ),
    ]


def test_build_context():
    documents = create_documents()

    context = build_context(documents)

    assert "attention.pdf" in context
    assert "page 3" in context
    assert "transformers.pdf" in context
    assert "page 5" in context
    assert "Attention allows models" in context


def test_build_context_rejects_empty_documents():
    with pytest.raises(
        ValueError,
        match="No documents",
    ):
        build_context([])


def test_build_prompt_contains_query_and_context():
    context = "Attention is used in transformers."

    prompt = build_prompt(
        "What is attention?",
        context,
    )

    assert "What is attention?" in prompt
    assert context in prompt
    assert "using only the provided context" in prompt


def test_build_prompt_rejects_empty_query():
    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        build_prompt(
            "",
            "Some context.",
        )


def test_build_prompt_rejects_empty_context():
    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        build_prompt(
            "What is attention?",
            "",
        )


def test_rag_generator_returns_generation_result():
    llm = FakeLLM()
    generator = RAGGenerator(llm=llm)

    result = generator.generate(
        "What is attention?",
        create_documents(),
    )

    assert isinstance(result, GenerationResult)
    assert result.answer
    assert result.context
    assert result.citations


def test_rag_generator_calls_llm_with_prompt():
    llm = FakeLLM()
    generator = RAGGenerator(llm=llm)

    generator.generate(
        "What is attention?",
        create_documents(),
    )

    assert llm.received_prompt is not None
    assert "What is attention?" in llm.received_prompt
    assert "attention.pdf" in llm.received_prompt


def test_rag_generator_preserves_citations():
    llm = FakeLLM()
    generator = RAGGenerator(llm=llm)

    result = generator.generate(
        "What is attention?",
        create_documents(),
    )

    assert result.citations == [
        "attention.pdf, page 3",
        "transformers.pdf, page 5",
    ]


def test_rag_generator_rejects_empty_documents():
    llm = FakeLLM()
    generator = RAGGenerator(llm=llm)

    with pytest.raises(
        ValueError,
        match="No retrieved documents",
    ):
        generator.generate(
            "What is attention?",
            [],
        )


def test_rag_generator_accepts_string_llm_response():
    llm = FakeLLM()
    generator = RAGGenerator(llm=llm)

    result = generator.generate(
        "What is attention?",
        create_documents(),
    )

    assert result.answer == (
        "Attention allows models to focus on relevant information."
    )


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=5):
        return self.documents[:top_k]


class FakeReranker:
    def rerank(self, query, documents, top_k=5):
        # Reverse documents to verify reranker was executed
        return list(reversed(documents))[:top_k]


def test_rag_generator_end_to_end_with_retriever():
    llm = FakeLLM()
    retriever = FakeRetriever(create_documents())
    generator = RAGGenerator(llm=llm, retriever=retriever)

    result = generator.generate("What is attention?", top_k=2)

    assert isinstance(result, GenerationResult)
    assert result.answer == "Attention allows models to focus on relevant information."
    assert "attention.pdf" in result.context
    assert len(result.citations) == 2


def test_rag_generator_end_to_end_with_reranker():
    llm = FakeLLM()
    retriever = FakeRetriever(create_documents())
    reranker = FakeReranker()
    generator = RAGGenerator(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
    )

    result = generator.generate("What is attention?", top_k=2)

    assert isinstance(result, GenerationResult)
    # The first citation should be transformers.pdf because reranker reversed the order
    assert result.citations[0] == "transformers.pdf, page 5"


def test_rag_generator_rejects_empty_query():
    llm = FakeLLM()
    generator = RAGGenerator(llm=llm)

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        generator.generate("", create_documents())


def test_rag_generator_rejects_no_retriever_and_no_documents():
    llm = FakeLLM()
    generator = RAGGenerator(llm=llm)

    with pytest.raises(ValueError, match="No retrieved documents"):
        generator.generate("What is attention?")


def test_get_default_llm_fails_without_api_key(monkeypatch):
    from rag_benchmark.generation.generator import get_default_llm
    import rag_benchmark.generation.generator as gen_module

    monkeypatch.setattr(gen_module, "LLM_API_KEY", None)

    with pytest.raises(ValueError, match="LLM API key not found"):
        get_default_llm()