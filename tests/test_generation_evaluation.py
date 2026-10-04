import pytest

from rag_benchmark.evaluation.generation_evaluator import (
    GenerationEvaluator,
)
from rag_benchmark.evaluation.generation_metrics import (
    answer_relevance,
    context_recall,
    context_relevance,
    faithfulness,
)
from rag_benchmark.evaluation.generation_models import (
    GenerationEvaluationSample,
    GenerationMetrics,
)


def test_answer_relevance():
    score = answer_relevance(
        "What is attention?",
        "Attention allows models to focus on relevant information.",
    )

    assert score > 0


def test_answer_relevance_zero_overlap():
    score = answer_relevance(
        "What is attention?",
        "Images contain visual features.",
    )

    assert score == 0.0


def test_context_relevance():
    score = context_relevance(
        "Attention allows models to focus on relevant information.",
        [
            "Attention allows models to focus on relevant information."
        ],
    )

    assert score == 1.0


def test_context_recall():
    score = context_recall(
        "Attention models relationships between tokens.",
        [
            "Attention models relationships between tokens."
        ],
    )

    assert score == 1.0


def test_context_recall_partial():
    score = context_recall(
        "Attention models relationships between tokens.",
        [
            "Attention models relationships between inputs."
        ],
    )

    assert 0.0 < score < 1.0


def test_faithfulness():
    score = faithfulness(
        "Attention models relationships between tokens.",
        [
            "Attention models relationships between tokens."
        ],
    )

    assert score == 1.0


def test_faithfulness_detects_unsupported_content():
    score = faithfulness(
        "Attention models relationships between tokens and images.",
        [
            "Attention models relationships between tokens."
        ],
    )

    assert score < 1.0


def test_generation_evaluator():
    samples = [
        GenerationEvaluationSample(
            question="What is attention?",
            answer="Attention models relationships between tokens.",
            contexts=[
                "Attention models relationships between tokens."
            ],
            ground_truth="Attention models relationships between tokens.",
        )
    ]

    evaluator = GenerationEvaluator()

    metrics = evaluator.evaluate(samples)

    assert isinstance(metrics, GenerationMetrics)
    assert metrics.faithfulness == 1.0
    assert metrics.answer_relevance > 0
    assert metrics.context_precision == 1.0
    assert metrics.context_recall == 1.0


def test_generation_evaluator_multiple_samples():
    samples = [
        GenerationEvaluationSample(
            question="What is attention?",
            answer="Attention models relationships.",
            contexts=[
                "Attention models relationships."
            ],
            ground_truth="Attention models relationships.",
        ),
        GenerationEvaluationSample(
            question="What are embeddings?",
            answer="Embeddings represent text as vectors.",
            contexts=[
                "Embeddings represent text as vectors."
            ],
            ground_truth="Embeddings represent text as vectors.",
        ),
    ]

    evaluator = GenerationEvaluator()

    metrics = evaluator.evaluate(samples)

    assert metrics.faithfulness == 1.0
    assert metrics.context_precision == 1.0
    assert metrics.context_recall == 1.0


def test_generation_evaluator_rejects_empty_samples():
    evaluator = GenerationEvaluator()

    with pytest.raises(
        ValueError,
        match="samples cannot be empty",
    ):
        evaluator.evaluate([])


def test_metrics_reject_empty_context():
    with pytest.raises(
        ValueError,
        match="contexts cannot be empty",
    ):
        faithfulness(
            "Some answer.",
            [],
        )


def test_metrics_reject_empty_question():
    with pytest.raises(
        ValueError,
        match="question must be a non-empty string",
    ):
        answer_relevance(
            "",
            "Some answer.",
        )


def test_metrics_reject_empty_ground_truth():
    with pytest.raises(
        ValueError,
        match="ground_truth must be a non-empty string",
    ):
        context_recall(
            "",
            ["Some context."],
        )