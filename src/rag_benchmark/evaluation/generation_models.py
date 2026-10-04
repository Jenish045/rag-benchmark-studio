from dataclasses import dataclass


@dataclass
class GenerationEvaluationSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


@dataclass
class GenerationMetrics:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float