from dataclasses import dataclass

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


@dataclass
class GenerationEvaluator:

    def evaluate(
        self,
        samples: list[GenerationEvaluationSample],
    ) -> GenerationMetrics:
        if not samples:
            raise ValueError("samples cannot be empty.")

        faithfulness_scores = []
        relevance_scores = []
        precision_scores = []
        recall_scores = []

        for sample in samples:
            faithfulness_scores.append(
                faithfulness(
                    sample.answer,
                    sample.contexts,
                )
            )

            relevance_scores.append(
                answer_relevance(
                    sample.question,
                    sample.answer,
                )
            )

            precision_scores.append(
                context_relevance(
                    sample.answer,
                    sample.contexts,
                )
            )

            recall_scores.append(
                context_recall(
                    sample.ground_truth,
                    sample.contexts,
                )
            )

        count = len(samples)

        return GenerationMetrics(
            faithfulness=sum(faithfulness_scores) / count,
            answer_relevance=sum(relevance_scores) / count,
            context_precision=sum(precision_scores) / count,
            context_recall=sum(recall_scores) / count,
        )