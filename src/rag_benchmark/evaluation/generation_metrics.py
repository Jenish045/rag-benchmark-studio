import re


def _tokens(text: str) -> set[str]:
    return set(
        re.findall(
            r"\b\w+\b",
            text.lower(),
        )
    )


def _validate_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def answer_relevance(
    question: str,
    answer: str,
) -> float:
    _validate_text(question, "question")
    _validate_text(answer, "answer")

    question_tokens = _tokens(question)
    answer_tokens = _tokens(answer)

    if not question_tokens:
        return 0.0

    return len(
        question_tokens.intersection(answer_tokens)
    ) / len(question_tokens)


def context_relevance(
    reference: str,
    contexts: list[str],
) -> float:
    _validate_text(reference, "reference")

    if not contexts:
        raise ValueError("contexts cannot be empty.")

    reference_tokens = _tokens(reference)

    if not reference_tokens:
        return 0.0

    context_tokens = set()

    for context in contexts:
        _validate_text(context, "context")
        context_tokens.update(_tokens(context))

    if not context_tokens:
        return 0.0

    return len(
        reference_tokens.intersection(context_tokens)
    ) / len(context_tokens)


def context_recall(
    ground_truth: str,
    contexts: list[str],
) -> float:
    _validate_text(ground_truth, "ground_truth")

    if not contexts:
        raise ValueError("contexts cannot be empty.")

    ground_truth_tokens = _tokens(ground_truth)

    if not ground_truth_tokens:
        return 0.0

    context_tokens = set()

    for context in contexts:
        _validate_text(context, "context")
        context_tokens.update(_tokens(context))

    return len(
        ground_truth_tokens.intersection(context_tokens)
    ) / len(ground_truth_tokens)


def faithfulness(
    answer: str,
    contexts: list[str],
) -> float:
    _validate_text(answer, "answer")

    if not contexts:
        raise ValueError("contexts cannot be empty.")

    answer_tokens = _tokens(answer)

    if not answer_tokens:
        return 0.0

    context_tokens = set()

    for context in contexts:
        _validate_text(context, "context")
        context_tokens.update(_tokens(context))

    return len(
        answer_tokens.intersection(context_tokens)
    ) / len(answer_tokens)