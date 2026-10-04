from collections.abc import Iterable

from langchain_core.documents import Document


def _validate_inputs(
    retrieved_documents: list[Document],
    relevant_sources: Iterable[str],
    k: int,
) -> set[str]:
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if not isinstance(retrieved_documents, list):
        raise ValueError("retrieved_documents must be a list.")

    relevant_sources = set(relevant_sources)

    if not relevant_sources:
        raise ValueError("relevant_sources cannot be empty.")

    return relevant_sources


def recall_at_k(
    retrieved_documents: list[Document],
    relevant_sources: Iterable[str],
    k: int,
) -> float:
    relevant_sources = _validate_inputs(
        retrieved_documents,
        relevant_sources,
        k,
    )

    retrieved_sources = {
        document.metadata.get("source")
        for document in retrieved_documents[:k]
    }

    retrieved_relevant = retrieved_sources.intersection(
        relevant_sources
    )

    return len(retrieved_relevant) / len(relevant_sources)


def hit_rate_at_k(
    retrieved_documents: list[Document],
    relevant_sources: Iterable[str],
    k: int,
) -> float:
    relevant_sources = _validate_inputs(
        retrieved_documents,
        relevant_sources,
        k,
    )

    retrieved_sources = {
        document.metadata.get("source")
        for document in retrieved_documents[:k]
    }

    return float(
        bool(retrieved_sources.intersection(relevant_sources))
    )


def reciprocal_rank(
    retrieved_documents: list[Document],
    relevant_sources: Iterable[str],
) -> float:
    relevant_sources = set(relevant_sources)

    if not relevant_sources:
        raise ValueError("relevant_sources cannot be empty.")

    for rank, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        source = document.metadata.get("source")

        if source in relevant_sources:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    rankings: list[list[Document]],
    relevant_sources_list: list[Iterable[str]],
) -> float:
    if not rankings:
        raise ValueError("rankings cannot be empty.")

    if len(rankings) != len(relevant_sources_list):
        raise ValueError(
            "rankings and relevant_sources_list must have "
            "the same length."
        )

    reciprocal_ranks = [
        reciprocal_rank(
            ranking,
            relevant_sources,
        )
        for ranking, relevant_sources in zip(
            rankings,
            relevant_sources_list,
        )
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)