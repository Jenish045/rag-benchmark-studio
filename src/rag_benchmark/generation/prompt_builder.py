SYSTEM_PROMPT = """You are a question-answering assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information to answer the question,
say that the answer cannot be determined from the provided context.

Do not invent facts, sources, or citations.

When using information from the context, refer to the corresponding source
using its source label.
"""


def build_prompt(query: str, context: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be a non-empty string.")

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n"
        f"{context}\n\n"
        f"Question:\n"
        f"{query}\n\n"
        f"Answer:"
    )