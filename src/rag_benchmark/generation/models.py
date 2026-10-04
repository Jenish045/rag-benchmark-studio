from dataclasses import dataclass


@dataclass
class GenerationResult:
    answer: str
    citations: list[str]
    context: str