"""Smart router: picks best model based on query complexity."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryComplexity(str, Enum):
    SIMPLE = "simple"    # Short factual Q&A
    MEDIUM = "medium"    # Code gen, short summaries
    COMPLEX = "complex"  # Multi-step reasoning, long context


# Default routing table (override via config/registry)
MODEL_ROUTING: dict[QueryComplexity, str] = {
    QueryComplexity.SIMPLE: "llama3.2:1b",
    QueryComplexity.MEDIUM: "llama3.2:3b",
    QueryComplexity.COMPLEX: "qwen2.5:7b",
}


COMPLEX_KEYWORDS = re.compile(
    r"\b("
    r"analyze|reasoning|step by step|explain why|prove|derive|"
    r"multi[-\s]step|compare and contrast|pros and cons|"
    r"code review|refactor|architecture|design patterns"
    r")\b",
    re.IGNORECASE,
)

CODE_KEYWORDS = re.compile(
    r"\b(write code|implement|function|class|algorithm|debug|"
    r"python|javascript|rust|golang|typescript)\b",
    re.IGNORECASE,
)


@dataclass
class RoutingDecision:
    complexity: QueryComplexity
    model: str
    reasoning: str
    prompt_tokens_estimate: int


def classify_query(prompt: str) -> QueryComplexity:
    """Heuristic query complexity classification.

    Upgrade path: fine-tune a small classifier (distilbert) for higher accuracy.
    """
    word_count = len(prompt.split())
    has_complex_kw = bool(COMPLEX_KEYWORDS.search(prompt))
    has_code_kw = bool(CODE_KEYWORDS.search(prompt))

    # Very long prompts usually need capable models
    if word_count > 500:
        return QueryComplexity.COMPLEX

    # Explicit reasoning keywords
    if has_complex_kw:
        return QueryComplexity.COMPLEX

    # Short factual questions
    if word_count < 20 and prompt.rstrip().endswith("?"):
        return QueryComplexity.SIMPLE

    # Code generation
    if has_code_kw:
        return QueryComplexity.MEDIUM

    # Default medium
    if word_count < 100:
        return QueryComplexity.MEDIUM

    return QueryComplexity.COMPLEX


@dataclass
class SmartRouter:
    routing_table: dict[QueryComplexity, str]

    @classmethod
    def default(cls) -> SmartRouter:
        return cls(routing_table=dict(MODEL_ROUTING))

    def route(self, prompt: str) -> RoutingDecision:
        complexity = classify_query(prompt)
        model = self.routing_table[complexity]
        return RoutingDecision(
            complexity=complexity,
            model=model,
            reasoning=f"Classified as {complexity.value} (len={len(prompt.split())})",
            prompt_tokens_estimate=len(prompt.split()) * 4 // 3,  # rough ratio
        )

    def override(self, complexity: QueryComplexity, model: str) -> None:
        self.routing_table[complexity] = model
