"""Automated LLM evaluation using LLM-as-a-judge."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from internal.benchmark.ollama_bench import benchmark_ollama

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are an expert evaluator. Rate the response on the given criterion from 1 (worst) to 10 (best).

Criterion: {criterion}

User Prompt:
{prompt}

Response to Evaluate:
{response}

{reference_section}

Return ONLY a JSON object like {{"score": 8, "reasoning": "..."}}.
"""


@dataclass
class EvalResult:
    prompt: str
    response: str
    criterion: str
    score: float  # 0.0..10.0
    reasoning: str
    model_under_test: str
    judge_model: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSummary:
    model: str
    n_samples: int
    mean_score: float
    per_criterion: dict[str, float] = field(default_factory=dict)
    results: list[EvalResult] = field(default_factory=list)


class ModelEvaluator:
    """LLM-as-a-judge evaluator."""

    DEFAULT_CRITERIA = [
        "correctness",     # Is the answer factually correct?
        "relevance",       # Is the answer on-topic?
        "clarity",         # Is the answer clear and readable?
        "completeness",    # Does it fully address the prompt?
    ]

    def __init__(
        self,
        judge_model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        criteria: list[str] | None = None,
    ) -> None:
        self.judge_model = judge_model
        self.base_url = base_url
        self.criteria = criteria or self.DEFAULT_CRITERIA

    async def _judge(
        self,
        prompt: str,
        response: str,
        criterion: str,
        reference: str | None = None,
    ) -> tuple[float, str]:
        ref_section = f"Reference Answer:\n{reference}\n" if reference else ""
        judge_prompt = JUDGE_PROMPT.format(
            criterion=criterion,
            prompt=prompt,
            response=response,
            reference_section=ref_section,
        )
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.judge_model, "prompt": judge_prompt, "stream": False},
            )
            r.raise_for_status()
            text = r.json().get("response", "")

        # Extract JSON
        match = re.search(r'\{.*?"score".*?\}', text, re.DOTALL)
        if not match:
            return 0.0, f"Could not parse judge response: {text[:200]}"
        try:
            data = json.loads(match.group())
            return float(data.get("score", 0)), str(data.get("reasoning", ""))
        except (json.JSONDecodeError, ValueError):
            return 0.0, f"Parse error: {text[:200]}"

    async def evaluate_single(
        self,
        prompt: str,
        model: str,
        reference: str | None = None,
    ) -> list[EvalResult]:
        """Run model once, then judge on all criteria."""
        bench = await benchmark_ollama(model, prompt, base_url=self.base_url)
        if not bench.success:
            return []

        # Retrieve actual response via non-streaming call
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            r.raise_for_status()
            response_text = r.json().get("response", "")

        results = []
        for criterion in self.criteria:
            score, reasoning = await self._judge(prompt, response_text, criterion, reference)
            results.append(EvalResult(
                prompt=prompt,
                response=response_text,
                criterion=criterion,
                score=score,
                reasoning=reasoning,
                model_under_test=model,
                judge_model=self.judge_model,
                metadata={"tokens_per_sec": bench.tokens_per_sec},
            ))
        return results

    async def evaluate_dataset(
        self,
        dataset: list[dict[str, Any]],  # [{prompt, reference?}, ...]
        model: str,
    ) -> EvalSummary:
        all_results: list[EvalResult] = []
        for item in dataset:
            prompt = item["prompt"]
            reference = item.get("reference")
            results = await self.evaluate_single(prompt, model, reference)
            all_results.extend(results)

        per_criterion: dict[str, list[float]] = {}
        for r in all_results:
            per_criterion.setdefault(r.criterion, []).append(r.score)

        mean_score = (
            sum(r.score for r in all_results) / len(all_results) if all_results else 0.0
        )
        return EvalSummary(
            model=model,
            n_samples=len(dataset),
            mean_score=mean_score,
            per_criterion={k: sum(v)/len(v) for k, v in per_criterion.items()},
            results=all_results,
        )
