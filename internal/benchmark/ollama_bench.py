"""Ollama benchmarking engine."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx


@dataclass
class BenchmarkResult:
    model: str
    prompt_name: str
    ttft_ms: float  # Time To First Token
    tokens_per_sec: float
    total_tokens: int
    prompt_tokens: int
    total_time_s: float
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def benchmark_ollama(
    model: str,
    prompt: str,
    prompt_name: str = "custom",
    base_url: str = "http://localhost:11434",
    options: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> BenchmarkResult:
    """Benchmark a single Ollama generate call."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }
    if options:
        payload["options"] = options

    start = time.perf_counter()
    ttft = None
    tokens = 0
    prompt_tokens = 0

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{base_url}/api/generate", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if ttft is None and data.get("response"):
                        ttft = (time.perf_counter() - start) * 1000

                    if data.get("done"):
                        tokens = data.get("eval_count", 0)
                        prompt_tokens = data.get("prompt_eval_count", 0)
                        break

        total = time.perf_counter() - start
        return BenchmarkResult(
            model=model,
            prompt_name=prompt_name,
            ttft_ms=ttft or 0.0,
            tokens_per_sec=tokens / total if total > 0 and tokens > 0 else 0.0,
            total_tokens=tokens,
            prompt_tokens=prompt_tokens,
            total_time_s=total,
            success=True,
        )
    except Exception as e:  # noqa: BLE001
        total = time.perf_counter() - start
        return BenchmarkResult(
            model=model,
            prompt_name=prompt_name,
            ttft_ms=0.0,
            tokens_per_sec=0.0,
            total_tokens=0,
            prompt_tokens=0,
            total_time_s=total,
            success=False,
            error=str(e),
        )


async def benchmark_suite(
    model: str,
    prompts: list[tuple[str, str]],  # [(name, prompt), ...]
    base_url: str = "http://localhost:11434",
    warmup: bool = True,
) -> list[BenchmarkResult]:
    """Run full benchmark suite against a model."""
    if warmup:
        await benchmark_ollama(
            model, "Hello", "warmup", base_url, timeout=60
        )

    results = []
    for name, prompt in prompts:
        result = await benchmark_ollama(model, prompt, name, base_url)
        results.append(result)
    return results


def aggregate_results(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Compute summary stats."""
    successful = [r for r in results if r.success]
    if not successful:
        return {"error": "all benchmarks failed"}

    tps = [r.tokens_per_sec for r in successful]
    ttfts = [r.ttft_ms for r in successful]

    return {
        "model": successful[0].model,
        "n_prompts": len(results),
        "n_success": len(successful),
        "avg_tokens_per_sec": sum(tps) / len(tps),
        "min_tokens_per_sec": min(tps),
        "max_tokens_per_sec": max(tps),
        "avg_ttft_ms": sum(ttfts) / len(ttfts),
        "total_tokens": sum(r.total_tokens for r in successful),
    }
