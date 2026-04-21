# Benchmarking LLMs: Why tokens/sec Lies to You

*Published: Week 2 of Ollama Optimizer v2 development*

Every LLM tool reports tokens-per-second. It's become the default way to compare local inference speed. But tokens/sec is a **misleading metric** for real applications. In this post, I'll explain why, and walk through the benchmarking engine I built that captures what actually matters.

## The Problem with tokens/sec

Say Model A does 50 tokens/sec and Model B does 40 tokens/sec. Model A wins, right?

Not necessarily. Consider:

| Metric | Model A | Model B |
|--------|---------|---------|
| Tokens/sec | 50 | 40 |
| Time to First Token (TTFT) | 2000ms | 400ms |
| Context prefill time (4K ctx) | 5s | 1s |
| Memory peak | 5.5 GB | 3.8 GB |
| Quality (LLM-judge) | 8.2 | 8.5 |

For a chat application, Model B is objectively better — users see a response 5× faster and it uses less memory. The tokens/sec advantage of Model A only shows up in the middle of long responses users never wait for.

## What to Actually Measure

### 1. Time to First Token (TTFT)

The single most important metric for UX. If TTFT > 1 second, the experience feels laggy regardless of throughput.

```python
start = time.perf_counter()
ttft = None
async for line in resp.aiter_lines():
    if ttft is None and data.get("response"):
        ttft = (time.perf_counter() - start) * 1000
```

### 2. Tokens/sec (steady-state)

Still useful, but measured **after** the first token (otherwise you're averaging prefill).

### 3. Memory Peak

On resource-constrained systems (M2 Air, edge devices), this decides feasibility more than speed.

### 4. Thermal throttling

On fanless devices, sustained load cuts sustained throughput by 30-40%. Include a long-running test.

## The Prompt Suite

Generic "lorem ipsum" prompts don't expose real-world behavior. I built 7 prompt categories to stress-test different aspects:

```python
STANDARD_PROMPTS = [
    BenchmarkPrompt("short_qa", ...),       # Factoid Q&A — short prefill, short completion
    BenchmarkPrompt("medium_essay", ...),   # Long completion — tests sustained throughput
    BenchmarkPrompt("code_gen", ...),       # Code — checks structured output quality
    BenchmarkPrompt("summary", ...),        # Summarization — medium prefill
    BenchmarkPrompt("reasoning", ...),      # Chain-of-thought — complex routing
    BenchmarkPrompt("long_context", ...),   # 2K+ tokens — tests KV cache
    BenchmarkPrompt("translation", ...),    # Short + structured
]
```

Each captures a distinct performance profile. A model optimized for short Q&A might tank on long-context work.

## The Streaming Benchmark Core

```python
async def benchmark_ollama(model, prompt, base_url, options=None):
    start = time.perf_counter()
    ttft = None
    tokens = 0

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", f"{base_url}/api/generate",
                                  json={"model": model, "prompt": prompt, "stream": True}) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)

                if ttft is None and data.get("response"):
                    ttft = (time.perf_counter() - start) * 1000

                if data.get("done"):
                    tokens = data.get("eval_count", 0)
                    break

    total = time.perf_counter() - start
    return BenchmarkResult(
        model=model,
        ttft_ms=ttft or 0.0,
        tokens_per_sec=tokens / total if total > 0 else 0.0,
        total_tokens=tokens,
        total_time_s=total,
    )
```

Key details:
- **Streaming mode** — avoids artificial latency from buffering
- **httpx.AsyncClient** — async means you can benchmark multiple models in parallel
- **Separate TTFT timer** — captures UX quality, not just throughput

## Warmup Matters

First inference on a cold model can be 10× slower than subsequent calls:

```python
if warmup:
    await benchmark_ollama(model, "Hello", "warmup", base_url, timeout=60)
```

Always warm up before benchmarking. If you don't, you're measuring model load time, not inference speed.

## Aggregation: Beyond Averages

Raw averages hide variance. Report:

```python
def aggregate_results(results):
    tps = [r.tokens_per_sec for r in results if r.success]
    return {
        "avg_tokens_per_sec": sum(tps) / len(tps),
        "min_tokens_per_sec": min(tps),
        "max_tokens_per_sec": max(tps),
        "p95_tokens_per_sec": sorted(tps)[int(len(tps) * 0.95)],  # not shown here
        "n_success": len(tps),
    }
```

p95 exposes tail latency, which matters for production.

## Real Results on M2

Running `ollama-opt bench --model llama3.2:3b` on a 16GB M2:

```
Prompt              TTFT    tokens/sec   Total
short_qa            180ms   52.3         23 tok
medium_essay        220ms   48.7         412 tok
code_gen            195ms   50.1         187 tok
reasoning           310ms   46.2         298 tok   ← complex prompt slower
long_context        890ms   39.8         156 tok   ← big prefill
```

TTFT jumps 5× for long-context because prefill dominates. That's the real constraint for RAG applications, not tokens/sec.

## What's Next

In Part 3, I'll cover the **auto-tuning engine** — how to pick Q4_K_M vs. Q5_K_M automatically based on your hardware and quality requirements.

Code: [github.com/Mahesh2023/ollama-optimizer-v2](https://github.com/Mahesh2023/ollama-optimizer-v2)
