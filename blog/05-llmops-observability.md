# LLMOps Observability: MLflow + Langfuse + Prometheus for Local Models

*Published: Week 5 of Ollama Optimizer v2 development*

Running an LLM without observability is flying blind. You don't know which prompts break, which models drift, or where costs concentrate. In this post I'll share how I wired a full LLMOps stack into Ollama Optimizer v2 — using only free tools.

## The Three Layers of LLM Observability

| Layer | Tool | Answers |
|-------|------|---------|
| **Infra metrics** | Prometheus | "How many requests? Latency p95? Cache hit rate?" |
| **LLM traces** | Langfuse | "What was the exact prompt? Quality score? User session?" |
| **Model registry** | MLflow | "Which model version is in prod? What were its eval scores?" |

Skipping any one leaves blind spots.

## Layer 1: Prometheus — the Foundation

Standard HTTP API metrics first:

```python
REQUESTS = Counter("ollama_optimizer_requests_total", "...",
                   labelnames=["model", "complexity", "variant"])
LATENCY = Histogram("ollama_optimizer_latency_seconds", "...",
                    labelnames=["model", "complexity"],
                    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0))
TTFT = Histogram("ollama_optimizer_ttft_seconds", "...", labelnames=["model"])
TOKENS = Counter("ollama_optimizer_tokens_total", "...",
                 labelnames=["model", "type"])  # type: prompt|completion
CACHE_HITS = Counter(...); CACHE_MISSES = Counter(...)
```

Key label choices:
- `model` — isolate performance per model version
- `complexity` — correlate routing with latency
- `variant` — A/B test group (control vs treatment)
- `type` — separate prompt tokens (cheap) from completion tokens (expensive)

Exposed at `/metrics` in Prometheus text format. Grafana scrapes every 15s.

### Why histograms, not averages

Averages hide bimodal distributions. A mix of cached 50ms and uncached 2s responses averages to 1s, which tells you nothing. Histograms with sensible buckets show you p50/p95/p99 for free.

## Layer 2: Langfuse — LLM-specific Tracing

Langfuse understands LLMs natively — prompts, completions, tokens, scores, sessions. Free tier is generous (50K events/month).

```python
class LangfuseTracer:
    def trace_generation(self, name, model, prompt, output, usage,
                         metadata=None, user_id=None, session_id=None):
        trace = self.client.trace(
            name=name, user_id=user_id, session_id=session_id, metadata=metadata
        )
        trace.generation(
            name=f"{name}-gen",
            model=model,
            input=prompt,
            output=output,
            usage=usage,
        )
        return trace.id
```

What this unlocks:
- **Session view** — see all turns of a conversation
- **Latency breakdown** — TTFT vs generation vs post-processing
- **Quality scoring** — attach eval scores to traces
- **Prompt versioning** — track which prompt template produced which output

### Graceful degradation

Langfuse is optional. If keys aren't set, tracer becomes a no-op:

```python
def __init__(self, public_key, secret_key, host):
    self.enabled = LANGFUSE_AVAILABLE and bool(public_key and secret_key)
```

Users can run without MLOps infrastructure for local dev, opt in for prod.

## Layer 3: MLflow — Model Registry

MLflow tracks **what changed** and **why** over time. For our optimizer, a "model" is a tuning configuration (quantization, num_gpu, context):

```python
def log_benchmark_run(self, model_name, tuning_config, metrics, modelfile_content):
    with mlflow.start_run(run_name=f"tune-{model_name}") as run:
        mlflow.log_params(tuning_config)    # quant, gpu_layers, ctx
        mlflow.log_metrics(metrics)         # tokens_per_sec, quality_score
        mlflow.log_text(modelfile_content, "Modelfile")
        return run.info.run_id
```

Then promote the best run to Production:

```python
def promote_to_production(self, model_name, version):
    self.client.transition_model_version_stage(
        name=f"ollama-{model_name}-optimized",
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
```

Now every deployment references `models:/ollama-llama3.2-optimized/Production`, not a hardcoded version. Roll back = promote older version.

## A Full Request, Traced

Here's what one request generates:

**Prometheus:**
```
ollama_optimizer_requests_total{model="llama3.2:3b",complexity="medium",variant="control"} 1
ollama_optimizer_latency_seconds_bucket{model="llama3.2:3b",le="1.0"} 1
ollama_optimizer_tokens_total{model="llama3.2:3b",type="completion"} 147
```

**Langfuse trace:**
```json
{
  "id": "trace-abc123",
  "user_id": "user-42",
  "session_id": "sess-xyz",
  "name": "chat-completion",
  "input": "Summarize the article...",
  "output": "The article covers three main...",
  "model": "llama3.2:3b",
  "usage": {"prompt_tokens": 523, "completion_tokens": 147},
  "latency": 0.68,
  "metadata": {"complexity": "medium", "cached": false}
}
```

**MLflow:** unchanged (model registry is queried, not updated per request).

## Free-Tier Deployment

For personal projects:

| Service | Tier | Limits |
|---------|------|--------|
| Prometheus | Self-host on Render (free) | 512 MB RAM works for ~1M metrics |
| Grafana Cloud | Free | 10K metrics, 14-day retention |
| Langfuse Cloud | Free | 50K events/month |
| MLflow | Self-host via Docker | No limits |

Total monthly cost: **$0**. Yet this is the same stack startups use to run millions of queries.

## Drift Detection (Lightweight Version)

Once you have Langfuse data, feed it back for drift detection:

```python
class DriftDetector:
    def check(self, current_prompts):
        # Statistical drift: mean length, entropy
        ref_len = _avg_length(self._reference)
        cur_len = _avg_length(current_prompts)
        delta = (cur_len - ref_len) / ref_len * 100
        if abs(delta) > 20:
            return DriftReport(drift_detected=True, metric="length", delta_pct=delta)
```

For production, I use **Evidently AI** which handles this with pandas DataFrames and produces HTML reports. The lightweight version is pure Python — no deps, runs anywhere.

## Why Not Just OpenTelemetry?

OTel is great for traditional microservices. For LLMs, you lose domain-specific abstractions:

- Prompt/completion aren't first-class concepts
- Token counts are custom span attributes (fragile)
- No built-in quality scoring

Langfuse is purpose-built. Use OTel for HTTP layer, Langfuse for LLM semantics.

## The Payoff

With all three layers instrumented, you can answer questions like:
- "Which prompt style causes 90%+ of my p99 latency?" (Langfuse filter)
- "Did routing changes improve mean latency?" (Prometheus rate/histogram)
- "Why did the new model version score lower on helpfulness?" (MLflow run comparison)

These are questions that matter. They're invisible without the stack.

## What's Next

In Part 6, the **launch post**: real benchmark numbers, blog post analytics, lessons learned, and where the project goes from here.

Code: [github.com/Mahesh2023/ollama-optimizer-v2](https://github.com/Mahesh2023/ollama-optimizer-v2)
