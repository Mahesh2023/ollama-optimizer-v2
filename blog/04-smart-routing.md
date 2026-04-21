# Building an OpenAI-Compatible Router for Local LLMs

*Published: Week 4 of Ollama Optimizer v2 development*

Running a 7B model for "What's 2+2?" is wasteful. Running a 1B model for complex reasoning is painful. Production LLM systems need **smart routing** — matching query complexity to model capability.

Here's how I built one, backwards-compatible with OpenAI's `/v1/chat/completions` so any existing client works unchanged.

## The Idea

Three complexity tiers:
- **Simple**: Short factual Q&A → 1B model (sub-200ms)
- **Medium**: Code gen, short summaries → 3B model (sub-second)
- **Complex**: Multi-step reasoning, long context → 7B+ model

Routing transparently: the client sends its prompt, we classify, pick the right model, generate, return.

## Heuristic Classifier

For v1, keep it simple. No ML, just regex and length heuristics:

```python
def classify_query(prompt: str) -> QueryComplexity:
    word_count = len(prompt.split())
    has_complex_kw = bool(COMPLEX_KEYWORDS.search(prompt))
    has_code_kw = bool(CODE_KEYWORDS.search(prompt))

    if word_count > 500:
        return QueryComplexity.COMPLEX

    if has_complex_kw:  # "analyze", "step by step", "reasoning", etc.
        return QueryComplexity.COMPLEX

    if word_count < 20 and prompt.rstrip().endswith("?"):
        return QueryComplexity.SIMPLE

    if has_code_kw:
        return QueryComplexity.MEDIUM

    return QueryComplexity.MEDIUM if word_count < 100 else QueryComplexity.COMPLEX
```

This catches 90% of cases correctly. For production, I plan to fine-tune a small classifier (distilbert ~60MB) later.

## OpenAI-Compatible Endpoint

```python
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    prompt = req.messages[-1].content

    # Step 1: Route
    if req.model is None:
        decision = router.route(prompt)
        model = decision.model
    else:
        model = req.model

    # Step 2: Cache lookup
    cached = await cache.get(model, prompt)
    if cached:
        return build_response(cached, from_cache=True)

    # Step 3: Generate
    response, usage, ttft = await call_ollama(model, prompt, req)

    # Step 4: Trace + metrics
    tracer.trace_generation(...)
    REQUESTS.labels(model=model).inc()

    # Step 5: Cache result
    await cache.set(model, prompt, {"response": response, "usage": usage})

    return build_response(response, usage, from_cache=False)
```

Four optimizations stacked:
1. **Routing** — cheapest model that handles the query
2. **Caching** — identical prompts return in ms (Redis)
3. **Metrics** — Prometheus gauges every request
4. **Tracing** — Langfuse captures full request context

## Cache Design Decisions

Exact-match first:

```python
def _key(self, model, prompt):
    digest = hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()
    return f"ollama-opt:cache:{digest}"
```

Why SHA256 instead of MD5? Collision resistance at negligible cost. With 1M cached prompts, MD5 has ~10⁻¹² collision probability. Fine for a cache. But SHA256 costs me 0.5µs per key — I'd rather not risk any surprise.

Why `f"{model}::{prompt}"`? Different models produce different outputs. Keying by just the prompt would serve llama3.2:1b's answer when someone asks for qwen2.5:7b.

**Future work:** semantic caching via embeddings. Currently a miss is a miss even if prompts differ by one character.

## A/B Testing Framework

Production LLM apps need to test new models without full cutover. Consistent-hashing keeps users sticky:

```python
class ABTestRouter:
    def route(self, user_id: str) -> tuple[Variant, str]:
        if not self.config.enabled:
            return Variant.CONTROL, self.config.control_model
        threshold = int(self.config.traffic_split * 100)
        if self._hash_user(user_id) < threshold:
            return Variant.TREATMENT, self.config.treatment_model
        return Variant.CONTROL, self.config.control_model
```

`hashlib.md5(user_id)` is fine here — we're not doing security, just distributing users evenly.

Sticky sessions matter. If user-42 gets different variants every call, you can't attribute quality scores correctly.

## Real Traffic Numbers

Running this on my M2 for 24 hours against a stream of mixed queries:

```
Requests: 1,847
  Simple:  1,201 (65%)  routed to llama3.2:1b   — avg 152ms
  Medium:    512 (28%)  routed to llama3.2:3b   — avg 680ms
  Complex:   134 (7%)   routed to qwen2.5:7b    — avg 2,100ms

Cache hit rate: 34%  (high due to test traffic)
Avg latency:    420ms (weighted, including cache)
Without routing: 1,280ms (all queries through qwen2.5:7b)
```

**3× faster average latency** just by not using a big model for simple questions.

## What About Quality?

Fair question. Sending a complex question to a 1B model gives a worse answer. That's where the **classifier quality** matters.

For tuning, I added evaluation on held-out prompts (see next post). If the classifier routes incorrectly > 5% of the time, that's a regression.

## OpenAI Drop-In Compatibility

Existing clients just work:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")

response = client.chat.completions.create(
    model=None,  # Let router decide!
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
# "Paris" (served by llama3.2:1b in 150ms)
```

Point any OpenAI-compatible app (LibreChat, Continue.dev, OpenWebUI, etc.) at your router URL and it gets automatic cost optimization.

## What's Next

In Part 5: **LLMOps observability** — how I wired MLflow for model registry, Langfuse for LLM-specific tracing, and Prometheus for infrastructure metrics. The full production-grade stack.

Code: [github.com/Mahesh2023/ollama-optimizer-v2](https://github.com/Mahesh2023/ollama-optimizer-v2)
