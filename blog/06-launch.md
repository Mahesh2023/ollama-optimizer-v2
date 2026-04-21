# Launch: Ollama Optimizer v2 — A 6-Week Journey

*Published: Week 6 of Ollama Optimizer v2 development*

Six weeks ago, I started building a production-grade LLMOps platform for local LLM inference. Today, I'm launching Ollama Optimizer v2 as open-source. Here's what I built, what I learned, and what's next.

## What It Does

Ollama Optimizer v2 is a layer between your apps and Ollama that adds:

1. **Hardware auto-detection** — NVIDIA, Apple Silicon (M1/M2/M3), AMD
2. **Benchmark engine** — TTFT, tokens/sec, memory across quantizations
3. **Auto-tuning** — optimal quantization + GPU layer offload + context length
4. **Smart routing** — OpenAI-compatible API that picks models by query complexity
5. **Prompt caching** — Redis-backed, 34% hit rate in testing
6. **LLMOps stack** — MLflow model registry, Langfuse tracing, Prometheus metrics
7. **Evaluation** — LLM-as-judge automated quality checks
8. **Drift detection** — statistical drift on prompt/response distributions

All runs on a MacBook Air M2 (4GB GPU) or any Linux server with NVIDIA/AMD GPUs.

## The Numbers

### Benchmarks on M2 (16GB unified memory)

| Model | Quant | TTFT | tokens/sec | Memory |
|-------|-------|------|------------|--------|
| llama3.2:1b | Q4_K_M | 120ms | 58.3 | 0.6 GB |
| llama3.2:3b | Q4_K_M | 180ms | 52.1 | 1.8 GB |
| qwen2.5:7b | Q4_K_M | 320ms | 41.7 | 4.2 GB |

### Routing Impact

With smart routing enabled (1B/3B/7B tiered):

```
Avg latency: 420ms (vs 1,280ms without routing)
Cache hit rate: 34%
Cost per query: 0.003¢ (estimated based on compute time)
```

### Repository Stats

- **Files**: 58 (Python, TypeScript, Docker, configs)
- **Lines of code**: ~4,200 (excluding tests/docs)
- **Test coverage**: 85% (pytest)
- **Dependencies**: 15 (minimal footprint)

## What Went Right

### 1. Hardware Detection First

Starting with a clean hardware abstraction made everything easier. Adding MLX support for Apple Silicon was a 30-line change because the `GPUInfo` dataclass already existed.

### 2. Streaming Benchmarks

Measuring TTFT separately from throughput exposed real UX issues. A 7B model with 2000ms TTFT feels laggy even at 50 tokens/sec. I prioritized TTFT in tuning.

### 3. OpenAI Compatibility

The `/v1/chat/completions` endpoint meant I could test with existing tools immediately. LibreChat, Continue.dev, and OpenWebUI all worked without modification.

### 4. Graceful Degradation

Langfuse and MLflow are optional. The project runs without them. Users can start with zero MLOps infrastructure, opt in later.

## What Went Wrong

### 1. Over-Engineering the Classifier

I initially built a transformer-based query classifier. It added 200MB of dependencies and was marginally better than regex heuristics. Reverted to simple keywords for v1.

### 2. Ignoring Thermal Throttling on M2

My benchmarks assumed sustained throughput. On fanless M2 Air, after 2 minutes of continuous generation, tokens/sec dropped 40%. Added a thermal throttling note to docs, but should instrument it.

### 3. KV Cache Estimation Heuristic

The `0.5MB per 1K context per 1B params` rule works for llama.cpp but not vLLM. I need a per-backend estimator.

### 4. No Semantic Caching Yet

Exact-match caching works, but semantically similar prompts miss. Adding embedding-based semantic cache is next on the roadmap.

## The Stack

| Component | Tool | Why |
|-----------|------|-----|
| HTTP | FastAPI | Async-native, OpenAPI auto-gen |
| Schema | Pydantic v2 | Type safety, validation |
| Cache | Redis | Industry standard, async client |
| Metrics | prometheus_client | Native Python, standard format |
| Tracing | Langfuse | LLM-specific, generous free tier |
| Registry | MLflow | De-facto standard for ML |
| CLI | Typer | Type-annotated, Rich integration |
| Frontend | Next.js + Tailwind | Modern, SSR for SEO |
| Deployment | Docker + Render | Simple, free tier available |

## Deployment Strategy

**Free tier:**
- Router API on Render (CPU-only, 512MB RAM)
- Redis on Render (free tier)
- MLflow self-hosted via Docker on local machine
- Grafana Cloud free tier for dashboards
- Langfuse Cloud free tier for tracing

**Production:**
- Kubernetes with HPA
- External MLflow (Databricks or self-hosted)
- Langfuse Cloud or self-hosted
- Grafana Cloud or self-hosted

## What's Next

### Short Term (v0.2)
- [ ] Semantic caching via embeddings
- [ ] Fine-tuned query classifier (distilbert)
- [ ] Per-backend KV cache estimator
- [ ] Thermal throttling instrumentation

### Medium Term (v0.3)
- [ ] Distributed inference across multiple GPUs (tensor parallelism)
- [ ] Multi-tenant with team quotas
- [ ] Billing integration (Stripe)
- [ ] SaaS deployment (multi-tenant)

### Long Term (v1.0)
- [ ] Full benchmark comparison dashboard (Recharts)
- [ ] Model registry browser (MLflow integration)
- [ ] A/B test UI with statistical significance
- [ ] Prompt template versioning

## How to Try It

```bash
git clone https://github.com/Mahesh2023/ollama-optimizer-v2.git
cd ollama-optimizer-v2
make install
ollama-opt detect
ollama-opt bench --model llama3.2:3b
ollama-opt tune --model llama3.2:3b --params-b 3.0
ollama-opt serve
```

Then visit http://localhost:3000 for the dashboard.

## Acknowledgments

- Ollama team for the excellent local LLM engine
- Langfuse team for the generous free tier
- MLflow community for the model registry standard
- All contributors to llama.cpp, MLX, and the open-source LLM ecosystem

## Call for Contributors

This is a solo project today, but I'd love help:
- **Windows support** — currently untested
- **AMD ROCm** — I don't have AMD hardware to test
- **More benchmarks** — expand the prompt suite
- **Documentation** — tutorials, examples
- **Frontend** — the dashboard is minimal, needs love

Open an issue or PR. All contributions welcome.

---

**Repository:** https://github.com/Mahesh2023/ollama-optimizer-v2
**License:** Apache 2.0
**Blog series:** 6 posts covering hardware detection, benchmarking, auto-tuning, routing, LLMOps, and launch.
