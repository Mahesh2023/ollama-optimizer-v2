# Architecture

## High-Level Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        Client (any language)                 │
│                OpenAI-compatible /v1/chat/completions         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Ollama Optimizer Router API                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │
│  │ Query Class- │ │ A/B Test     │ │ Prompt Cache │          │
│  │  ifier       │→│  Router      │→│  (Redis)     │          │
│  └──────────────┘ └──────────────┘ └──────────────┘          │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │        Telemetry: Prometheus + Langfuse              │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────┬───────────────────────────────────────┘
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Ollama   │   │ Ollama   │   │ Ollama   │
│ (1B)     │   │ (3B)     │   │ (7B+)    │
│ simple   │   │ medium   │   │ complex  │
└──────────┘   └──────────┘   └──────────┘
```

## Component Breakdown

### 1. Hardware Detection (`internal/hardware/`)

- Detects NVIDIA (via `nvidia-smi`), Apple Silicon (unified memory), AMD (`rocm-smi`)
- MLX backend detection for Apple Silicon
- Selects optimal Ollama backend (metal/cuda/rocm/cpu)

### 2. Benchmark Engine (`internal/benchmark/`)

- Streaming-based TTFT and tokens/sec measurement
- Standard prompt suite covering 7 categories
- Aggregation and comparison utilities

### 3. Auto-Tuner (`internal/tuner/`)

- Quantization selection based on VRAM + quality tradeoff
- GPU layer offload calculator
- Context length estimator
- Modelfile generator for Ollama

### 4. Smart Router (`internal/router/`)

- Heuristic query classifier (simple/medium/complex)
- Routing table mapping complexity → model
- A/B testing framework with consistent hashing

### 5. LLMOps Stack (`internal/registry/`, `internal/telemetry/`)

- **MLflow** model registry for versioning tuned configs
- **Langfuse** traces for every generation with user/session context
- **Prometheus** metrics: requests, tokens, latency, TTFT, cache stats

### 6. Evaluation (`internal/eval/`)

- LLM-as-judge evaluation on multiple criteria
- Dataset-level summaries
- PR-integration via GitHub Actions

### 7. Drift Detection (`internal/monitoring/`)

- Lightweight statistical drift (length, entropy)
- Optional Evidently AI integration for full reports

### 8. Caching (`internal/cache/`)

- Redis-backed exact-match prompt cache
- TTL-based expiry
- Hit/miss metrics

## Data Flow: A Single Request

1. Client POSTs to `/v1/chat/completions`
2. Router extracts last message as prompt
3. **Classifier** runs on prompt → complexity label
4. **Routing table** maps complexity → Ollama model
5. **A/B test router** may override model based on user hash
6. **Cache** check: hit returns immediately with `cached=true` metadata
7. **Ollama generate** call with routed model
8. **Metrics** emitted: requests, latency, TTFT, tokens
9. **Langfuse** traces full generation with metadata
10. **Cache** stored for future identical prompts
11. Response returned in OpenAI format

## MLOps Lifecycle

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Benchmark  │→→→│   Tune      │→→→│   Evaluate  │→→→│   Promote   │
│  (TTFT,tps) │   │ (quant,ctx) │   │ (LLM judge) │   │ (MLflow)    │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
                                                              │
                                                              ▼
                                                    ┌─────────────┐
                                                    │   Deploy    │
                                                    │  (A/B test) │
                                                    └─────────────┘
                                                              │
                                                              ▼
                                                    ┌─────────────┐
                                                    │  Monitor    │
                                                    │ (drift +    │
                                                    │  Langfuse)  │
                                                    └─────────────┘
```

## Deployment Options

### Local Development
- Single `ollama-opt serve` command
- Redis optional for caching
- SQLite for metadata

### Docker Compose
- Bundle: optimizer + Ollama + Redis + MLflow + Prometheus + Grafana
- Single `docker compose up` brings up full stack

### Render (Free Tier)
- Router API only (no GPU needed)
- Use external Ollama (Modal, Replicate, local with tunnel)
- Redis as separate Render service

### Full Production
- Kubernetes with HPA
- External MLflow (Databricks, self-hosted)
- Langfuse Cloud
- Grafana Cloud for dashboards

## Technology Choices

| Layer | Tool | Why |
|-------|------|-----|
| HTTP | FastAPI | Async-native, OpenAPI auto-gen |
| Schema | Pydantic v2 | Type safety + validation |
| Cache | Redis | Industry standard, async client |
| Metrics | prometheus_client | Native Python, standard format |
| Tracing | Langfuse | LLM-specific, generous free tier |
| Registry | MLflow | De-facto standard for ML |
| CLI | Typer | Type-annotated, Rich integration |
| Testing | pytest + httpx | Async support, TestClient |
| Deployment | Docker + Render | Simple, free tier available |
