# Ollama Optimizer v2

> Production-grade LLMOps platform for local LLM inference. Auto-tunes, benchmarks, routes, monitors.

[![CI](https://github.com/Mahesh2023/ollama-optimizer-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/Mahesh2023/ollama-optimizer-v2/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Hardware auto-detection** — NVIDIA CUDA, Apple Silicon (Metal), CPU fallback
- **Benchmark engine** — TTFT, tokens/sec, memory across quantizations
- **Auto-tuning** — optimal quantization + GPU layer offload for your hardware
- **Smart router** — OpenAI-compatible API that picks model by query complexity
- **LLMOps** — MLflow model registry, Langfuse tracing, A/B testing, drift detection
- **Prompt caching** — Redis-backed exact & semantic cache
- **Evaluation** — LLM-as-judge automated quality checks

## Quick Start

```bash
# Install
pip install -e .

# Detect your hardware
ollama-opt detect

# Benchmark a model across quantizations
ollama-opt bench --model llama3.2:3b

# Auto-tune for optimal config
ollama-opt tune --model llama3.2:3b

# Start the routing server
ollama-opt serve --port 8000
```

## Architecture

```
             ┌──────────────┐
  Client ───▶│  Router API  │
             └──────┬───────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Small model (1B)    Large model (7B+)
          │                   │
          └─────────┬─────────┘
                    ▼
             ┌──────────────┐
             │  Observability │
             │  MLflow + Langfuse
             └──────────────┘
```

## Roadmap

### v0.1.0 — Current (Alpha)
- Hardware detection (NVIDIA, Apple Silicon, CPU)
- Benchmarking engine (TTFT, tokens/sec, memory)
- Auto-tuning (quantization selection, GPU layer offload)
- Smart routing (query complexity-based model selection)
- OpenAI-compatible API
- Redis prompt caching
- MLflow model registry integration
- Langfuse LLM observability
- Basic evaluation framework (LLM-as-judge)
- Drift detection (statistical)

### v0.2.0 — Planned (Q2 2026)
- Semantic caching (vector-based)
- Multi-GPU support (tensor parallelism)
- Streaming response optimization
- Advanced evaluation (RAGAS, TruLens)
- GPU autoscaling integration
- Kubernetes deployment manifests
- Model fine-tuning pipeline (LoRA adapters)
- Cost tracking and optimization alerts

### v1.0.0 — Future (Q3 2026)
- Enterprise auth (OAuth, SSO)
- Multi-tenant isolation
- Model marketplace integration
- Distributed inference across nodes
- Advanced prompt engineering tools
- Real-time collaboration features
- Comprehensive audit logs
- SLA monitoring and alerting

### Research & Exploration
- Speculative decoding
- Continuous batching
- Knowledge distillation pipeline
- Custom quantization formats
- Cross-platform inference optimization

## License

Apache 2.0
