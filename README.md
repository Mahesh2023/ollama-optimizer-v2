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

- [x] Week 1: Hardware detection + CLI
- [x] Week 2: Benchmarking engine
- [x] Week 3: Auto-tuning
- [x] Week 4: Multi-model router + caching
- [x] Week 5: Observability dashboard
- [x] Week 6: Deploy + launch
- [x] Week 7: LLMOps (MLflow + Langfuse)
- [x] Week 8: Eval pipelines
- [x] Week 9: CI/CD for LLMs (DVC + CML)
- [x] Week 10: Drift detection

## License

Apache 2.0
