# Quickstart

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) running locally
- Redis (optional, for caching): `brew install redis && brew services start redis`
- Docker (optional, for docker-compose deployment)

## Installation

```bash
git clone https://github.com/Mahesh2023/ollama-optimizer-v2.git
cd ollama-optimizer-v2

# Option 1: Makefile (recommended)
make install

# Option 2: pip
pip install -e ".[dev,llmops]"
```

## First Run

### 1. Detect Hardware

```bash
ollama-opt detect
```

Expected output on M2 Mac:
```
┌──────────────┬────────────────────────────────────┐
│ Property     │ Value                              │
├──────────────┼────────────────────────────────────┤
│ OS           │ macOS-14.x-arm64                   │
│ Arch         │ arm64                              │
│ Platform     │ darwin                             │
│ CPU          │ Apple M2 (8C/8T)                   │
│ RAM          │ 16,384 MB (16.0 GB)                │
│ Apple Silicon│ YES                                │
│ GPUs         │ 1                                  │
│ Total VRAM   │ 16,384 MB (unified memory)         │
└──────────────┴────────────────────────────────────┘
```

### 2. Pull a Model

```bash
ollama pull llama3.2:3b
```

### 3. Benchmark It

```bash
ollama-opt bench --model llama3.2:3b
```

### 4. Auto-Tune

```bash
ollama-opt tune --model llama3.2:3b --params-b 3.0 --modelfile ./Modelfile.optimized
ollama create llama3.2:3b-opt -f Modelfile.optimized
```

### 5. Start the Router API

```bash
# Option A: direct
ollama-opt serve --port 8000

# Option B: via docker-compose (brings up Redis + MLflow + Grafana)
cd deploy && docker compose up -d
```

### 6. Test OpenAI-compatible Endpoint

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

The server will **auto-route** to the smallest suitable model (llama3.2:1b for simple questions, etc.).

## LLMOps Setup (Optional)

### MLflow (Model Registry)

```bash
make run-mlflow  # Starts MLflow on :5000
```

Visit http://localhost:5000 to browse runs and registered models.

### Langfuse (LLM Tracing)

1. Sign up at https://cloud.langfuse.com (free tier)
2. Copy keys into `.env`:
   ```bash
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   ```
3. Every request will now be traced. View at Langfuse dashboard.

### Prometheus + Grafana

Via docker-compose:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Deploy to Render

```bash
# Push to GitHub first
git push origin main

# Then on Render:
# - Connect repo
# - Use deploy/render.yaml blueprint
# - Free tier: CPU-only, ~512MB RAM (sufficient for router API)
```

Note: Render free tier spins down after 15min idle. For always-on inference,
run Ollama locally and proxy via your router deployed on Render.

## Apple Silicon Optimization (M1/M2/M3)

Install MLX for native Metal inference:

```bash
pip install mlx mlx-lm
```

The hardware detector will now report MLX availability:

```bash
ollama-opt detect  # Will show MLX: available
```

## Troubleshooting

### Ollama connection refused
Make sure Ollama is running: `ollama serve`

### Redis connection refused
Install and start Redis: `brew install redis && brew services start redis`

### Metal OOM on M2
Try smaller quantization: `ollama-opt tune --model llama3.2:3b --params-b 3.0` then follow the generated Modelfile with Q4_K_M or lower.
