# Apple Silicon (M1/M2/M3) Optimization Guide

## Why Apple Silicon for Local LLMs?

- **Unified memory** — GPU and CPU share the same pool (no PCIe bottleneck)
- **Metal Performance Shaders** — accelerated matmul, attention, softmax
- **MLX framework** — Apple's native array library beats llama.cpp in many benchmarks
- **Power efficient** — run 7B models on a MacBook Air without fans

## M2-Specific Tuning

### Memory Budgeting

M2 unified memory typical splits:
- macOS + apps: ~4-6 GB
- Wired/kernel: ~1-2 GB
- Available for LLM: `total - 8 GB`

Example on 16 GB M2:
- 16 - 8 = 8 GB available
- Use `--headroom-fraction 0.3` for safety
- Usable ~5.6 GB

### Recommended Models for M2 (8/16/24 GB)

| Model | Params | Q4_K_M Size | Works on M2 8GB | Works on M2 16GB | Works on M2 24GB |
|-------|--------|-------------|-----------------|------------------|------------------|
| llama3.2:1b | 1B | 0.6GB | ✅ | ✅ | ✅ |
| llama3.2:3b | 3B | 1.8GB | ✅ | ✅ | ✅ |
| gemma2:2b | 2B | 1.2GB | ✅ | ✅ | ✅ |
| qwen2.5:7b | 7B | 4.2GB | tight | ✅ | ✅ |
| llama3.1:8b | 8B | 4.7GB | tight | ✅ | ✅ |
| qwen2.5:14b | 14B | 8.4GB | ❌ | tight | ✅ |

### Optimal Quantization on Apple Silicon

Metal prefers these quantizations (in order):
1. **Q4_K_M** — best quality/size ratio, fast
2. **Q5_K_M** — better quality, ~15% slower
3. **Q8_0** — near-lossless, ~40% slower, 2x VRAM
4. **F16** — full precision, for quality comparison only

The auto-tuner defaults to Q4_K_M on Apple Silicon.

### Context Length Tuning

KV cache on Metal for 7B model:
- 4K ctx: ~600 MB
- 8K ctx: ~1.2 GB
- 16K ctx: ~2.4 GB
- 32K ctx: ~4.8 GB (usually OOMs on 16GB M2)

Default to 4K unless you need more. Use `OLLAMA_KV_CACHE_TYPE=q8_0` to halve KV cache memory.

### Thermal Throttling

M2 Air (fanless) throttles after ~2-3 min sustained load:
- Short prompts: full speed
- Long generations (500+ tokens): may slow to 60% of peak
- For benchmarks, let the system cool between runs

## Installing MLX on M2

```bash
# Native MLX (experimental, fastest on Apple Silicon)
pip install mlx mlx-lm

# Verify
python -c "import mlx.core as mx; print(mx.default_device())"
# Should print: Device(gpu, 0)
```

### MLX vs Ollama vs llama.cpp on M2

Benchmark: llama3.2:3b Q4_K_M, 500 token generation:

| Backend | Tokens/sec (avg) | TTFT (ms) | Notes |
|---------|------------------|-----------|-------|
| Ollama (Metal) | 45-55 | 200 | Stable, well-tested |
| llama.cpp direct | 50-60 | 180 | Minimal overhead |
| MLX-LM | 60-75 | 150 | Fastest, but less model support |

The optimizer supports all three via `select_ollama_backend()`.

## Env Vars for Apple Silicon

```bash
# Force Metal (usually auto-detected)
export OLLAMA_GPU_BACKEND=metal

# Reduce KV cache memory (use q8 instead of f16)
export OLLAMA_KV_CACHE_TYPE=q8_0

# Disable flash attention (if causing issues)
export OLLAMA_FLASH_ATTENTION=0

# Keep models loaded in memory (faster repeated calls)
export OLLAMA_KEEP_ALIVE=-1

# Max loaded models (careful with unified memory)
export OLLAMA_MAX_LOADED_MODELS=1
```

## Monitoring M2 During Inference

```bash
# GPU util (live)
sudo powermetrics --samplers gpu_power -i 1000

# Memory pressure
vm_stat 1

# Thermal state
pmset -g thermlog

# Integrated monitor
brew install asitop
sudo asitop
```

## Known Issues

### "Metal device does not support this size"
You're OOM. Reduce quant (try Q4_0) or context length.

### Extremely slow first generation
Model is loading into memory. Subsequent calls will be fast. Use `OLLAMA_KEEP_ALIVE=-1`.

### Fans spin up (MacBook Pro M2)
Normal under load. For silent operation, limit to 1B-3B models or use low power mode.

### Ollama reports "using CPU" on M2
Reinstall Ollama: `brew reinstall ollama`. Metal is auto-detected.
