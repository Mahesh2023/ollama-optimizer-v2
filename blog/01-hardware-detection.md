# Auto-Detecting Hardware for Local LLM Inference

*Published: Week 1 of Ollama Optimizer v2 development*

When you run a 7B model on an M2 MacBook vs. a NVIDIA A100 server, the optimal configuration differs wildly — quantization level, GPU layer offload, context length, batch size. Yet most local LLM tools ship with a single default that fits nobody perfectly.

In this post, I'll walk through how I built a hardware auto-detection module for Ollama Optimizer v2 that cleanly supports NVIDIA, Apple Silicon (M1/M2/M3), and AMD GPUs under one API.

## Why Hardware Detection Matters for LLMs

Modern LLM inference engines (Ollama, llama.cpp, vLLM, MLX) expose dozens of knobs:
- `num_gpu_layers` — how many transformer layers to offload to GPU
- `num_ctx` — context window size (directly affects KV cache memory)
- `num_batch` — prompt batch size
- Quantization choice (Q2_K through F16)

Getting these wrong means either OOM crashes or 10× slower inference. Getting them right means buttery-smooth streaming on modest hardware.

## Design Goals

1. **Zero dependencies beyond `psutil`** — no heavy CUDA toolkits required
2. **Platform-aware** — detect Apple Silicon vs. NVIDIA vs. AMD
3. **Vendor-specific metrics** — VRAM, driver version, compute capability
4. **Fail gracefully** — CPU-only fallback when no GPU found

## Detecting NVIDIA GPUs

The cleanest approach is parsing `nvidia-smi --query-gpu`:

```python
def detect_nvidia_gpus() -> list[GPUInfo]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []
    gpus = []
    for line in out.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, vram, driver, cc = parts
        gpus.append(GPUInfo(
            name=name, vram_mb=int(vram),
            driver_version=driver, compute_capability=cc,
            vendor="nvidia",
        ))
    return gpus
```

Key insights:
- CSV output is machine-parseable
- `--format=csv,noheader,nounits` strips units so you get integers
- Compute capability matters — `8.0` unlocks Tensor Cores, `9.0` unlocks Transformer Engine

## Apple Silicon is Different

On Apple Silicon, there's no discrete GPU with its own VRAM. Instead, **unified memory** means the GPU shares the entire system RAM. This is a game-changer for LLMs because:

- No PCIe transfer bottleneck
- You can run models larger than any discrete consumer GPU
- KV cache and activations stay on the same physical memory

Detection uses platform checks + `sysctl`:

```python
def detect_apple_silicon_gpu() -> list[GPUInfo]:
    if platform.system() != "Darwin":
        return []
    if "arm" not in platform.machine().lower():
        return []
    mem_mb = psutil.virtual_memory().total // (1024 * 1024)
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
    return [GPUInfo(
        name=f"{chip} (unified memory)",
        vram_mb=mem_mb,
        driver_version="Metal",
        vendor="apple",
    )]
```

## The MLX Backend Bonus

Apple's MLX framework is dramatically faster than llama.cpp on M-series chips for some workloads. Detection lets us recommend the best backend:

```python
def detect_mlx() -> MLXCapabilities:
    if platform.system() != "Darwin" or "arm" not in platform.machine().lower():
        return MLXCapabilities(available=False)
    try:
        import mlx.core as mx
        return MLXCapabilities(
            available=True,
            recommended_backend="mlx",
            notes="MLX + Metal available.",
        )
    except ImportError:
        return MLXCapabilities(available=False, recommended_backend="metal")
```

## Putting It Together

```bash
$ ollama-opt detect
┌──────────────┬──────────────────────────────┐
│ Property     │ Value                        │
├──────────────┼──────────────────────────────┤
│ Platform     │ darwin                       │
│ CPU          │ Apple M2 (8C/8T)             │
│ RAM          │ 16,384 MB (16.0 GB)          │
│ Apple Silicon│ YES                          │
│ GPUs         │ 1                            │
│ Total VRAM   │ 16,384 MB (unified memory)   │
└──────────────┴──────────────────────────────┘
```

This structured output then feeds the auto-tuner, which I'll cover in the next post: how I decide Q4_K_M vs. Q5_K_M based on available VRAM while maximizing quality.

## What's Next

In Part 2, I'll cover the benchmarking engine — why TTFT matters more than tokens/sec for chat UX, and how to build a prompt suite that exposes real-world performance characteristics.

Code: [github.com/Mahesh2023/ollama-optimizer-v2](https://github.com/Mahesh2023/ollama-optimizer-v2)

---

*This is part 1 of a series documenting the build of a production-grade LLMOps platform for local LLM inference.*
