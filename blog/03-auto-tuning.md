# Smart Quantization: Saving 40% VRAM Without Losing Quality

*Published: Week 3 of Ollama Optimizer v2 development*

LLM quantization is the art of representing model weights in fewer bits. Less memory, faster inference, but sometimes worse quality. How do you pick the right level for your hardware?

Most people just guess. I built an auto-tuner that picks optimally based on available VRAM and quality tolerance.

## The Quantization Landscape

From highest to lowest quality (and largest to smallest):

| Quant | Bytes/param | Quality (F16=1.0) | Notes |
|-------|-------------|-------------------|-------|
| F16 | 2.0 | 1.00 | Full precision, research use |
| Q8_0 | 1.0 | 0.995 | Near-lossless, 2× smaller |
| Q6_K | 0.8 | 0.99 | High quality, good speed |
| Q5_K_M | 0.7 | 0.98 | Great tradeoff |
| **Q4_K_M** | **0.6** | **0.96** | **Sweet spot for most uses** |
| Q4_0 | 0.55 | 0.93 | Older, slightly worse than K-family |
| Q3_K_M | 0.45 | 0.88 | Noticeable quality drop |
| Q2_K | 0.35 | 0.80 | Extreme compression, visible quality loss |

The **K-family** (Q4_K_M, Q5_K_M, etc.) uses smarter block-wise quantization. They're almost always better than same-bit-depth non-K variants.

## The Selection Algorithm

Given a model's parameter count and available VRAM, pick the largest quantization that fits with headroom:

```python
PREFERRED_ORDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"]

def select_optimal_quant(model_params_b, available_vram_mb, headroom_fraction=0.2):
    available_gb = available_vram_mb / 1024
    usable_gb = available_gb * (1 - headroom_fraction)

    for quant in PREFERRED_ORDER:
        size_gb = model_params_b * QUANT_RATIOS[quant]
        quality = QUANT_QUALITY[quant]
        if size_gb <= usable_gb and quality >= 0.90:
            return quant

    return "Q4_0"  # fallback
```

Key decisions:
- **20% headroom** — accounts for KV cache, activations, OS overhead
- **Quality floor of 0.90** — refuse to go below Q3_K_M unless forced
- **Best-first search** — picks F16 if it fits, degrades gracefully

## GPU Layer Offload

If a model doesn't fully fit on GPU, you can offload some layers to CPU. This is much slower, but the fraction matters:

```python
def calculate_gpu_layers(model_layers, model_size_gb, available_vram_mb, overhead_gb=2.0):
    available_gb = available_vram_mb / 1024 - overhead_gb
    if available_gb <= 0:
        return 0
    if model_size_gb <= available_gb:
        return model_layers
    return int(model_layers * (available_gb / model_size_gb))
```

A 7B model at Q4_K_M is ~4.2 GB. With only 3 GB of available VRAM, you offload ~21 of 32 layers.

## Context Length Estimation

KV cache memory scales with context × params. On small VRAM, requesting 32K context will OOM:

```python
def estimate_context_length(model_params_b, available_vram_mb, target_ctx=4096):
    # Reserve 80% of VRAM for weights; 20% for KV cache + activations
    ctx_vram_mb = available_vram_mb * 0.2
    mb_per_1k_ctx = 0.5 * model_params_b  # heuristic
    max_ctx = int((ctx_vram_mb / mb_per_1k_ctx) * 1024)
    return min(max_ctx, target_ctx)
```

For a 3B model on 4 GB VRAM: ~819 MB for KV cache → ~5400 tokens max context. Round down to 4K.

## The Full Tuner

```python
def tune_model(model, model_params_b, sys_info, model_layers=32, target_ctx=4096):
    available_vram_mb = sys_info.total_vram_mb or sys_info.ram_mb // 2

    quant, reasoning = select_optimal_quant(model_params_b, available_vram_mb)
    size_gb = model_params_b * QUANT_RATIOS[quant]
    gpu_layers = calculate_gpu_layers(model_layers, size_gb, available_vram_mb)
    ctx = estimate_context_length(model_params_b, available_vram_mb, target_ctx)

    return TuningResult(
        model=model,
        selected_quant=quant,
        num_gpu_layers=gpu_layers,
        context_length=ctx,
        quality_score=QUANT_QUALITY[quant],
        reasoning=reasoning,
    )
```

## Output: A Generated Modelfile

The tuner emits a ready-to-use Ollama Modelfile:

```
FROM llama3.2:3b-q4_K_M

# Auto-tuned by ollama-optimizer v2
# Quantization: Q4_K_M
# Quality score: 0.96
# Reasoning: Q4_K_M (1.8GB) fits in 3.2GB usable VRAM with quality score 0.96

PARAMETER num_ctx 4096
PARAMETER num_gpu 32
PARAMETER num_thread 0
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

Build with: `ollama create llama3.2:3b-opt -f Modelfile.optimized`

## Real Numbers on My M2

Running `ollama-opt tune --model llama3.2:3b --params-b 3.0`:

- Detected: Apple M2, 16 GB unified memory
- Selected: Q5_K_M (not Q4_K_M — there's plenty of room)
- Context: 4096 (could go higher, but default is safe)
- Quality: 0.98

For a 7B model on the same machine:

- Selected: Q4_K_M
- Context: 4096
- Quality: 0.96
- Reasoning: Q5_K_M (4.9 GB) would be tight with KV cache

## Future Work: Latency-Aware Tuning

Pure "fit-biggest-quant" isn't always optimal. Q4_K_M is sometimes **faster** than Q5_K_M on M2 because of SIMD dispatch differences. I'm adding a benchmark-driven tuning mode that actually measures speed at each quant.

## What's Next

In Part 4: **Building an OpenAI-compatible router** that picks models based on query complexity — so simple questions use a 1B model (fast, cheap) while complex reasoning routes to 7B.

Code: [github.com/Mahesh2023/ollama-optimizer-v2](https://github.com/Mahesh2023/ollama-optimizer-v2)
