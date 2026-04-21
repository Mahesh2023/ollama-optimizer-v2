"""Quantization and auto-tuning logic."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from internal.hardware.detector import SystemInfo


# Bytes-per-parameter ratios for each quantization level (relative to F32 = 4 bytes/param)
# Values represent typical disk/memory footprint in GB per billion parameters.
QUANT_RATIOS: dict[str, float] = {
    "F32": 4.0,
    "F16": 2.0,
    "Q8_0": 1.0,
    "Q6_K": 0.8,
    "Q5_K_M": 0.7,
    "Q5_0": 0.68,
    "Q4_K_M": 0.6,
    "Q4_0": 0.55,
    "Q3_K_M": 0.45,
    "Q2_K": 0.35,
}

# Quantization quality rankings (1.0 = F16 baseline, lower is worse quality)
QUANT_QUALITY: dict[str, float] = {
    "F32": 1.0,
    "F16": 1.0,
    "Q8_0": 0.995,
    "Q6_K": 0.99,
    "Q5_K_M": 0.98,
    "Q5_0": 0.97,
    "Q4_K_M": 0.96,  # Sweet spot
    "Q4_0": 0.93,
    "Q3_K_M": 0.88,
    "Q2_K": 0.80,
}

# Preferred order (best quality that fits goes first)
PREFERRED_ORDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"]


@dataclass
class TuningResult:
    model: str
    selected_quant: str
    model_params_b: float
    estimated_size_gb: float
    available_vram_gb: float
    num_gpu_layers: int
    context_length: int
    quality_score: float
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_optimal_quant(
    model_params_b: float,
    available_vram_mb: int,
    headroom_fraction: float = 0.2,
    min_quality: float = 0.90,
) -> tuple[str, str]:
    """Select largest quantization that fits in VRAM with headroom.

    Returns (quant_name, reasoning).
    """
    available_gb = available_vram_mb / 1024
    usable_gb = available_gb * (1 - headroom_fraction)

    for quant in PREFERRED_ORDER:
        size_gb = model_params_b * QUANT_RATIOS[quant]
        quality = QUANT_QUALITY[quant]
        if size_gb <= usable_gb and quality >= min_quality:
            reasoning = (
                f"{quant} ({size_gb:.1f}GB) fits in {usable_gb:.1f}GB usable VRAM "
                f"with quality score {quality:.2f}"
            )
            return quant, reasoning

    return "Q4_0", f"Fallback to Q4_0; model may be larger than available VRAM ({available_gb:.1f}GB)"


def calculate_gpu_layers(
    model_layers: int,
    model_size_gb: float,
    available_vram_mb: int,
    overhead_gb: float = 2.0,
) -> int:
    """Calculate optimal number of GPU layers to offload."""
    available_gb = available_vram_mb / 1024 - overhead_gb
    if available_gb <= 0:
        return 0
    if model_size_gb <= available_gb:
        return model_layers  # All layers on GPU

    fraction = available_gb / model_size_gb
    return int(model_layers * fraction)


def estimate_context_length(
    model_params_b: float,
    available_vram_mb: int,
    target_ctx: int = 4096,
) -> int:
    """Estimate maximum safe context length given VRAM.

    KV cache memory ~ 2 * n_layers * ctx * dim * 2 bytes (FP16).
    Simplified: use heuristic of ~0.5MB per 1K context per 1B params.
    """
    # Reserve 80% for weights
    ctx_vram_mb = available_vram_mb * 0.2
    mb_per_1k_ctx = 0.5 * model_params_b
    if mb_per_1k_ctx <= 0:
        return target_ctx
    max_ctx = int((ctx_vram_mb / mb_per_1k_ctx) * 1024)
    return min(max_ctx, target_ctx)


def tune_model(
    model: str,
    model_params_b: float,
    sys_info: SystemInfo,
    model_layers: int = 32,
    target_ctx: int = 4096,
) -> TuningResult:
    """Generate optimal tuning configuration."""
    available_vram_mb = sys_info.total_vram_mb
    if available_vram_mb == 0:
        available_vram_mb = sys_info.ram_mb // 2  # CPU inference fallback

    quant, reasoning = select_optimal_quant(model_params_b, available_vram_mb)
    size_gb = model_params_b * QUANT_RATIOS[quant]
    gpu_layers = calculate_gpu_layers(model_layers, size_gb, available_vram_mb) if sys_info.has_gpu else 0
    ctx = estimate_context_length(model_params_b, available_vram_mb, target_ctx)

    return TuningResult(
        model=model,
        selected_quant=quant,
        model_params_b=model_params_b,
        estimated_size_gb=size_gb,
        available_vram_gb=available_vram_mb / 1024,
        num_gpu_layers=gpu_layers,
        context_length=ctx,
        quality_score=QUANT_QUALITY[quant],
        reasoning=reasoning,
        metadata={
            "platform": sys_info.platform,
            "is_apple_silicon": sys_info.is_apple_silicon,
            "num_gpus": len(sys_info.gpus),
        },
    )


def generate_modelfile(result: TuningResult, base_model: str) -> str:
    """Generate Ollama Modelfile from tuning result."""
    return f"""FROM {base_model}

# Auto-tuned by ollama-optimizer v2
# Quantization: {result.selected_quant}
# Quality score: {result.quality_score:.2f}
# Reasoning: {result.reasoning}

PARAMETER num_ctx {result.context_length}
PARAMETER num_gpu {result.num_gpu_layers}
PARAMETER num_thread 0
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
