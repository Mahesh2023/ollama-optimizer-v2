"""Tests for quantization tuner."""
from internal.hardware.detector import GPUInfo, SystemInfo
from internal.tuner.quant import (
    QUANT_QUALITY,
    QUANT_RATIOS,
    calculate_gpu_layers,
    estimate_context_length,
    generate_modelfile,
    select_optimal_quant,
    tune_model,
)


def _sys_info(vram_mb: int = 0, apple: bool = False) -> SystemInfo:
    gpus = [GPUInfo(name="Test", vram_mb=vram_mb)] if vram_mb > 0 else []
    return SystemInfo(
        os="test", arch="arm64", cpu_model="test",
        cpu_cores=8, cpu_threads=16, ram_mb=16384,
        platform="darwin" if apple else "linux",
        is_apple_silicon=apple, gpus=gpus,
    )


def test_quant_ratios_monotonic():
    """Higher bit quants should have higher ratios."""
    assert QUANT_RATIOS["F16"] > QUANT_RATIOS["Q8_0"]
    assert QUANT_RATIOS["Q8_0"] > QUANT_RATIOS["Q4_K_M"]
    assert QUANT_RATIOS["Q4_K_M"] > QUANT_RATIOS["Q2_K"]


def test_quality_scores_monotonic():
    """Quality should decrease with lower quantization."""
    assert QUANT_QUALITY["F16"] >= QUANT_QUALITY["Q8_0"]
    assert QUANT_QUALITY["Q8_0"] >= QUANT_QUALITY["Q4_K_M"]


def test_select_optimal_quant_fits_large_vram():
    quant, _ = select_optimal_quant(model_params_b=3.0, available_vram_mb=24000)
    # With 24GB for 3B model, F16 (6GB) should fit
    assert quant in ("F16", "Q8_0")


def test_select_optimal_quant_tight_vram():
    quant, _ = select_optimal_quant(model_params_b=7.0, available_vram_mb=4096)
    # 4GB can't fit 7B even at Q2_K — fallback
    assert quant in ("Q4_0", "Q3_K_M", "Q2_K")


def test_gpu_layers_all_fit():
    layers = calculate_gpu_layers(
        model_layers=32, model_size_gb=2.0, available_vram_mb=8192,
    )
    assert layers == 32


def test_gpu_layers_partial_offload():
    layers = calculate_gpu_layers(
        model_layers=32, model_size_gb=8.0, available_vram_mb=4096,
    )
    assert 0 < layers < 32


def test_gpu_layers_zero_if_no_vram():
    layers = calculate_gpu_layers(
        model_layers=32, model_size_gb=2.0, available_vram_mb=0,
    )
    assert layers == 0


def test_estimate_context_length():
    ctx = estimate_context_length(
        model_params_b=3.0, available_vram_mb=8192, target_ctx=4096,
    )
    assert 0 < ctx <= 4096


def test_tune_model_full_flow():
    sys = _sys_info(vram_mb=4096, apple=True)
    result = tune_model("llama3.2:3b", model_params_b=3.0, sys_info=sys)
    assert result.model == "llama3.2:3b"
    assert result.selected_quant in QUANT_RATIOS
    assert result.num_gpu_layers >= 0
    assert result.context_length > 0
    assert 0 < result.quality_score <= 1.0


def test_tune_model_no_gpu_fallback():
    sys = _sys_info(vram_mb=0)
    result = tune_model("llama3.2:1b", model_params_b=1.0, sys_info=sys)
    assert result.num_gpu_layers == 0
    assert result.available_vram_gb > 0  # Uses RAM fallback


def test_generate_modelfile_content():
    sys = _sys_info(vram_mb=4096, apple=True)
    result = tune_model("llama3.2:3b", 3.0, sys)
    mf = generate_modelfile(result, "llama3.2:3b")
    assert "FROM llama3.2:3b" in mf
    assert "PARAMETER num_ctx" in mf
    assert "PARAMETER num_gpu" in mf
