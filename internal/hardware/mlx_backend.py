"""Apple MLX framework detection and capabilities.

MLX is Apple's native array framework optimized for Apple Silicon (M1/M2/M3).
It leverages unified memory for zero-copy GPU operations and is the fastest
local inference backend on Apple hardware.
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MLXCapabilities:
    available: bool
    version: str = ""
    metal_available: bool = False
    unified_memory: bool = True
    recommended_backend: str = "cpu"  # "mlx", "metal", "cpu", "cuda"
    notes: str = ""


def detect_mlx() -> MLXCapabilities:
    """Detect MLX framework availability on Apple Silicon."""
    if platform.system() != "Darwin":
        return MLXCapabilities(available=False, notes="MLX is Apple Silicon only.")

    machine = platform.machine().lower()
    is_apple_silicon = "arm" in machine or "aarch64" in machine
    if not is_apple_silicon:
        return MLXCapabilities(available=False, notes="Not Apple Silicon.")

    try:
        import mlx.core as mx  # type: ignore
        version = getattr(mx, "__version__", "unknown")
        return MLXCapabilities(
            available=True,
            version=version,
            metal_available=True,
            unified_memory=True,
            recommended_backend="mlx",
            notes="MLX + Metal available. Use MLX for fastest inference.",
        )
    except ImportError:
        return MLXCapabilities(
            available=False,
            metal_available=True,
            unified_memory=True,
            recommended_backend="metal",
            notes="MLX not installed, but Metal is available. pip install mlx mlx-lm",
        )


def get_metal_device_info() -> dict[str, str]:
    """Get Metal device info using system_profiler on macOS."""
    if platform.system() != "Darwin":
        return {}

    import subprocess
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            timeout=5, text=True
        )
        import json
        data = json.loads(out)
        displays = data.get("SPDisplaysDataType", [])
        if displays:
            d = displays[0]
            return {
                "name": d.get("sppci_model", "Unknown Metal GPU"),
                "vendor": d.get("spdisplays_vendor", "Apple"),
                "metal_family": d.get("spdisplays_metalfamily", "unknown"),
                "vram_mb": d.get("spdisplays_vram", "0 MB").replace(" MB", "").replace(",", ""),
            }
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError) as e:
        logger.debug("system_profiler failed: %s", e)
    return {}


def select_ollama_backend() -> str:
    """Return the Ollama backend flag for current hardware.

    Ollama auto-detects, but we can override via OLLAMA_ env vars.
    """
    if platform.system() == "Darwin":
        # Ollama uses Metal automatically on macOS
        return "metal"
    # Check for NVIDIA
    import shutil
    if shutil.which("nvidia-smi"):
        return "cuda"
    # Check for ROCm
    if shutil.which("rocm-smi"):
        return "rocm"
    return "cpu"
