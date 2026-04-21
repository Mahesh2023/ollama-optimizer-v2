"""Hardware detection for LLM inference optimization."""
from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

import psutil


@dataclass
class GPUInfo:
    name: str
    vram_mb: int
    driver_version: str = ""
    cuda_version: str | None = None
    compute_capability: str | None = None
    vendor: str = "unknown"  # nvidia, apple, amd, intel

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SystemInfo:
    os: str
    arch: str
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    ram_mb: int
    platform: str
    is_apple_silicon: bool
    gpus: list[GPUInfo] = field(default_factory=list)

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_mb for g in self.gpus)

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "total_vram_mb": self.total_vram_mb}


def _run(cmd: list[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return ""


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
        try:
            gpus.append(GPUInfo(
                name=name,
                vram_mb=int(vram),
                driver_version=driver,
                compute_capability=cc,
                vendor="nvidia",
            ))
        except ValueError:
            continue
    return gpus


def detect_apple_silicon_gpu() -> list[GPUInfo]:
    if platform.system() != "Darwin":
        return []
    machine = platform.machine().lower()
    if "arm" not in machine and "aarch64" not in machine:
        return []
    # Apple Silicon has unified memory
    mem_mb = psutil.virtual_memory().total // (1024 * 1024)
    # Get chip model
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
    return [GPUInfo(
        name=f"{chip} (unified memory)",
        vram_mb=mem_mb,
        driver_version="Metal",
        vendor="apple",
    )]


def detect_amd_gpus() -> list[GPUInfo]:
    # ROCm rocm-smi integration
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if not out:
        return []
    # Simplified parsing
    gpus = []
    for line in out.split("\n")[1:]:  # skip header
        if "," in line:
            try:
                parts = line.split(",")
                vram_mb = int(parts[1]) // (1024 * 1024)
                gpus.append(GPUInfo(
                    name="AMD GPU", vram_mb=vram_mb, vendor="amd"
                ))
            except (ValueError, IndexError):
                continue
    return gpus


def detect_system() -> SystemInfo:
    """Detect full system hardware for LLM inference optimization."""
    gpus = (
        detect_nvidia_gpus()
        + detect_apple_silicon_gpu()
        + detect_amd_gpus()
    )

    return SystemInfo(
        os=platform.platform(),
        arch=platform.machine(),
        cpu_model=platform.processor() or _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "unknown",
        cpu_cores=psutil.cpu_count(logical=False) or 1,
        cpu_threads=psutil.cpu_count(logical=True) or 1,
        ram_mb=psutil.virtual_memory().total // (1024 * 1024),
        platform=platform.system().lower(),
        is_apple_silicon=(
            platform.system() == "Darwin"
            and "arm" in platform.machine().lower()
        ),
        gpus=gpus,
    )
