"""Tests for hardware detection."""
from unittest.mock import patch

from internal.hardware.detector import GPUInfo, SystemInfo, detect_system
from internal.hardware.mlx_backend import detect_mlx, select_ollama_backend


def test_system_info_structure():
    info = detect_system()
    assert isinstance(info, SystemInfo)
    assert info.cpu_cores >= 1
    assert info.cpu_threads >= info.cpu_cores
    assert info.ram_mb > 0
    assert info.platform in ("linux", "darwin", "windows")


def test_system_info_to_dict():
    info = detect_system()
    d = info.to_dict()
    assert "total_vram_mb" in d
    assert "gpus" in d
    assert "platform" in d


def test_gpu_info_to_dict():
    g = GPUInfo(name="Test GPU", vram_mb=8192, vendor="nvidia")
    d = g.to_dict()
    assert d["name"] == "Test GPU"
    assert d["vram_mb"] == 8192


def test_total_vram_computation():
    info = SystemInfo(
        os="Linux", arch="x86_64", cpu_model="Xeon",
        cpu_cores=8, cpu_threads=16, ram_mb=32768,
        platform="linux", is_apple_silicon=False,
        gpus=[
            GPUInfo(name="A", vram_mb=8192),
            GPUInfo(name="B", vram_mb=8192),
        ],
    )
    assert info.total_vram_mb == 16384
    assert info.has_gpu


def test_has_gpu_false_with_empty_gpu_list():
    info = SystemInfo(
        os="Linux", arch="x86_64", cpu_model="Xeon",
        cpu_cores=8, cpu_threads=16, ram_mb=32768,
        platform="linux", is_apple_silicon=False, gpus=[],
    )
    assert not info.has_gpu
    assert info.total_vram_mb == 0


def test_detect_mlx_on_non_mac():
    with patch("platform.system", return_value="Linux"):
        caps = detect_mlx()
        assert not caps.available
        assert "Apple Silicon only" in caps.notes


def test_select_ollama_backend_returns_known_value():
    backend = select_ollama_backend()
    assert backend in ("metal", "cuda", "rocm", "cpu")
