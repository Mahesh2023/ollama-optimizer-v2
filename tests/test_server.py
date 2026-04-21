"""Basic FastAPI server tests."""
from fastapi.testclient import TestClient

from cmd.server.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_system_endpoint():
    with TestClient(app) as client:
        r = client.get("/system")
        assert r.status_code == 200
        data = r.json()
        assert "cpu_cores" in data
        assert "ram_mb" in data
        assert "total_vram_mb" in data


def test_metrics_endpoint():
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "ollama_optimizer_requests_total" in r.text


def test_routing_table_endpoint():
    with TestClient(app) as client:
        r = client.get("/admin/routing")
        assert r.status_code == 200
        assert "simple" in r.json()
        assert "complex" in r.json()
