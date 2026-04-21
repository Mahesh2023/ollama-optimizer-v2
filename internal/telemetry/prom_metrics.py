"""Prometheus metrics."""
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

REQUESTS = Counter(
    "ollama_optimizer_requests_total",
    "Total requests",
    labelnames=["model", "complexity", "variant"],
)

TOKENS = Counter(
    "ollama_optimizer_tokens_total",
    "Total tokens processed",
    labelnames=["model", "type"],  # type: prompt, completion
)

LATENCY = Histogram(
    "ollama_optimizer_latency_seconds",
    "Request latency",
    labelnames=["model", "complexity"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

TTFT = Histogram(
    "ollama_optimizer_ttft_seconds",
    "Time to first token",
    labelnames=["model"],
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)

CACHE_HITS = Counter("ollama_optimizer_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("ollama_optimizer_cache_misses_total", "Cache misses")


def metrics_response() -> tuple[bytes, str]:
    """Return latest metrics in Prometheus exposition format."""
    return generate_latest(), CONTENT_TYPE_LATEST
