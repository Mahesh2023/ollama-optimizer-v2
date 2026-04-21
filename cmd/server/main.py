"""FastAPI server: OpenAI-compatible routing API with LLMOps instrumentation."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from internal.cache.redis_cache import PromptCache
from internal.config import get_settings
from internal.hardware import detect_system
from internal.router.ab_test import ABTestConfig, ABTestRouter, Variant
from internal.router.router import QueryComplexity, SmartRouter
from internal.telemetry.langfuse_client import LangfuseTracer
from internal.telemetry.prom_metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    LATENCY,
    REQUESTS,
    TOKENS,
    TTFT,
    metrics_response,
)

logger = logging.getLogger(__name__)


# ---- Lifecycle --------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logger.info("Ollama Optimizer v2 starting up...")

    app.state.settings = settings
    app.state.router = SmartRouter.default()
    app.state.cache = PromptCache(redis_url=settings.redis_url)
    app.state.tracer = LangfuseTracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    app.state.ab_router = ABTestRouter(
        ABTestConfig(
            experiment_name="model-v2-test",
            control_model="llama3.2:3b",
            treatment_model="llama3.2:3b-optimized",
            traffic_split=settings.ab_traffic_split,
            enabled=False,  # toggle via /admin/ab-test
        )
    )
    app.state.system_info = detect_system()

    yield

    logger.info("Shutting down...")
    await app.state.cache.close()
    app.state.tracer.flush()


# ---- App --------------------------------------------------------------------

app = FastAPI(
    title="Ollama Optimizer v2",
    version="0.1.0",
    description="Production-grade LLMOps platform for local LLM inference.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Schemas (OpenAI-compatible) -------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None  # None = auto-route
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int | None = None
    stream: bool = False
    user: str = "anonymous"


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict[str, int]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---- Endpoints --------------------------------------------------------------

@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "service": "Ollama Optimizer v2",
        "version": "0.1.0",
        "description": "Production-grade LLMOps platform for local LLM inference",
        "endpoints": {
            "health": "/health",
            "system": "/system",
            "metrics": "/metrics",
            "chat_completions": "/v1/chat/completions",
            "cache_stats": "/admin/cache/stats",
            "routing_table": "/admin/routing",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "ollama-optimizer-v2", "version": "0.1.0"}


@app.get("/system")
async def system_info(request: Request) -> dict[str, Any]:
    """Return detected hardware capabilities."""
    info = request.app.state.system_info
    return info.to_dict()


@app.get("/metrics")
async def metrics() -> Response:
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    req: ChatCompletionRequest, request: Request
) -> ChatCompletionResponse:
    """OpenAI-compatible chat completion with smart routing + caching + A/B."""
    settings: Any = request.app.state.settings
    router: SmartRouter = request.app.state.router
    cache: PromptCache = request.app.state.cache
    tracer: LangfuseTracer = request.app.state.tracer
    ab: ABTestRouter = request.app.state.ab_router

    if not req.messages:
        return _error_response(req, "empty messages")

    prompt = req.messages[-1].content
    start = time.perf_counter()

    # Routing
    if req.model is None:
        decision = router.route(prompt)
        model = decision.model
        complexity = decision.complexity
    else:
        model = req.model
        complexity = QueryComplexity.MEDIUM

    # A/B test override
    variant, ab_model = ab.route(req.user)
    if variant == Variant.TREATMENT:
        model = ab_model

    # Cache lookup
    cached = await cache.get(model, prompt)
    if cached is not None:
        CACHE_HITS.inc()
        REQUESTS.labels(model=model, complexity=complexity.value, variant=variant.value).inc()
        return _build_response(req, model, cached["response"], cached["usage"], from_cache=True)

    CACHE_MISSES.inc()

    # Forward to Ollama
    full_prompt = _messages_to_prompt(req.messages)
    try:
        response_text, usage, ttft_s = await _call_ollama(
            settings.ollama_base_url, model, full_prompt, req
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ollama call failed")
        return _error_response(req, str(exc))

    latency = time.perf_counter() - start

    # Metrics
    REQUESTS.labels(model=model, complexity=complexity.value, variant=variant.value).inc()
    LATENCY.labels(model=model, complexity=complexity.value).observe(latency)
    TTFT.labels(model=model).observe(ttft_s)
    TOKENS.labels(model=model, type="prompt").inc(usage.get("prompt_tokens", 0))
    TOKENS.labels(model=model, type="completion").inc(usage.get("completion_tokens", 0))

    # Langfuse tracing
    tracer.trace_generation(
        name="chat-completion",
        model=model,
        prompt=prompt,
        output=response_text,
        usage=usage,
        user_id=req.user,
        metadata={
            "complexity": complexity.value,
            "variant": variant.value,
            "latency_s": latency,
            "ttft_s": ttft_s,
            "cached": False,
        },
    )

    # Cache successful response
    await cache.set(model, prompt, {"response": response_text, "usage": usage})

    return _build_response(req, model, response_text, usage, from_cache=False)


# ---- Helpers ---------------------------------------------------------------

def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Naive chat-template rendering; Ollama has per-model templates."""
    lines = []
    for m in messages:
        lines.append(f"[{m.role}] {m.content}")
    lines.append("[assistant]")
    return "\n".join(lines)


async def _call_ollama(
    base_url: str, model: str, prompt: str, req: ChatCompletionRequest
) -> tuple[str, dict[str, int], float]:
    """Call Ollama generate. Returns (text, usage, ttft_seconds)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": req.temperature,
            "top_p": req.top_p,
        },
    }
    if req.max_tokens:
        payload["options"]["num_predict"] = req.max_tokens

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{base_url}/api/generate", json=payload)
        r.raise_for_status()
        ttft_s = time.perf_counter() - start
        data = r.json()

    usage = {
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
    }
    return data.get("response", ""), usage, ttft_s


def _build_response(
    req: ChatCompletionRequest,
    model: str,
    text: str,
    usage: dict[str, int],
    from_cache: bool,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=model,
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=text),
        )],
        usage=usage,
        metadata={"cached": from_cache, "routed_model": model},
    )


def _error_response(req: ChatCompletionRequest, msg: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-err-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=req.model or "unknown",
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=f"ERROR: {msg}"),
            finish_reason="error",
        )],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        metadata={"error": msg},
    )


# ---- Admin Endpoints -------------------------------------------------------

@app.get("/admin/cache/stats")
async def cache_stats(request: Request) -> dict[str, Any]:
    cache: PromptCache = request.app.state.cache
    return await cache.stats()


@app.post("/admin/ab-test/toggle")
async def toggle_ab_test(request: Request, enabled: bool = True) -> dict[str, Any]:
    ab: ABTestRouter = request.app.state.ab_router
    ab.config.enabled = enabled
    return {"enabled": ab.config.enabled, "config": ab.config.__dict__}


@app.get("/admin/routing")
async def routing_table(request: Request) -> dict[str, str]:
    r: SmartRouter = request.app.state.router
    return {k.value: v for k, v in r.routing_table.items()}
