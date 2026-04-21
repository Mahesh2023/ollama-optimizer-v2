"""FastAPI server: OpenAI-compatible routing API with LLMOps instrumentation."""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Mount static files for Next.js frontend
static_dir = Path("/app/public")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


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
async def root() -> Response:
    """Serve dashboard HTML directly."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama Optimizer v2</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #0f172a;
            --gray: #64748b;
            --light: #f1f5f9;
            --white: #ffffff;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            color: var(--white);
            line-height: 1.6;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        header {
            text-align: center;
            margin-bottom: 3rem;
            animation: fadeInDown 0.8s ease-out;
        }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        header h1 {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }
        header p {
            font-size: 1.25rem;
            color: var(--gray);
            font-weight: 300;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--success);
            margin-top: 1rem;
        }
        .status-badge::before {
            content: '';
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .card {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: all 0.3s ease;
            animation: fadeInUp 0.8s ease-out;
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .card:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.5);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        .card:nth-child(2) { animation-delay: 0.1s; }
        .card:nth-child(3) { animation-delay: 0.2s; }
        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        .card-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }
        .card h2 {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--gray);
        }
        .card .value {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--white) 0%, var(--gray) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 1rem 0;
            letter-spacing: -0.02em;
        }
        .card .label {
            color: var(--gray);
            font-size: 0.875rem;
            font-weight: 400;
        }
        .playground {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            animation: fadeInUp 0.8s ease-out 0.3s backwards;
        }
        .playground h2 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        textarea {
            width: 100%;
            min-height: 120px;
            padding: 1rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            font-family: 'Inter', inherit;
            font-size: 1rem;
            color: var(--white);
            resize: vertical;
            transition: all 0.3s ease;
        }
        textarea:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }
        textarea::placeholder {
            color: var(--gray);
        }
        button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: var(--white);
            border: none;
            padding: 0.875rem 2rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
        }
        button:disabled {
            background: var(--gray);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .response {
            margin-top: 1.5rem;
            padding: 1.5rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 12px;
            min-height: 100px;
            max-height: 400px;
            overflow-y: auto;
        }
        .response pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            color: var(--white);
            font-family: 'Inter', monospace;
            font-size: 0.875rem;
            line-height: 1.6;
        }
        .endpoints {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 16px;
            padding: 2rem;
            margin-top: 2rem;
            animation: fadeInUp 0.8s ease-out 0.4s backwards;
        }
        .endpoints h2 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .endpoint {
            display: flex;
            align-items: center;
            padding: 1rem;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 10px;
            margin-bottom: 0.75rem;
            transition: all 0.3s ease;
        }
        .endpoint:hover {
            border-color: rgba(99, 102, 241, 0.5);
            background: rgba(15, 23, 42, 0.8);
        }
        .method {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: var(--white);
            padding: 0.375rem 0.875rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-right: 1rem;
            min-width: 60px;
            text-align: center;
        }
        .method.GET { background: linear-gradient(135deg, var(--success) 0%, #059669 100%); }
        .method.POST { background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%); }
        .path {
            font-family: 'Inter', monospace;
            color: var(--gray);
            font-size: 0.875rem;
        }
        @media (max-width: 768px) {
            header h1 { font-size: 2.5rem; }
            .grid { grid-template-columns: 1fr; }
            .container { padding: 1rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⚡ Ollama Optimizer v2</h1>
            <p>Production-grade LLMOps platform for local LLM inference</p>
            <div class="status-badge" id="status">
                <span>Connected</span>
            </div>
        </header>
        <div class="grid">
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">📊</div>
                    <h2>Total Requests</h2>
                </div>
                <div class="value" id="requests">0</div>
                <div class="label">API requests processed</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">⚡</div>
                    <h2>Avg Latency</h2>
                </div>
                <div class="value" id="latency">0ms</div>
                <div class="label">Average response time</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">💾</div>
                    <h2>Cache Hit Rate</h2>
                </div>
                <div class="value" id="cache">0%</div>
                <div class="label">Redis cache efficiency</div>
            </div>
        </div>
        <div class="playground">
            <h2>💬 Chat Playground</h2>
            <div class="chat-container">
                <textarea id="prompt" placeholder="Enter your prompt here..."></textarea>
                <button onclick="sendPrompt()" id="sendBtn">
                    <span>Send Message</span>
                    <span>→</span>
                </button>
            </div>
            <div class="response" id="response">
                <pre>Response will appear here...</pre>
            </div>
        </div>
        <div class="endpoints">
            <h2>📡 API Endpoints</h2>
            <div class="endpoint">
                <span class="method GET">GET</span>
                <span class="path">/health</span>
            </div>
            <div class="endpoint">
                <span class="method GET">GET</span>
                <span class="path">/system</span>
            </div>
            <div class="endpoint">
                <span class="method GET">GET</span>
                <span class="path">/metrics</span>
            </div>
            <div class="endpoint">
                <span class="method POST">POST</span>
                <span class="path">/v1/chat/completions</span>
            </div>
            <div class="endpoint">
                <span class="method GET">GET</span>
                <span class="path">/admin/cache/stats</span>
            </div>
            <div class="endpoint">
                <span class="method GET">GET</span>
                <span class="path">/admin/routing</span>
            </div>
        </div>
    </div>
    <script>
        async function sendPrompt() {
            const prompt = document.getElementById('prompt').value;
            const btn = document.getElementById('sendBtn');
            const responseDiv = document.getElementById('response');
            if (!prompt.trim()) return;
            btn.disabled = true;
            responseDiv.innerHTML = '<pre>Loading...</pre>';
            try {
                const res = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        messages: [{ role: 'user', content: prompt }],
                        stream: false
                    })
                });
                const text = await res.text();
                try {
                    const data = JSON.parse(text);
                    responseDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
                } catch {
                    responseDiv.innerHTML = `<pre>Status: ${res.status}\\nResponse:\\n${text}</pre>`;
                }
            } catch (error) {
                responseDiv.innerHTML = `<pre>Error: ${error.message}</pre>`;
            }
            btn.disabled = false;
        }
        async function updateMetrics() {
            try {
                const res = await fetch('/metrics');
                const text = await res.text();
                const lines = text.split('\\n');
                let requests = 0;
                let latency = 0;
                let cacheHits = 0;
                let cacheMisses = 0;
                lines.forEach(line => {
                    if (line.startsWith('hpc_requests_total')) {
                        requests = parseFloat(line.split(' ')[1]) || 0;
                    }
                    if (line.startsWith('hpc_latency_seconds') && !line.includes('quantile')) {
                        latency = (parseFloat(line.split(' ')[1]) * 1000).toFixed(0) || 0;
                    }
                    if (line.startsWith('hpc_cache_hits_total')) {
                        cacheHits = parseFloat(line.split(' ')[1]) || 0;
                    }
                    if (line.startsWith('hpc_cache_misses_total')) {
                        cacheMisses = parseFloat(line.split(' ')[1]) || 0;
                    }
                });
                document.getElementById('requests').textContent = requests;
                document.getElementById('latency').textContent = latency + 'ms';
                const total = cacheHits + cacheMisses;
                const rate = total > 0 ? ((cacheHits / total) * 100).toFixed(0) : 0;
                document.getElementById('cache').textContent = rate + '%';
            } catch (error) {
                console.error('Failed to fetch metrics:', error);
            }
        }
        setInterval(updateMetrics, 5000);
        updateMetrics();
    </script>
</body>
</html>"""
    return Response(content=html_content, media_type="text/html")


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
