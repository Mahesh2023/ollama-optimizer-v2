"""Redis-backed prompt cache."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis


class PromptCache:
    """Exact-match prompt cache using Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = 3600) -> None:
        self.redis_url = redis_url
        self.ttl = ttl
        self._client: aioredis.Redis | None = None

    async def _conn(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _key(self, model: str, prompt: str) -> str:
        digest = hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()
        return f"ollama-opt:cache:{digest}"

    async def get(self, model: str, prompt: str) -> dict[str, Any] | None:
        r = await self._conn()
        raw = await r.get(self._key(model, prompt))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set(
        self, model: str, prompt: str, response: dict[str, Any], ttl: int | None = None
    ) -> None:
        r = await self._conn()
        await r.setex(
            self._key(model, prompt),
            ttl or self.ttl,
            json.dumps(response),
        )

    async def delete(self, model: str, prompt: str) -> None:
        r = await self._conn()
        await r.delete(self._key(model, prompt))

    async def stats(self) -> dict[str, Any]:
        r = await self._conn()
        info = await r.info("stats")
        return {
            "hits": int(info.get("keyspace_hits", 0)),
            "misses": int(info.get("keyspace_misses", 0)),
            "total_commands": int(info.get("total_commands_processed", 0)),
        }

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
