"""Langfuse LLM observability integration."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    LANGFUSE_AVAILABLE = False
    Langfuse = None  # type: ignore


class LangfuseTracer:
    """Thin wrapper around Langfuse for LLM tracing."""

    def __init__(
        self,
        public_key: str = "",
        secret_key: str = "",
        host: str = "https://cloud.langfuse.com",
    ) -> None:
        self.enabled = LANGFUSE_AVAILABLE and bool(public_key and secret_key)
        if self.enabled:
            self.client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        else:
            self.client = None
            if not LANGFUSE_AVAILABLE:
                logger.info("Langfuse not installed; tracing disabled.")
            elif not (public_key and secret_key):
                logger.info("Langfuse keys not set; tracing disabled.")

    def trace_generation(
        self,
        name: str,
        model: str,
        prompt: str,
        output: str,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> str | None:
        """Record an LLM generation event. Returns trace_id."""
        if not self.enabled:
            return None
        try:
            trace = self.client.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
            trace.generation(
                name=f"{name}-gen",
                model=model,
                input=prompt,
                output=output,
                usage=usage or {},
                metadata=metadata or {},
            )
            return trace.id
        except Exception as e:  # noqa: BLE001
            logger.warning("Langfuse trace failed: %s", e)
            return None

    def score(self, trace_id: str, name: str, value: float, comment: str = "") -> None:
        """Attach a quality score to a trace."""
        if not self.enabled or not trace_id:
            return
        try:
            self.client.score(trace_id=trace_id, name=name, value=value, comment=comment)
        except Exception as e:  # noqa: BLE001
            logger.warning("Langfuse score failed: %s", e)

    def flush(self) -> None:
        if self.enabled and self.client is not None:
            self.client.flush()
