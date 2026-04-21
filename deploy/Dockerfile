FROM python:3.12-slim AS python-builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY internal ./internal
COPY cmd ./cmd

RUN pip install --no-cache-dir --prefix=/install ".[llmops]"

FROM node:18 AS node-builder

WORKDIR /web
COPY web/package*.json ./
RUN npm install
COPY web/ ./
RUN npm run build

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-builder /install /usr/local
COPY --from=node-builder /web/.next /app/.next
COPY --from=node-builder /web/public /app/public
COPY --from=node-builder /web/package.json /app/package.json

WORKDIR /app
COPY internal ./internal
COPY cmd ./cmd

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["uvicorn", "cmd.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
