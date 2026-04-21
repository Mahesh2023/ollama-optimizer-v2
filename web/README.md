# Web Dashboard

Next.js 14 + Tailwind CSS dashboard for Ollama Optimizer v2.

## Development

```bash
cd web
npm install
npm run dev
# Visit http://localhost:3000
```

The dashboard expects the API at `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

## Build for production

```bash
npm run build
npm start
```

## Deploy

- **Vercel** (recommended): connect repo, set `NEXT_PUBLIC_API_URL`
- **Cloudflare Pages**: build command `npm run build`
- **Render**: use static site service

## Pages

- `/` — Real-time dashboard (requests, tokens, cache hit rate, hardware)
- `/playground` — Test chat completions with live routing metadata
- `/benchmarks` — Benchmark comparisons (TODO)
- `/models` — Model registry (TODO)
- `/hardware` — Hardware deep-dive (TODO)
- `/settings` — Config editor (TODO)
