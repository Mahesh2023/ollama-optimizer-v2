"use client";

import { useEffect, useState } from "react";
import StatsCard from "@/components/StatsCard";
import { fetchMetrics, fetchSystem, fetchCacheStats, parsePromMetrics } from "@/lib/api";

export default function Dashboard() {
  const [metrics, setMetrics] = useState<Record<string, number>>({});
  const [system, setSystem] = useState<any>(null);
  const [cache, setCache] = useState<any>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [m, s, c] = await Promise.all([
          fetchMetrics(),
          fetchSystem(),
          fetchCacheStats().catch(() => ({ hits: 0, misses: 0 })),
        ]);
        setMetrics(parsePromMetrics(m));
        setSystem(s);
        setCache(c);
      } catch (e) {
        console.error(e);
      }
    };
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  const totalRequests = Object.entries(metrics)
    .filter(([k]) => k.startsWith("ollama_optimizer_requests_total"))
    .reduce((sum, [, v]) => sum + v, 0);

  const totalTokens = Object.entries(metrics)
    .filter(([k]) => k.startsWith("ollama_optimizer_tokens_total"))
    .reduce((sum, [, v]) => sum + v, 0);

  const cacheHitRate =
    cache && cache.hits + cache.misses > 0
      ? ((cache.hits / (cache.hits + cache.misses)) * 100).toFixed(1)
      : "0.0";

  return (
    <div className="space-y-8">
      <header>
        <h2 className="text-2xl font-bold">Dashboard</h2>
        <p className="text-muted">Real-time LLMOps metrics</p>
      </header>

      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard label="Total Requests" value={totalRequests.toLocaleString()} />
        <StatsCard label="Tokens Served" value={totalTokens.toLocaleString()} accentColor="accent" />
        <StatsCard label="Cache Hit Rate" value={cacheHitRate} unit="%" accentColor="success" />
        <StatsCard label="Cache Hits" value={cache?.hits?.toLocaleString() ?? "0"} accentColor="warning" />
      </section>

      {system && (
        <section className="bg-surface border border-border rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">Hardware</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-muted">Platform</p>
              <p className="font-mono">{system.platform}</p>
            </div>
            <div>
              <p className="text-muted">CPU</p>
              <p className="font-mono">
                {system.cpu_cores}C / {system.cpu_threads}T
              </p>
            </div>
            <div>
              <p className="text-muted">RAM</p>
              <p className="font-mono">{(system.ram_mb / 1024).toFixed(1)} GB</p>
            </div>
            <div>
              <p className="text-muted">GPUs / VRAM</p>
              <p className="font-mono">
                {system.gpus.length} / {(system.total_vram_mb / 1024).toFixed(1)} GB
              </p>
            </div>
          </div>
          {system.gpus.length > 0 && (
            <div className="mt-4">
              <h4 className="text-sm font-semibold text-muted mb-2">GPUs</h4>
              <ul className="space-y-1 font-mono text-sm">
                {system.gpus.map((g: any, i: number) => (
                  <li key={i}>
                    #{i} — {g.name} · {g.vram_mb} MB · {g.vendor}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
