export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchHealth() {
  const r = await fetch(`${API_BASE}/health`);
  return r.json();
}

export async function fetchSystem() {
  const r = await fetch(`${API_BASE}/system`);
  return r.json();
}

export async function fetchMetrics() {
  const r = await fetch(`${API_BASE}/metrics`);
  return r.text();
}

export async function fetchCacheStats() {
  const r = await fetch(`${API_BASE}/admin/cache/stats`);
  return r.json();
}

export async function fetchRoutingTable() {
  const r = await fetch(`${API_BASE}/admin/routing`);
  return r.json();
}

export async function chatComplete(body: {
  messages: { role: string; content: string }[];
  model?: string;
  user?: string;
}) {
  const r = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

/** Parse Prometheus text exposition format into a dict of metric → value (for gauges/counters only). */
export function parsePromMetrics(text: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const line of text.split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const match = line.match(/^(\S+)\s+([\d.eE+-]+)$/);
    if (match) {
      out[match[1]] = parseFloat(match[2]);
    }
  }
  return out;
}
