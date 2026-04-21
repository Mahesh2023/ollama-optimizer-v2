"use client";

import { useState } from "react";
import { chatComplete } from "@/lib/api";

export default function Playground() {
  const [prompt, setPrompt] = useState("What is the capital of France?");
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const res = await chatComplete({
        messages: [{ role: "user", content: prompt }],
        user: "playground-user",
      });
      setResponse(res);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-2xl font-bold">Playground</h2>
        <p className="text-muted">Test smart routing + caching</p>
      </header>

      <div className="bg-surface border border-border rounded-xl p-6 space-y-4">
        <div>
          <label className="block text-sm text-muted mb-2">Prompt</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={6}
            className="w-full p-3 bg-bg border border-border rounded-lg font-mono text-sm focus:outline-none focus:border-primary"
          />
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading || !prompt.trim()}
          className="px-5 py-2 bg-primary text-bg font-semibold rounded-lg hover:bg-primary/90 disabled:opacity-50"
        >
          {loading ? "Generating..." : "Send"}
        </button>
      </div>

      {response && (
        <div className="bg-surface border border-border rounded-xl p-6 space-y-4">
          <div>
            <h3 className="text-sm text-muted">Routed Model</h3>
            <p className="font-mono text-primary">{response.model}</p>
          </div>

          <div>
            <h3 className="text-sm text-muted">Response</h3>
            <pre className="bg-bg p-4 rounded-lg whitespace-pre-wrap text-sm font-mono">
              {response.choices?.[0]?.message?.content}
            </pre>
          </div>

          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-muted">Prompt Tokens</p>
              <p className="font-mono">{response.usage?.prompt_tokens}</p>
            </div>
            <div>
              <p className="text-muted">Completion Tokens</p>
              <p className="font-mono">{response.usage?.completion_tokens}</p>
            </div>
            <div>
              <p className="text-muted">Cached</p>
              <p className="font-mono">{response.metadata?.cached ? "✔ HIT" : "✘ MISS"}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
