"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, BarChart3, FlaskConical, Database, Settings, Cpu } from "lucide-react";
import { clsx } from "clsx";

const LINKS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/benchmarks", label: "Benchmarks", icon: BarChart3 },
  { href: "/playground", label: "Playground", icon: FlaskConical },
  { href: "/models", label: "Models", icon: Database },
  { href: "/hardware", label: "Hardware", icon: Cpu },
  { href: "/settings", label: "Settings", icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-60 bg-surface border-r border-border flex flex-col">
      <div className="p-6 border-b border-border">
        <h1 className="text-xl font-bold text-primary">Ollama Optimizer</h1>
        <p className="text-xs text-muted mt-1">v0.1.0 · LLMOps</p>
      </div>
      <nav className="flex-1 p-4 space-y-1">
        {LINKS.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors",
                active ? "bg-primary/10 text-primary" : "text-muted hover:bg-border hover:text-white",
              )}
            >
              <Icon size={16} />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="p-4 border-t border-border text-xs text-muted">
        <p>Status: <span className="text-success">●</span> Online</p>
      </div>
    </aside>
  );
}
