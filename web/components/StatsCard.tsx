import { clsx } from "clsx";

interface StatsCardProps {
  label: string;
  value: string | number;
  unit?: string;
  trend?: "up" | "down" | "stable";
  accentColor?: string;
}

export default function StatsCard({ label, value, unit, trend, accentColor = "primary" }: StatsCardProps) {
  return (
    <div className="bg-surface border border-border rounded-xl p-5">
      <p className="text-xs uppercase tracking-wider text-muted">{label}</p>
      <div className="mt-2 flex items-baseline gap-2">
        <span className={clsx("text-3xl font-bold font-mono", `text-${accentColor}`)}>{value}</span>
        {unit && <span className="text-sm text-muted">{unit}</span>}
      </div>
      {trend && (
        <p className="mt-2 text-xs text-muted">
          Trend:{" "}
          <span
            className={clsx(
              trend === "up" && "text-success",
              trend === "down" && "text-danger",
              trend === "stable" && "text-warning",
            )}
          >
            {trend}
          </span>
        </p>
      )}
    </div>
  );
}
