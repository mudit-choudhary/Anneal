/** Shared formatting helpers, so dates and sizes read the same everywhere. */

export function dayLabel(iso?: string | null): string {
  if (!iso) return "Unknown date";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
  if (isNaN(d.getTime())) return "Unknown date";
  const today = new Date();
  const startOf = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((startOf(today) - startOf(d)) / 86400000);
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days} days ago`;
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

export function timeLabel(iso?: string | null): string {
  if (!iso) return "";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
  return isNaN(d.getTime()) ? "" : d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

export function duration(ms?: number | null): string {
  if (ms == null) return "";
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  return s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export function bytes(n?: number | null): string {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  const units = ["kB", "MB", "GB"];
  let v = n / 1024;
  for (const u of units) {
    if (v < 1024) return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${u}`;
    v /= 1024;
  }
  return `${v.toFixed(1)} TB`;
}

export function eta(seconds?: number | null): string {
  if (seconds == null) return "—";
  if (seconds < 90) return `~${Math.round(seconds)}s`;
  if (seconds < 5400) return `~${Math.round(seconds / 60)} min`;
  return `~${(seconds / 3600).toFixed(1)} h`;
}
