"use client";

import { useState, useMemo, useEffect } from "react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Legend, ReferenceLine, Cell,
} from "recharts";

type Curve = { threshold: number; hit_ratio: number; coverage: number };
type Point = { x: string; close: number; prob: number; pred: string; actual: string; correct: boolean };
type Track = {
  key: string; label: string; model: string; horizon: string;
  overall_hit_ratio: number; baseline: number; edge: number; n: number;
  confidence_curve: Curve[]; timeline: Point[];
};

const COLORS: Record<string, string> = { intraday: "#6366f1", weekly: "#10b981", monthly: "#f59e0b" };

export default function OurModels() {
  const [data, setData] = useState<Track[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sel, setSel] = useState<string>("intraday");

  useEffect(() => {
    fetch(`/data/our_models.json?t=${Date.now()}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((j) => setData(j.tracks))
      .catch((e) => setErr(String(e)));
  }, []);

  // bar chart: overall hit ratio vs baseline per track
  const barData = useMemo(
    () => (data ?? []).map((t) => ({
      name: t.label.split("—")[0].trim(), hit: t.overall_hit_ratio, baseline: t.baseline, key: t.key,
    })), [data]);

  // merged confidence curve (one line per track)
  const mergedCurve = useMemo(() => {
    if (!data) return [];
    const ths = Array.from(new Set(data.flatMap((t) => t.confidence_curve.map((c) => c.threshold)))).sort();
    return ths.map((th) => {
      const row: any = { threshold: th };
      data.forEach((t) => {
        const c = t.confidence_curve.find((x) => x.threshold === th);
        if (c) row[t.key] = c.hit_ratio;
      });
      return row;
    });
  }, [data]);

  const track = useMemo(() => (data ?? []).find((t) => t.key === sel), [data, sel]);

  // cumulative hit rate over time for the selected track
  const cumHit = useMemo(() => {
    if (!track) return [];
    let hits = 0;
    return track.timeline.map((p, i) => {
      hits += p.correct ? 1 : 0;
      return { x: p.x, close: p.close, hitRate: (hits / (i + 1)) * 100 };
    });
  }, [track]);

  if (err) return <Shell><p className="text-red-600">Failed to load our_models.json ({err}). Run export_our_results.py.</p></Shell>;
  if (!data) return <Shell><p className="text-zinc-500">Loading…</p></Shell>;

  return (
    <Shell>
      <div className="mb-6">
        <a href="/" className="text-sm text-indigo-600 hover:underline">← Back to dashboard</a>
        <h1 className="text-2xl font-bold mt-2">New NIFTY 50 Regime Models</h1>
        <p className="text-zinc-500 text-sm">Honest walk-forward · leak-free (shuffle-tested) · overfitting-checked</p>
      </div>

      {/* summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        {data.map((t) => (
          <div key={t.key} className="rounded-xl border border-zinc-200 bg-white p-5">
            <div className="text-xs font-semibold uppercase tracking-wide" style={{ color: COLORS[t.key] }}>
              {t.label}
            </div>
            <div className="text-4xl font-bold mt-2">{t.overall_hit_ratio}%</div>
            <div className="text-xs text-zinc-500 mt-1">hit ratio · baseline {t.baseline}% · edge +{t.edge}</div>
            <div className="text-xs text-zinc-400 mt-3">{t.model}</div>
            <div className="text-xs text-zinc-400">horizon: {t.horizon} · n={t.n}</div>
          </div>
        ))}
      </div>

      {/* hit ratio vs baseline */}
      <Panel title="Hit ratio vs baseline (per horizon)">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={barData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis dataKey="name" /><YAxis domain={[40, 80]} unit="%" />
            <Tooltip /><Legend />
            <Bar dataKey="baseline" name="Baseline" fill="#cbd5e1" />
            <Bar dataKey="hit" name="Model hit ratio">
              {barData.map((d) => <Cell key={d.key} fill={COLORS[d.key]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      {/* confidence curve */}
      <Panel title="Hit ratio rises with confidence (trade only confident calls)">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={mergedCurve}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis dataKey="threshold" type="number" domain={[0.5, 0.9]} tickFormatter={(v) => v.toFixed(2)} />
            <YAxis domain={[50, 90]} unit="%" />
            <Tooltip /><Legend />
            {data.map((t) => (
              <Line key={t.key} type="monotone" dataKey={t.key} name={t.label.split("—")[0].trim()}
                stroke={COLORS[t.key]} strokeWidth={2} dot connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Panel>

      {/* per-track timeline */}
      <Panel title="Prediction timeline & cumulative hit rate">
        <div className="flex gap-2 mb-3">
          {data.map((t) => (
            <button key={t.key} onClick={() => setSel(t.key)}
              className={`px-3 py-1 rounded-md text-xs font-semibold border ${sel === t.key ? "text-white" : "text-zinc-600 bg-white"}`}
              style={sel === t.key ? { background: COLORS[t.key], borderColor: COLORS[t.key] } : { borderColor: "#e4e4e7" }}>
              {t.label.split("—")[0].trim()}
            </button>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={cumHit} syncId="ourtl">
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis dataKey="x" hide /><YAxis yAxisId="p" hide domain={["auto", "auto"]} />
            <Tooltip />
            <Line yAxisId="p" type="monotone" dataKey="close" name="NIFTY close" stroke="#64748b" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={cumHit} syncId="ourtl">
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis dataKey="x" hide /><YAxis domain={[40, 90]} unit="%" />
            <Tooltip />
            <ReferenceLine y={track?.baseline} stroke="#cbd5e1" strokeDasharray="4 4" label="baseline" />
            <Line type="monotone" dataKey="hitRate" name="cumulative hit rate"
              stroke={COLORS[sel]} dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Panel>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="min-h-screen bg-[#fafafa] text-zinc-900 px-6 py-8 max-w-5xl mx-auto">{children}</div>;
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-5 mb-6">
      <div className="text-sm font-semibold mb-4">{title}</div>
      {children}
    </div>
  );
}
