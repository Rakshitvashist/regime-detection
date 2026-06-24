"use client";

import { useState, useMemo, useEffect, type ReactNode, type ChangeEvent, type DragEvent } from "react";
import {
  ComposedChart,
  Line,
  Area,
  AreaChart,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
  CartesianGrid,
  Cell,
  LineChart,
  Bar,
  BarChart,
  Brush,
} from "recharts";

// --- Advanced Quant Icons ---
const ZapIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
  </svg>
);
const ActivityIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
  </svg>
);
const RadarIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="2" /><path d="M12 19a7 7 0 1 0-7-7" /><path d="M12 5a7 7 0 0 1 7 7" /><path d="M12 12l5-5" /></svg>;
const ShieldIcon = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>;

export default function Page() {
  const [history, setHistory] = useState<any[]>([]);
  const [index, setIndex] = useState<number>(0);
  const [view, setView] = useState<'dashboard' | 'analyzer' | 'forecast' | 'horizons' | 'model'>('dashboard');
  const [forecastData, setForecastData] = useState<any>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [primaryFile, setPrimaryFile] = useState<string>("NIFTY_500");
  const [confidenceFilter, setConfidenceFilter] = useState(0.5);
  const [isLoading, setIsLoading] = useState(false);
  const [modelMetrics, setModelMetrics] = useState<any>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 2 | 5 | 10 | 20>(1);
  const [isExpanded, setIsExpanded] = useState(false);

  // Instrument catalogue — in-sample + cross-instrument OOS
  const IN_SAMPLE_INSTRUMENTS = [
    { id: "NIFTY_500",    label: "NIFTY 500",    file: "regime_NIFTY_500.json" },
    { id: "NIFTY",        label: "NIFTY 50",     file: "regime_NIFTY.json" },
    { id: "BANKNIFTY",    label: "BANK NIFTY",   file: "regime_BANKNIFTY.json" },
  ];
  const OOS_INSTRUMENTS = [
    { id: "NIFTY_on_NIFTY500",      label: "NIFTY → NIFTY 500",     file: "regime_NIFTY_on_NIFTY500.json",      trainedOn: "NIFTY" },
    { id: "NIFTY_on_BANKNIFTY",     label: "NIFTY → BANK NIFTY",    file: "regime_NIFTY_on_BANKNIFTY.json",     trainedOn: "NIFTY" },
    { id: "NIFTY500_on_NIFTY",      label: "NIFTY 500 → NIFTY 50",  file: "regime_NIFTY500_on_NIFTY.json",      trainedOn: "NIFTY_500" },
    { id: "NIFTY500_on_BANKNIFTY",  label: "NIFTY 500 → BANK NIFTY",file: "regime_NIFTY500_on_BANKNIFTY.json",  trainedOn: "NIFTY_500" },
    { id: "BANKNIFTY_on_NIFTY",     label: "BANK NIFTY → NIFTY 50", file: "regime_BANKNIFTY_on_NIFTY.json",     trainedOn: "BANKNIFTY" },
    { id: "BANKNIFTY_on_NIFTY500",  label: "BANK NIFTY → NIFTY 500",file: "regime_BANKNIFTY_on_NIFTY500.json",  trainedOn: "BANKNIFTY" },
    { id: "NIFTY_on_CRUDE",         label: "NIFTY → Crude Oil",     file: "regime_NIFTY_on_CRUDE.json",         trainedOn: "NIFTY" },
    { id: "NIFTY_on_WTI",           label: "NIFTY → WTI",           file: "regime_NIFTY_on_WTI.json",           trainedOn: "NIFTY" },
    { id: "NIFTY_on_USDINR",        label: "NIFTY → USDINR",        file: "regime_NIFTY_on_USDINR.json",        trainedOn: "NIFTY" },
    { id: "NIFTY500_on_CRUDE",      label: "NIFTY 500 → Crude Oil", file: "regime_NIFTY500_on_CRUDE.json",      trainedOn: "NIFTY_500" },
    { id: "NIFTY500_on_WTI",        label: "NIFTY 500 → WTI",       file: "regime_NIFTY500_on_WTI.json",       trainedOn: "NIFTY_500" },
    { id: "NIFTY500_on_USDINR",     label: "NIFTY 500 → USDINR",    file: "regime_NIFTY500_on_USDINR.json",    trainedOn: "NIFTY_500" },
  ];
  const ALL_INSTRUMENTS = [...IN_SAMPLE_INSTRUMENTS, ...OOS_INSTRUMENTS];
  const isOOS = OOS_INSTRUMENTS.some(i => i.id === primaryFile);
  const playbackIntervalMs = Math.max(20, Math.round(220 / playbackSpeed));

  const fetchInstrument = async (symbol: string) => {
    setIsLoading(true);
    try {
      const entry = ALL_INSTRUMENTS.find(i => i.id === symbol);
      const file = entry?.file ?? `regime_${symbol}.json`;
      const response = await fetch(`data/${file}?t=${Date.now()}`);
      const parsed = await response.json();
      if (Array.isArray(parsed)) {
        setHistory(parsed);
        setIndex(parsed.length - 1);
        setPrimaryFile(symbol);
        
        try {
          const mSymbol = (entry as any)?.trainedOn ?? symbol;
          const metricResp = await fetch(`data/metrics_${mSymbol}.json?t=${Date.now()}`);
          if (metricResp.ok) {
            setModelMetrics(await metricResp.json());
          } else {
            setModelMetrics(null);
          }
        } catch {
          setModelMetrics(null);
        }

        // Multi-horizon forward forecast (in-sample symbols only; OOS series
        // have no forecast file, so this stays null and the tab shows a notice).
        try {
          const fSymbol = (entry as any)?.trainedOn ?? symbol;
          const fcResp = await fetch(`data/forecast_${fSymbol}.json?t=${Date.now()}`);
          setForecastData(fcResp.ok ? await fcResp.json() : null);
        } catch {
          setForecastData(null);
        }
      }
    } catch (err) {
      console.error("Failed to fetch instrument:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchInstrument("NIFTY_500");
  }, []);

  const lockedHistory = useMemo(() => {
    if (!history.length) return [];
    let currentLocked = history[0].regime;
    let hits = 0;
    let validCount = 0;
    return history.map((d) => {
      // Hysteresis lock
      if (d.regime !== currentLocked && (d.confidence || 0) >= 0.75) {
        currentLocked = d.regime;
      }
      
      // Compute cumulative running accuracy mapped to expected regime vs actual
      if (d.regime_actual && d.regime_actual !== "None") {
         validCount++;
         if (d.regime === d.regime_actual) hits++;
      }
      const hitRate = validCount > 0 ? (hits / validCount) * 100 : 0;
      
      return { 
        ...d, 
        locked_regime: currentLocked, 
        raw_regime: d.regime, 
        locked_confidence: d.confidence,
        accuracy: validCount > 0 ? hits / validCount : 0,
        hitRate: hitRate
      };
    });
  }, [history]);

  const currentSlice = useMemo(() => lockedHistory.slice(0, index + 1), [lockedHistory, index]);

  const minPrice = useMemo(() => {
    if (currentSlice.length === 0) return 0;
    return Math.min(...currentSlice.map(d => d.close)) * 0.98;
  }, [currentSlice]);
  const maxPrice = useMemo(() => {
    if (currentSlice.length === 0) return 100;
    return Math.max(...currentSlice.map(d => d.close)) * 1.02;
  }, [currentSlice]);

  const globalRegimeSpans = useMemo(() => {
    if (history.length === 0) return [];
    const spans = [];
    let start = history[0].date;
    let regime = history[0].regime;
    for (let i = 1; i < history.length; i++) {
        if (history[i].regime !== regime) {
            spans.push({ start, end: history[i].date, regime, startIndex: i - 1 });
            start = history[i].date;
            regime = history[i].regime;
        }
    }
    spans.push({ start, end: history[history.length - 1].date, regime, startIndex: history.length - 1 });
    return spans;
  }, [history]);

  const globalSuccessSpans = useMemo(() => {
    if (history.length === 0) return [];
    const spans = [];
    const getStatus = (d: any) => d.regime === d.regime_actual ? 'hit' : (d.regime_actual && d.regime_actual !== "None" ? 'miss' : 'neutral');
    let start = history[0].date;
    let status = getStatus(history[0]);
    for (let i = 1; i < history.length; i++) {
        const s = getStatus(history[i]);
        if (s !== status) {
            spans.push({ start, end: history[i].date, status, startIndex: i - 1 });
            start = history[i].date;
            status = s;
        }
    }
    spans.push({ start, end: history[history.length - 1].date, status, startIndex: history.length - 1 });
    return spans;
  }, [history]);

  const regimeSpans = useMemo(() => {
    return globalRegimeSpans.filter(s => {
        const foundIndex = history.findIndex(h => h.date === s.start);
        return foundIndex <= index;
    }).map(s => {
        const endIndex = history.findIndex(h => h.date === s.end);
        return { ...s, end: endIndex > index ? history[index].date : s.end };
    });
  }, [globalRegimeSpans, index, history]);

  const successSpans = useMemo(() => {
    return globalSuccessSpans.filter(s => {
        const foundIndex = history.findIndex(h => h.date === s.start);
        return foundIndex <= index;
    }).map(s => {
        const endIndex = history.findIndex(h => h.date === s.end);
        return { ...s, end: endIndex > index ? history[index].date : s.end };
    });
  }, [globalSuccessSpans, index, history]);

  const loadJsonFile = (file: File) => {
    setPrimaryFile(file.name);
    const reader = new FileReader();
    reader.onload = (event: ProgressEvent<FileReader>) => {
      try {
        const text = event.target?.result;
        if (typeof text !== "string") return;
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed)) {
          setHistory(parsed);
          setIndex(parsed.length - 1);
        }
      } catch {
        alert("Invalid JSON data format.");
      }
    };
    reader.readAsText(file);
  };

  const handleUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    loadJsonFile(file);
  };

  const handleDropJson = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) loadJsonFile(file);
  };

  const playForward = () => setIsPlaying(true);
  const pause = () => setIsPlaying(false);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isPlaying && index < history.length - 1) {
      interval = setInterval(() => setIndex((prev) => prev + 1), playbackIntervalMs);
    } else {
      setIsPlaying(false);
    }
    return () => clearInterval(interval);
  }, [isPlaying, index, history.length, playbackIntervalMs]);

  const getStats = (dataList: any[]) => {
    if (!dataList || dataList.length === 0) return null;
    const validMatches = dataList.filter(d => d.regime_actual !== "None");
    const filteredMatches = validMatches.filter(d => (d.confidence || 0) >= confidenceFilter);
    if (filteredMatches.length === 0) return null;

    const hitResults = filteredMatches.map(d => d.regime === d.regime_actual);
    const globalAccuracy = (hitResults.filter(Boolean).length / hitResults.length) * 100;

    const regimes = ['bull', 'sideways', 'bear'];
    const confusion: any = {};
    regimes.forEach(p => {
       confusion[p] = {};
       regimes.forEach(a => {
          confusion[p][a] = filteredMatches.filter(d => d.regime === p && d.regime_actual === a).length;
       });
    });

    const yearlyMap: any = {};
    dataList.forEach(d => {
      if (!d.date || typeof d.date !== 'string' || !d.date.includes('-')) return;
      const parts = d.date.split('-');
      const year = parts[0];
      const month = parseInt(parts[1]) - 1;
      if (isNaN(month) || month < 0 || month > 11) return;
      
      if (!yearlyMap[year]) yearlyMap[year] = Array(12).fill(null).map(() => ({ hits: 0, total: 0 }));
      if (d.regime_actual !== "None") {
        yearlyMap[year][month].total++;
        if (d.regime === d.regime_actual) yearlyMap[year][month].hits++;
      }
    });

    const yearlyPerformance = Object.keys(yearlyMap).sort().reverse().map(year => {
      const monthStats = yearlyMap[year].map((m: any) => m.total > 0 ? (m.hits / m.total) * 100 : null);
      const totalYearHits = yearlyMap[year].reduce((acc: number, curr: any) => acc + curr.hits, 0);
      const totalYearCount = yearlyMap[year].reduce((acc: number, curr: any) => acc + curr.total, 0);
      return {
        year,
        months: monthStats,
        yearlyAvg: totalYearCount > 0 ? (totalYearHits / totalYearCount) * 100 : null
      };
    });

    return { 
      accuracy: globalAccuracy, 
      avgConfidence: filteredMatches.reduce((acc, curr) => acc + (curr.confidence || 0), 0) / filteredMatches.length,
      yearlyPerformance,
      confusion,
      bullPrecision: (filteredMatches.filter(d => d.regime === 'bull' && d.regime_actual === 'bull').length / filteredMatches.filter(d => d.regime === 'bull').length) * 100 || 0,
      bearPrecision: (filteredMatches.filter(d => d.regime === 'bear' && d.regime_actual === 'bear').length / filteredMatches.filter(d => d.regime === 'bear').length) * 100 || 0,
    };
  };

  const primaryStats = useMemo(() => getStats(history), [history, confidenceFilter]);

  const data = lockedHistory[index] || null;

  const stability = useMemo(() => {
    if (lockedHistory.length < 5 || index < 5) return 100;
    const last5 = lockedHistory.slice(index - 4, index + 1);
    const currentLocked = lockedHistory[index]?.locked_regime;
    const matches = last5.filter(d => d.locked_regime === currentLocked).length;
    return (matches / 5) * 100;
  }, [lockedHistory, index]);

  const viewTabs: { id: typeof view; label: string }[] = [
    { id: "dashboard", label: "Dashboard" },
    { id: "analyzer", label: "Performance" },
    { id: "forecast", label: "Checklist" },
    { id: "horizons", label: "Forecast" },
    { id: "model", label: "Model Visualisation" }
  ];

  const statRibbonItems = useMemo(() => {
    if (!data) return [];
    const acc = primaryStats?.accuracy ?? 0;
    return [
      { label: "Regime", value: String(data.locked_regime ?? "—"), hint: "Locked signal" },
      { label: "Model", value: String(data.raw_regime ?? "—"), hint: "Raw output" },
      { label: "Confidence", value: `${(data.confidence * 100).toFixed(1)}%`, hint: "At this bar" },
      { label: "Stability", value: `${stability.toFixed(0)}%`, hint: "5-bar lock" },
      { label: "RSI (14)", value: data.rsi_14?.toFixed(1) ?? "—", hint: undefined },
      { label: "Hit rate", value: `${acc.toFixed(1)}%`, hint: "Full sample" },
    ];
  }, [data, primaryStats, stability]);

  return (
    <div className="flex h-screen bg-[#f4f4f5] text-zinc-900 font-mono selection:bg-emerald-500/25 overflow-hidden print:bg-white print:p-0 print:h-auto">
      
      {/* Sidebar — instruments & upload (Prosperity-style) */}
      {!isExpanded && (
        <aside className="w-[300px] border-r border-zinc-200/90 bg-white flex flex-col p-6 shadow-[0_1px_0_rgba(0,0,0,0.03)] shrink-0 overflow-y-auto print:hidden">
          <div className="flex items-center gap-3 mb-10 px-1">
          <div className="w-11 h-11 bg-gradient-to-br from-zinc-800 to-zinc-600 rounded-xl flex items-center justify-center text-white font-semibold text-lg shadow-md">R</div>
          <div className="flex flex-col min-w-0">
            <span className="text-[15px] font-semibold tracking-tight text-zinc-900 truncate">Regime Visualizer</span>
            <span className="text-[11px] text-zinc-500">Market regime audit</span>
          </div>
        </div>

        <a
          href="/our-models"
          className="mb-6 block rounded-lg bg-indigo-600 text-white text-center text-[12px] font-semibold py-2.5 hover:bg-indigo-700 transition-colors print:hidden"
        >
          ★ New Models — Intraday / Weekly / Monthly
        </a>

        <div className="flex flex-col gap-6 flex-1">

          {/* In-sample instruments */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 px-1">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">In-Sample</span>
            </div>
            <div className="flex flex-col gap-1.5">
              {IN_SAMPLE_INSTRUMENTS.map((inst) => (
                <button
                  key={inst.id}
                  type="button"
                  onClick={() => fetchInstrument(inst.id)}
                  className={`px-4 py-2.5 rounded-xl text-left text-[12px] font-medium border transition-all flex justify-between items-center gap-2 ${
                    primaryFile === inst.id
                      ? "bg-zinc-900 text-white border-zinc-900 shadow-sm"
                      : "bg-zinc-50/80 text-zinc-600 border-zinc-200/60 hover:bg-zinc-100 hover:border-zinc-300"
                  }`}
                >
                  <span className="truncate">{inst.label}</span>
                  {primaryFile === inst.id ? <ActivityIcon /> : null}
                </button>
              ))}
            </div>
          </div>

          {/* Cross-instrument OOS */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 px-1">
              <div className="w-1.5 h-1.5 rounded-full bg-amber-400" />
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Cross-Instrument OOS</span>
            </div>
            <p className="text-[9px] text-zinc-400 px-1 leading-snug -mt-1">Model trained on A, applied to B — true generalization test</p>
            <div className="flex flex-col gap-1.5">
              {OOS_INSTRUMENTS.map((inst) => (
                <button
                  key={inst.id}
                  type="button"
                  onClick={() => fetchInstrument(inst.id)}
                  className={`px-3 py-2.5 rounded-xl text-left text-[11px] font-medium border transition-all ${
                    primaryFile === inst.id
                      ? "bg-amber-500 text-white border-amber-500 shadow-sm"
                      : "bg-amber-50/60 text-zinc-600 border-amber-200/60 hover:bg-amber-100/60"
                  }`}
                >
                  <div className="font-semibold truncate">{inst.label}</div>
                  <div className={`text-[9px] mt-0.5 ${primaryFile === inst.id ? 'text-amber-100' : 'text-zinc-400'}`}>
                    model: {inst.trainedOn}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Upload */}
          <div className="space-y-2">
            <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest px-1 block">Upload</span>
            <label className="group cursor-pointer block">
              <input type="file" accept=".json,application/json" className="hidden" onChange={(e) => handleUpload(e)} />
              <div className="bg-zinc-50 hover:bg-zinc-100/80 border border-dashed border-zinc-200 text-zinc-500 p-3 rounded-xl flex items-center justify-between gap-2 transition-colors">
                <span className="text-[11px] font-medium truncate flex-1">{primaryFile || "Choose JSON…"}</span>
                <ShieldIcon />
              </div>
            </label>
          </div>

          <div className="space-y-2 pt-4 border-t border-zinc-100">
             <div className={`rounded-xl border px-4 py-3 ${isOOS ? 'border-amber-200/60 bg-amber-50/40' : 'border-emerald-200/60 bg-emerald-50/40'}`}>
                <div className="flex items-center gap-2 mb-0.5">
                  <span className={`text-[10px] font-bold uppercase tracking-wide ${isOOS ? 'text-amber-700' : 'text-emerald-700'}`}>
                    {isOOS ? 'OOS Generalization' : 'In-Sample'}
                  </span>
                  {isOOS && (
                    <span className="text-[8px] font-black bg-amber-400 text-white px-1.5 py-0.5 rounded-md uppercase tracking-wider">
                      CROSS-INSTR
                    </span>
                  )}
                </div>
                <p className={`text-[12px] font-semibold mt-0.5 truncate ${isOOS ? 'text-amber-900' : 'text-emerald-900'}`}>
                  {ALL_INSTRUMENTS.find(i => i.id === primaryFile)?.label ?? primaryFile}
                </p>
                {isOOS && (() => {
                  const entry = OOS_INSTRUMENTS.find(i => i.id === primaryFile);
                  return entry ? (
                    <p className="text-[9px] text-amber-600 mt-0.5">model: {entry.trainedOn}</p>
                  ) : null;
                })()}
                {isLoading ? <p className={`text-[10px] mt-1 ${isOOS ? 'text-amber-600/80' : 'text-emerald-600/80'}`}>Loading…</p> : null}
             </div>
          </div>
        </div>

        <div className="mt-auto pt-6 flex items-center gap-3 border-t border-zinc-100">
           <div className="w-9 h-9 rounded-full bg-zinc-100 border border-zinc-200/80 flex items-center justify-center text-zinc-700 font-semibold text-[11px]">N</div>
           <div className="flex flex-col min-w-0">
              <span className="text-[12px] font-semibold text-zinc-800 truncate">@Nabichan</span>
              <span className="text-[10px] text-zinc-500">Research</span>
           </div>
        </div>
      </aside>
      )}
      <main className="flex-1 flex flex-col overflow-hidden relative bg-[#fafafa] min-w-0">
        
        <header className="border-b border-zinc-200/80 bg-white/90 backdrop-blur-md shrink-0 z-50 print:bg-white">
          <div className="flex flex-wrap items-center justify-between gap-4 px-8 py-4">
            <div className="flex flex-col min-w-0">
              <h1 className="text-lg font-semibold tracking-tight text-zinc-900 truncate">
                {data?.symbol ? data.symbol.replace(/_/g, " ") : "Regime Visualizer"}
              </h1>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-1 text-[11px] text-zinc-500">
                <span>{data?.date ?? "Select or upload a series"}</span>
                {data ? (
                  <span className="inline-flex items-center rounded-full bg-emerald-50 border border-emerald-100 px-2 py-0.5 text-[10px] font-medium text-emerald-800">
                    Live audit
                  </span>
                ) : null}
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3 print:hidden">
              <div className="inline-flex rounded-full border border-zinc-200 bg-zinc-50/90 p-0.5 shadow-sm">
                {viewTabs.map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => {
                      setView(t.id);
                      if (t.id === "forecast") setIndex(history.length - 1);
                    }}
                    className={`px-4 py-1.5 rounded-full text-[11px] font-semibold transition-all ${
                      view === t.id ? "bg-white text-zinc-900 shadow-sm" : "text-zinc-500 hover:text-zinc-800"
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
              
              {/* Terminal Mode Toggle */}
              <button
                type="button"
                onClick={() => setIsExpanded(!isExpanded)}
                className={`px-4 py-1.5 rounded-xl text-[11px] font-bold transition-all flex items-center gap-2 border ${
                  isExpanded 
                    ? "bg-emerald-500 text-white border-emerald-500 shadow-lg shadow-emerald-500/20" 
                    : "bg-white text-zinc-600 border-zinc-200 hover:border-zinc-300"
                }`}
              >
                <ZapIcon />
                {isExpanded ? "TERMINAL_FOCUS" : "EXPAND_VIEW"}
              </button>

              <button
                type="button"
                onClick={() => window.print()}
                className="px-4 py-2 bg-zinc-900 text-white rounded-xl text-[11px] font-semibold hover:bg-zinc-800 transition-colors shadow-sm active:scale-[0.98]"
              >
                Export PDF
              </button>
            </div>
          </div>

          {data && statRibbonItems.length > 0 ? (
            <div className="px-8 pb-4 print:hidden">
              <StatRibbon items={statRibbonItems} />
            </div>
          ) : null}
        </header>

        {history.length > 0 ? (
          <div
            className="flex-1 flex flex-col bg-[#fafafa] overflow-hidden px-6 sm:px-10 py-6 sm:py-8 gap-6 print:p-0 print:gap-4 print:overflow-visible"
            onDragOver={(e) => {
              e.preventDefault();
              e.dataTransfer.dropEffect = "copy";
            }}
            onDrop={handleDropJson}
          >
            {/* Timeline — play, speed, scrub (Prosperity-style) */}
            <div className="bg-white rounded-2xl p-4 sm:p-5 border border-zinc-200/90 flex flex-col gap-4 shadow-[0_1px_2px_rgba(0,0,0,0.04)] shrink-0 print:hidden">
               <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                     <button
                        type="button"
                        onClick={isPlaying ? pause : playForward}
                        className="w-12 h-12 bg-zinc-900 text-white rounded-xl flex items-center justify-center transition-transform hover:scale-[1.02] active:scale-95 shadow-md text-lg leading-none"
                        aria-label={isPlaying ? "Pause" : "Play"}
                     >
                        {isPlaying ? "⏸" : "▶"}
                     </button>
                     <div className="text-[11px] text-zinc-500">
                        <span className="font-semibold text-zinc-800">Bar {index}</span>
                        <span className="text-zinc-400"> / {history.length - 1}</span>
                        {data?.date ? (
                           <span className="block text-[10px] text-zinc-400 mt-0.5 tabular-nums">TS: {data.date}</span>
                        ) : null}
                     </div>
                  </div>
                  <div className="flex items-center gap-1 rounded-full border border-zinc-200 bg-zinc-50/90 p-0.5">
                     {([1, 2, 5, 10, 20] as const).map((s) => (
                        <button
                           key={s}
                           type="button"
                           onClick={() => setPlaybackSpeed(s)}
                           className={`px-2.5 py-1 rounded-full text-[10px] font-semibold transition-colors ${
                              playbackSpeed === s ? "bg-white text-zinc-900 shadow-sm" : "text-zinc-500 hover:text-zinc-800"
                           }`}
                        >
                           {s}x
                        </button>
                     ))}
                  </div>
               </div>
               <div className="flex flex-col gap-2">
                  <input
                     type="range"
                     min={0}
                     max={history.length - 1}
                     value={index}
                     onChange={(e) => setIndex(Number(e.target.value))}
                     className="w-full accent-zinc-800 h-1.5 bg-zinc-200 rounded-full appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-zinc-400">
                     <span>
                        {history.length} bars · {primaryFile.replace(/_/g, " ")}
                     </span>
                     <span className="tabular-nums">{playbackSpeed}x playback</span>
                  </div>
               </div>
            </div>

            <div className="flex-1 overflow-y-auto no-scrollbar">
               {view === 'dashboard' && (
                <div className="flex-1 flex flex-col gap-6 overflow-y-auto no-scrollbar pb-20">
                   <div className="flex flex-col gap-6">
                      <ProsperityPanel
                        title={`Price & regime: ${(data?.symbol ?? primaryFile).replace(/_/g, " ")}`}
                        subtitle="Close with regime shading and validation ribbon"
                        right={
                          stability >= 100 ? (
                            <span className="shrink-0 rounded-full bg-emerald-500 px-3 py-1 text-[10px] font-semibold text-white shadow-sm">
                              Stable lock
                            </span>
                          ) : null
                        }
                        chartHeightClass={isExpanded ? "h-[750px]" : "h-[380px] sm:h-[420px]"}
                      >
                         <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                            <ComposedChart data={currentSlice} syncId="regime_audit" key={`dash-${primaryFile}`}>
                               <defs>
                                  <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                                     <stop offset="5%" stopColor="#10b981" stopOpacity={0.1}/>
                                     <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                                  </linearGradient>
                               </defs>
                               <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e4e4e7" opacity={0.5} />
                               <XAxis dataKey="date" hide />
                               <YAxis domain={['auto', 'auto']} orientation="right" tick={{fontSize: 10, fill: '#a1a1aa'}} axisLine={false} tickLine={false} />
                               <Tooltip 
                                 content={<CustomHud data={data} history={history} />} 
                                 cursor={{ stroke: '#e4e4e7', strokeWidth: 1 }}
                                 isAnimationActive={false}
                                 allowEscapeViewBox={{ x: false, y: true }}
                                 position={{ y: 0 }}
                               />
                               {regimeSpans.filter(s => s.regime !== 'sideways').map((span, i) => (
                                  <ReferenceArea key={`regime-${i}`} x1={span.start} x2={span.end} fill={span.regime === 'bull' ? '#10b981' : '#f43f5e'} fillOpacity={0.06} />
                               ))}
                               <Area type="monotone" dataKey="close" stroke="#10b981" strokeWidth={2} fill="url(#colorClose)" isAnimationActive={false} dot={false} />
                               {successSpans.filter(s => s.status !== 'neutral').map((span, i) => (
                                 <ReferenceArea 
                                   key={`success-${i}`} 
                                   x1={span.start} 
                                   x2={span.end} 
                                   y1={minPrice - (maxPrice - minPrice) * 0.15} 
                                   y2={minPrice - (maxPrice - minPrice) * 0.10} 
                                   fill={span.status === 'hit' ? '#10b981' : '#f43f5e'}
                                   fillOpacity={0.6}
                                   className="transition-all duration-300"
                                 />
                               ))}
                                <Brush 
                                    dataKey="date" 
                                    height={30} 
                                    stroke="#27272a" 
                                    fill="#fafafa" 
                                    tickFormatter={(val) => val} 
                                />
                             </ComposedChart>
                         </ResponsiveContainer>
                      </ProsperityPanel>

                      <ProsperityPanel
                        title="Momentum"
                        subtitle={`21d vs 63d returns · 21d: ${((data?.ret_21d ?? 0) * 100).toFixed(1)}%`}
                        chartHeightClass="h-[220px] sm:h-[240px]"
                      >
                         <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                            <LineChart data={currentSlice} syncId="regime_audit" key={`dash-mom-${primaryFile}`}>
                               <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f1f1" />
                               <XAxis dataKey="date" hide />
                               <YAxis domain={[-0.2, 0.2]} orientation="right" tick={{fontSize: 9, fill: '#a1a1aa'}} axisLine={false} tickLine={false} />
                               <ReferenceLine y={0.08} stroke="#10b981" strokeDasharray="3 3" opacity={0.3} label={{ value: 'Bull', position: 'right', fill: '#10b981', fontSize: 8 }} />
                               <ReferenceLine y={-0.08} stroke="#f43f5e" strokeDasharray="3 3" opacity={0.3} label={{ value: 'Bear', position: 'right', fill: '#f43f5e', fontSize: 8 }} />
                               <Line type="monotone" dataKey="ret_21d" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                               <Line type="monotone" dataKey="ret_63d" stroke="#10b981" strokeWidth={1} dot={false} opacity={0.3} isAnimationActive={false} />
                            </LineChart>
                         </ResponsiveContainer>
                      </ProsperityPanel>

                      <ProsperityPanel
                        title="Volatility & tail risk"
                        subtitle={`Vol ratio vs 63d · skew (21d): ${data?.skew_21d != null ? data.skew_21d.toFixed(2) : "—"}`}
                        chartHeightClass="h-[220px] sm:h-[240px]"
                      >
                         <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                            <AreaChart data={currentSlice} syncId="regime_audit" key={`dash-vol-${primaryFile}`}>
                               <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f1f1" />
                               <XAxis dataKey="date" hide />
                               <YAxis domain={[0, 2]} orientation="right" tick={{fontSize: 9, fill: '#a1a1aa'}} axisLine={false} />
                               <ReferenceLine y={1.3} stroke="#f43f5e" strokeDasharray="3 3" label={{ value: 'Panic', position: 'right', fill: '#f43f5e', fontSize: 8 }} />
                               <Area type="monotone" dataKey="vol_ratio" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.05} dot={false} isAnimationActive={false} />
                               <Line type="monotone" dataKey="skew_21d" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} />
                            </AreaChart>
                         </ResponsiveContainer>
                      </ProsperityPanel>
                      
                      <ProsperityPanel
                        title="Oscillator"
                        subtitle={`RSI (14): ${data?.rsi_14 != null ? Math.round(data.rsi_14) : "—"} · MACD histogram`}
                        chartHeightClass="h-[220px] sm:h-[240px]"
                      >
                         <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                            <ComposedChart data={currentSlice} syncId="regime_audit" key={`dash-osc-${primaryFile}`}>
                               <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f1f1" />
                               <XAxis dataKey="date" hide />
                               <YAxis domain={[0, 100]} orientation="right" tick={{fontSize: 9, fill: '#a1a1aa'}} axisLine={false} />
                               <ReferenceLine y={70} stroke="#f43f5e" strokeDasharray="3 3" opacity={0.2} />
                               <ReferenceLine y={30} stroke="#059669" strokeDasharray="3 3" opacity={0.2} />
                               <Area type="monotone" dataKey="rsi_14" stroke="#059669" fill="#059669" fillOpacity={0.05} isAnimationActive={false} />
                               <Bar dataKey="macd_h" isAnimationActive={false}>
                                  {currentSlice.map((entry, i) => (
                                     <Cell key={`cell-${i}`} fill={entry.macd_h > 0 ? '#10b981' : '#f43f5e'} fillOpacity={0.4} />
                                  ))}
                               </Bar>
                            </ComposedChart>
                         </ResponsiveContainer>
                      </ProsperityPanel>
                   </div>
                </div>
              )}

              {view === 'analyzer' && primaryStats && (
                <div className="flex flex-col py-2 space-y-12 pb-20">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                     <StatCard label="Hit rate" value={`${primaryStats.accuracy.toFixed(1)}%`} sub="Rolling backtest" />
                     <StatCard label="Bull precision" value={`${primaryStats.bullPrecision.toFixed(1)}%`} sub="When model says bull" />
                     <StatCard label="Bear precision" value={`${primaryStats.bearPrecision.toFixed(1)}%`} sub="When model says bear" />
                  </div>

                  <div className="bg-white rounded-2xl p-8 sm:p-10 border border-zinc-200/90 shadow-[0_1px_2px_rgba(0,0,0,0.04)] relative overflow-hidden">
                     <ModuleHeader title="Cumulative hit rate vs price" rightLabel="Historical (sample)" />
                     <div className="h-[350px] mt-6">
                        <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                          <ComposedChart data={currentSlice} syncId="regime_audit" key={`audit-${primaryFile}`}>
                             <defs>
                                <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                                   <stop offset="5%" stopColor="#10b981" stopOpacity={0.2}/>
                                   <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                                </linearGradient>
                             </defs>
                             <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f1f1" />
                             <XAxis dataKey="date" minTickGap={50} tick={{fontSize: 9, fill: '#a1a1aa'}} axisLine={false} tickLine={false} type="category" />
                             <YAxis yAxisId="left" domain={[20, 100]} orientation="left" tick={{fontSize: 9, fill: '#a1a1aa'}} tickFormatter={(val) => `${val}%`} axisLine={false} tickLine={false} />
                             <YAxis yAxisId="right" domain={['auto', 'auto']} orientation="right" tick={{fontSize: 9, fill: '#e4e4e7'}} axisLine={false} tickLine={false} />
                             <Tooltip 
                                contentStyle={{ borderRadius: '24px', border: '1px solid #e4e4e7', fontSize: '10px', fontWeight: 'bold', backgroundColor: '#ffffff', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} 
                             />
                             <ReferenceLine yAxisId="left" y={50} stroke="#f43f5e" strokeDasharray="3 3" opacity={0.3} label={{ value: 'RANDOM_CHANCE_(50%)', position: 'insideTopLeft', fill: '#f43f5e', fontSize: 8 }} />
                             <ReferenceLine yAxisId="left" y={65} stroke="#fbbf24" strokeDasharray="3 3" opacity={0.3} label={{ value: 'INSTITUTIONAL_TARGET_(65%)', position: 'insideTopLeft', fill: '#fbbf24', fontSize: 8 }} />
                             
                             {/* TIME-TRAVEL TRACKER */}
                             <ReferenceLine key={`marker-${index}`} x={history[index]?.date} stroke="#1a1a1a" strokeWidth={2} label={{ value: 'YOU_ARE_HERE', fill: '#1a1a1a', fontSize: 9, fontWeight: 'bold', position: 'top' }} />
                             <ReferenceArea key={`fill-${index}`} x1={history[0]?.date} x2={history[index]?.date} fill="#10b981" fillOpacity={0.05} />

                             <Area yAxisId="left" type="monotone" dataKey="hitRate" name="Hit_Rate" stroke="#10b981" strokeWidth={3} fill="url(#colorAcc)" isAnimationActive={false} />
                             <Line yAxisId="right" type="monotone" dataKey="close" name="Nifty_500" stroke="#a1a1aa" strokeWidth={1} dot={false} opacity={0.4} isAnimationActive={false} />
                          </ComposedChart>
                        </ResponsiveContainer>
                     </div>
                  </div>

                  <div className="bg-white rounded-2xl p-8 sm:p-10 border border-zinc-200/90 shadow-[0_1px_2px_rgba(0,0,0,0.04)] relative overflow-hidden">
                     <ModuleHeader title="Yearly hit rate heatmap" rightLabel="Grid view" />
                     <YearlyPerformanceTable data={primaryStats.yearlyPerformance} />
                  </div>

                  <div className="bg-white rounded-2xl p-8 sm:p-10 border border-zinc-200/90 shadow-[0_1px_2px_rgba(0,0,0,0.04)] relative overflow-hidden">
                     <ModuleHeader title="Confusion matrix" rightLabel="Regime reliability" />
                     <div className="grid grid-cols-4 gap-6 mt-10">
                        <div className="col-span-1" />
                        {['Actual_Bear', 'Actual_Side', 'Actual_Bull'].map(h => <div key={h} className="text-center text-[10px] font-black text-zinc-400 uppercase tracking-widest">{h}</div>)}
                        {['bear', 'sideways', 'bull'].map(p => (
                           <>
                              <div className="flex items-center text-[10px] font-black text-zinc-500 uppercase">Pred_{p}</div>
                              {['bear', 'sideways', 'bull'].map(a => {
                                 const count = primaryStats.confusion?.[p]?.[a] || 0;
                                 const isDiagonal = p === a;
                                 return (
                                    <div key={`${p}-${a}`} className={`h-24 flex flex-col items-center justify-center rounded-3xl border transition-all ${isDiagonal ? 'bg-emerald-50 border-emerald-100 shadow-sm' : 'bg-zinc-50 border-zinc-100'}`}>
                                       <span className={`text-2xl font-black ${isDiagonal ? 'text-emerald-700' : 'text-zinc-400'}`}>{count}</span>
                                       <span className="text-[9px] font-bold text-zinc-300 mt-1 uppercase tracking-tighter">Count</span>
                                    </div>
                                 );
                              })}
                           </>
                        ))}
                     </div>
                  </div>
                  
                  {/* SENSITIVITY AUDIT */}
                  <div className="bg-white rounded-2xl p-8 sm:p-10 border border-zinc-200/90 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
                     <ModuleHeader title="Confidence filter" rightLabel={`${(confidenceFilter * 100).toFixed(0)}% minimum`} />
                     <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-8 mt-6">
                        <div className="flex-1">
                           <input type="range" min="0.30" max="0.95" step="0.05" value={confidenceFilter} onChange={(e) => setConfidenceFilter(parseFloat(e.target.value))} className="w-full accent-zinc-800 h-1.5 bg-zinc-200 rounded-full appearance-none cursor-pointer" />
                           <div className="flex justify-between mt-3 text-[10px] font-medium text-zinc-500">
                              <span>Include weak signals</span>
                              <span>Strict only</span>
                           </div>
                        </div>
                        <div className="w-full sm:w-48 bg-zinc-50 p-6 rounded-2xl border border-zinc-200/80 flex flex-col items-center">
                           <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wide mb-1">Bars counted</span>
                           <span className="text-3xl font-semibold text-zinc-900 tabular-nums">{history.filter(d => d.confidence >= confidenceFilter && d.regime_actual !== "None").length}</span>
                        </div>
                     </div>
                  </div>
                </div>
              )}

              {view === 'forecast' && data && (
                <div className="flex flex-col py-8 gap-10 pb-20">
                   <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="bg-white rounded-2xl p-8 sm:p-10 border border-zinc-200/90 shadow-[0_1px_2px_rgba(0,0,0,0.04)] relative overflow-hidden group">
                         <div className="absolute top-0 right-0 p-10 opacity-[0.06]"><ZapIcon /></div>
                         <div className="flex justify-between items-start mb-6">
                             <span className="text-[11px] font-medium text-zinc-500">Strategic regime (21d)</span>
                             {stability >= 100 ? <span className="text-[10px] font-semibold text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-lg border border-emerald-100">Stable</span> : null}
                          </div>
                         <div className="text-5xl sm:text-6xl font-semibold tracking-tight mb-4 text-zinc-900 capitalize">{data.locked_regime}</div>
                         <div className="flex items-center gap-4">
                            <div className="flex-1 h-2 bg-zinc-100 rounded-full overflow-hidden">
                               <div className="h-full bg-emerald-500" style={{ width: `${data.locked_confidence * 100}%` }} />
                            </div>
                            <span className="text-xs font-semibold text-emerald-600 tabular-nums">{(data.locked_confidence*100).toFixed(1)}%</span>
                         </div>
                      </div>
                      <div className="bg-zinc-900 text-white rounded-2xl p-8 sm:p-10 shadow-lg relative overflow-hidden">
                          <div className="absolute inset-0 bg-gradient-to-br from-white/[0.07] to-transparent pointer-events-none" />
                          <span className="text-[11px] font-medium text-zinc-400 mb-6 block relative">Tactical pulse (5d)</span>
                          <div className="text-5xl sm:text-6xl font-semibold tracking-tight mb-4 relative">Off</div>
                          <div className="flex items-center gap-4 relative">
                             <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                                <div className="h-full bg-zinc-600" style={{ width: `0%` }} />
                             </div>
                             <span className="text-xs font-medium text-zinc-400">Not wired</span>
                          </div>
                       </div>
                    </div>
                    <div className="bg-white rounded-2xl border border-zinc-200/90 shadow-[0_1px_2px_rgba(0,0,0,0.04)] overflow-hidden">
                       {/* Panel Header */}
                       <div className="px-8 py-5 border-b border-zinc-100 flex items-center justify-between">
                         <div>
                           <h2 className="text-[14px] font-semibold text-zinc-900 tracking-tight">Signal Scorecard</h2>
                           <p className="text-[11px] text-zinc-400 mt-0.5">Two-layer confirmation — ML primary · Technicals cross-check</p>
                         </div>
                         <div className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-[11px] font-bold border ${
                           data.regime === 'bull' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                           data.regime === 'bear' ? 'bg-rose-50 text-rose-700 border-rose-200' :
                           'bg-zinc-50 text-zinc-600 border-zinc-200'
                         }`}>
                           {data.regime === 'bull' ? '▲' : data.regime === 'bear' ? '▼' : '◆'} {String(data.regime).toUpperCase()} REGIME
                         </div>
                       </div>

                       {/* Scorecard table */}
                       <SignalScorecard data={data} />

                       {/* Sideways note when applicable */}
                       {data.regime === 'sideways' && (
                         <div className="mx-6 mb-6 px-4 py-3 rounded-xl bg-zinc-50 border border-zinc-200/80 text-[11px] text-zinc-500 leading-relaxed">
                           <span className="font-bold text-zinc-700 block mb-1">◆ Sideways declared because:</span>
                           Conditions: |ret_21d| &lt; 5% ·&nbsp; ADX &lt; 20 (weak trend) ·&nbsp; vol_ratio 0.90–1.10 ·&nbsp; RSI 45–55 ·&nbsp; MACD near zero.
                           Evidence is mixed — take a cautious stance.
                         </div>
                       )}
                    </div>
                 </div>
               )}

               {view === 'horizons' && (
                 <div className="flex flex-col py-8 gap-8 pb-20">
                   {!forecastData ? (
                     <div className="bg-white rounded-2xl p-10 border border-zinc-200/90 text-center text-sm text-zinc-500">
                       No forward forecast for this series. Multi-horizon forecasts are generated for in-sample instruments (NIFTY 50, BANK NIFTY, NIFTY 500).
                     </div>
                   ) : (
                     <>
                       {/* Headline deployment gauge */}
                       <div className="bg-zinc-900 text-white rounded-2xl p-8 sm:p-10 shadow-lg relative overflow-hidden">
                         <div className="absolute inset-0 bg-gradient-to-br from-white/[0.07] to-transparent pointer-events-none" />
                         <div className="flex flex-wrap items-end justify-between gap-6 relative">
                           <div>
                             <span className="text-[11px] font-medium text-zinc-400 block mb-2">Recommended deployment · as of {forecastData.as_of}</span>
                             <div className="text-6xl font-semibold tracking-tight tabular-nums">{forecastData.deploy_pct}%</div>
                             <span className="text-[11px] text-zinc-400 mt-2 block">
                               {forecastData.detection?.regime
                                 ? `Current regime ${String(forecastData.detection.regime).toUpperCase()} · risk outlook trusted to ${forecastData.max_trusted_horizon}d`
                                 : ''}
                             </span>
                           </div>
                           <div className="w-full sm:w-64">
                             <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                               <div className="h-full bg-emerald-500" style={{ width: `${forecastData.deploy_pct}%` }} />
                             </div>
                             <div className="flex justify-between text-[10px] text-zinc-500 mt-1"><span>Defensive</span><span>Full</span></div>
                           </div>
                         </div>
                       </div>

                       {/* Detection: current-regime nowcast (high accuracy) */}
                       {forecastData.detection && (
                         <div className="bg-white rounded-2xl p-8 border border-zinc-200/90 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
                           <div className="flex flex-wrap items-center justify-between gap-4">
                             <div>
                               <span className="text-[11px] font-medium text-zinc-500 block mb-1">Current regime — detection (nowcast)</span>
                               <div className={`text-5xl font-semibold capitalize tracking-tight ${
                                 forecastData.detection.regime === 'bull' ? 'text-emerald-600'
                                   : forecastData.detection.regime === 'bear' ? 'text-rose-600' : 'text-zinc-500'
                               }`}>{forecastData.detection.regime}</div>
                               <span className="text-[11px] text-zinc-400 mt-1 block">{(forecastData.detection.confidence * 100).toFixed(0)}% confidence</span>
                             </div>
                             <div className="text-right text-[11px] text-zinc-500 space-y-1">
                               <div>Backtest accuracy <span className="font-semibold text-emerald-600 tabular-nums">{(forecastData.detection.oof_accuracy * 100).toFixed(1)}%</span></div>
                               <div>vs baseline <span className="tabular-nums">{(forecastData.detection.baseline * 100).toFixed(1)}%</span></div>
                               <div className="text-zinc-400">+{((forecastData.detection.oof_accuracy - forecastData.detection.baseline) * 100).toFixed(0)} pts edge</div>
                             </div>
                           </div>
                         </div>
                       )}

                       {/* Forward risk ladder: calm/stormy at 5 / 10 / 15 / 21 day */}
                       <div>
                         <h3 className="text-[12px] font-semibold text-zinc-700 mb-3 px-1">Forward risk outlook (next 5 / 10 / 15 / 21 days)</h3>
                         <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                           {(forecastData.risk || []).map((r: any) => {
                             const stormy = r.outlook === 'stormy';
                             const trustCls = r.trust === 'GREEN' ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                               : r.trust === 'AMBER' ? 'bg-amber-50 text-amber-700 border-amber-200'
                               : 'bg-zinc-100 text-zinc-500 border-zinc-200';
                             const dim = r.trust === 'RED' ? 'opacity-60' : '';
                             return (
                               <div key={r.horizon} className={`bg-white rounded-2xl p-6 border border-zinc-200/90 shadow-[0_1px_2px_rgba(0,0,0,0.04)] ${dim}`}>
                                 <div className="flex items-center justify-between mb-4">
                                   <span className="text-[13px] font-semibold text-zinc-900">{r.horizon}-day</span>
                                   <span className={`text-[10px] font-bold px-2 py-0.5 rounded-lg border ${trustCls}`}>{r.trust}</span>
                                 </div>
                                 <div className={`text-3xl font-semibold capitalize tracking-tight mb-1 ${stormy ? 'text-rose-600' : 'text-emerald-600'}`}>{r.outlook}</div>
                                 <div className="text-[11px] text-zinc-400 mb-4">{(r.storm_prob * 100).toFixed(0)}% storm probability</div>
                                 <div className="space-y-1.5 text-[11px] text-zinc-500 border-t border-zinc-100 pt-3">
                                   <div className="flex justify-between"><span>AUC (skill)</span><span className="tabular-nums text-zinc-700">{r.auc.toFixed(3)}</span></div>
                                   <div className="flex justify-between"><span>Balanced acc</span><span className="tabular-nums">{(r.balanced_acc * 100).toFixed(1)}%</span></div>
                                   <div className="flex justify-between"><span>Storm recall</span><span className="tabular-nums">{(r.storm_recall * 100).toFixed(0)}%</span></div>
                                 </div>
                               </div>
                             );
                           })}
                         </div>
                       </div>

                       {/* Honest reading guide */}
                       <div className="px-5 py-4 rounded-xl bg-zinc-50 border border-zinc-200/80 text-[11px] text-zinc-500 leading-relaxed">
                         <span className="font-bold text-zinc-700">How to read this:</span> <b>Detection</b> nowcasts the regime you are in now — the present is observable, so accuracy is ~90%. <b>Forward risk</b> forecasts whether the next 5–21 days will be calm or stormy, graded by <b>AUC</b> (ranking skill) because the calm class dominates and raw accuracy would be misleading. <span className="text-emerald-700 font-semibold">GREEN</span> AUC ≥ 0.62, <span className="text-amber-700 font-semibold">AMBER</span> ≥ 0.55, <span className="text-zinc-600 font-semibold">RED</span> below. Direction (up/down) is deliberately excluded — it is not forecastable at these horizons. Deploy % follows the detected regime, cut when a trusted storm is likely.
                       </div>
                     </>
                   )}
                 </div>
               )}

               {view === 'model' && modelMetrics && (
                 <div className="flex flex-col py-8 gap-8 pb-20">
                   <div className="flex flex-wrap items-center justify-between mb-2">
                     <div>
                       <h2 className="text-xl font-semibold text-zinc-900 tracking-tight">XGBoost Diagnostics</h2>
                       <p className="text-[12px] text-zinc-500 mt-1">Deep inspection of {primaryFile.replace(/_/g, " ")} regime classifier.</p>
                     </div>
                     <div className="flex items-center gap-4 bg-zinc-900 text-white px-4 py-2 rounded-xl text-[11px] font-mono">
                       <div><span className="text-zinc-400">OBJ</span> {modelMetrics.complexity?.objective}</div>
                       <div><span className="text-zinc-400">TREES</span> {modelMetrics.complexity?.total_trees}</div>
                       <div><span className="text-zinc-400">NODES</span> {modelMetrics.complexity?.est_nodes}</div>
                     </div>
                   </div>
                   <ModelVisualizer data={modelMetrics} />
                 </div>
               )}

            </div>
          </div>
        ) : (
          <div
            className="flex-1 flex flex-col items-center justify-center gap-4 text-zinc-400 bg-[#fafafa] px-6 border-2 border-dashed border-transparent hover:border-zinc-200/80 transition-colors"
            onDragOver={(e) => {
              e.preventDefault();
              e.dataTransfer.dropEffect = "copy";
            }}
            onDrop={handleDropJson}
          >
             <RadarIcon />
             <p className="text-sm font-medium text-zinc-500 text-center max-w-sm">No data loaded. Choose an instrument in the sidebar or drop a regime JSON file here.</p>
          </div>
        )}
      </main>
    </div>
  );
}

// --- SUB-COMPONENTS (LIGHT) ---

function StatRibbon({
  items,
}: {
  items: { label: string; value: string; hint?: string }[];
}) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-2.5 w-full">
      {items.map((it) => (
        <div
          key={it.label}
          className="rounded-xl border border-zinc-200/80 bg-white px-3 py-2.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.75)]"
        >
          <div className="text-[10px] font-medium text-zinc-500 uppercase tracking-wide">{it.label}</div>
          <div className="text-[13px] font-semibold text-zinc-900 tabular-nums mt-0.5 truncate" title={it.value}>
            {it.value}
          </div>
          {it.hint ? <div className="text-[10px] text-zinc-400 mt-0.5">{it.hint}</div> : null}
        </div>
      ))}
    </div>
  );
}

function ProsperityPanel({
  title,
  subtitle,
  right,
  chartHeightClass = "h-[300px]",
  children,
}: {
  title: string;
  subtitle?: string;
  right?: ReactNode;
  chartHeightClass?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col rounded-2xl border border-zinc-200/90 bg-white shadow-[0_1px_2px_rgba(0,0,0,0.04)] overflow-hidden">
      <div className="flex items-start justify-between gap-3 px-5 pt-4 pb-3 border-b border-zinc-100 shrink-0">
        <div className="min-w-0">
          <h3 className="text-[13px] font-semibold text-zinc-800 tracking-tight">{title}</h3>
          {subtitle ? <p className="text-[11px] text-zinc-500 mt-0.5">{subtitle}</p> : null}
        </div>
        {right}
      </div>
      <div className={`relative w-full ${chartHeightClass}`}>{children}</div>
      <div className="flex items-center justify-between px-5 py-2 border-t border-zinc-100/90 bg-zinc-50/50 shrink-0">
        <span className="text-[10px] text-zinc-400">Drag to zoom</span>
        <span className="text-[10px] text-zinc-500 tabular-nums">Expand</span>
      </div>
    </div>
  );
}

function YearlyPerformanceTable({ data }: { data: any[] }) {
  const months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
  
  const getCellStyles = (val: number | null) => {
    if (val === null) return "bg-zinc-50/50 text-zinc-300 border-transparent";
    if (val >= 80) return "bg-emerald-500 text-white shadow-[0_0_15px_rgba(16,185,129,0.3)] border-emerald-400/50 scale-[1.05] z-10";
    if (val >= 65) return "bg-emerald-50 text-emerald-700 border-emerald-100";
    if (val >= 50) return "bg-zinc-100 text-zinc-500 border-zinc-200";
    return "bg-rose-50 text-rose-700 border-rose-100";
  };

  return (
    <div className="mt-10 overflow-x-auto rounded-[32px] border border-zinc-100 bg-zinc-50/30 p-1">
      <table className="w-full border-separate border-spacing-1">
        <thead>
          <tr className="bg-zinc-900 rounded-2xl">
            <th className="text-[9px] font-black text-zinc-400 uppercase tracking-[0.2em] p-4 text-left rounded-l-2xl">YEAR_ID</th>
            {months.map(m => (
              <th key={m} className="text-[9px] font-black text-zinc-500 uppercase tracking-[0.2em] p-4 text-center">{m}</th>
            ))}
            <th className="text-[9px] font-black text-emerald-400 uppercase tracking-[0.2em] p-4 text-center rounded-r-2xl">ANNUAL_AVG</th>
          </tr>
        </thead>
        <tbody className="before:block before:h-2">
          {data.map((row, i) => (
            <tr key={i} className="group">
              <td className="text-xs font-black text-zinc-900 p-4 bg-white/50 rounded-l-xl border-l border-y border-zinc-100 tracking-tighter italic">
                {row.year}
              </td>
              {row.months.map((m: any, mi: number) => (
                <td key={mi} className="p-0.5">
                   <div className={`h-12 flex items-center justify-center rounded-xl text-[10px] font-mono font-black border transition-all duration-300 group-hover:opacity-60 hover:!opacity-100 hover:scale-110 cursor-default shadow-sm ${getCellStyles(m)}`}>
                      {m !== null ? `${m.toFixed(0)}%` : '—'}
                   </div>
                </td>
              ))}
              <td className="p-1">
                 <div className="h-12 flex items-center justify-center rounded-r-xl bg-zinc-900 text-white text-[11px] font-mono font-black shadow-xl italic border border-zinc-800 tracking-tighter">
                   {row.yearlyAvg !== null ? `${row.yearlyAvg.toFixed(1)}%` : 'DATA_NA'}
                 </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatCard({ label, value, sub }: { label: string, value: string, sub: string }) {
  return (
     <div className="bg-white border border-zinc-200/90 p-6 sm:p-8 rounded-2xl flex flex-col justify-between group hover:border-zinc-300 transition-colors shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
        <span className="text-[11px] font-medium text-zinc-500 mb-6">{label}</span>
        <div className="mt-auto flex flex-col gap-1">
           <span className="text-4xl sm:text-5xl font-semibold text-zinc-900 tracking-tight tabular-nums">{value}</span>
           <span className="text-[11px] text-zinc-500 mt-2">{sub}</span>
        </div>
     </div>
  );
}

function ChecklistItem({ label, active }: { label: string, active: boolean }) {
  return (
     <div className="flex items-center justify-between group">
        <span className={`text-xs font-black tracking-tighter transition-all ${active ? 'text-zinc-900' : 'text-zinc-300'}`}>{label}</span>
        <div className={`w-10 h-6 rounded-full p-1 transition-all flex items-center ${active ? 'bg-emerald-500' : 'bg-zinc-100 border border-zinc-200'}`}>
           <div className={`w-4 h-4 rounded-full transition-all ${active ? 'translate-x-4 bg-white shadow-sm' : 'bg-zinc-300'}`} />
        </div>
     </div>
  );
}

function ModuleHeader({ title, rightLabel }: { title: string, rightLabel?: string }) {
  return (
    <div className="flex flex-wrap justify-between items-start gap-3 mb-8">
      <h2 className="text-[15px] font-semibold text-zinc-900 tracking-tight border-l-[3px] border-emerald-600 pl-3">{title}</h2>
      {rightLabel ? (
         <div className="px-3 py-1.5 rounded-lg bg-zinc-50 border border-zinc-200/80 text-[11px] font-medium text-zinc-500">
            {rightLabel}
         </div>
      ) : null}
    </div>
  );
}

function SignalScorecard({ data }: { data: any }) {
  const regime = data?.regime ?? 'sideways';
  const cl = data?.checklist;

  // Build rows with live values and thresholds
  // active: whether the condition is met
  // value: the raw live number to display
  // threshold: the boundary condition (shown as text)
  // deviation: 0..1 normalized "how far above/past threshold" for the bar
  const bullRows = [
    {
      cat: 'Momentum', signal: 'Monthly return', condition: '> +5%',
      value: data?.ret_21d != null ? `${(data.ret_21d * 100).toFixed(1)}%` : '—',
      active: cl?.bull?.momentum_21d ?? false,
      deviation: Math.min(Math.max((data?.ret_21d ?? 0) / 0.10, 0), 1),
    },
    {
      cat: 'Momentum', signal: '3-month trend', condition: '> +8%',
      value: data?.ret_63d != null ? `${(data.ret_63d * 100).toFixed(1)}%` : '—',
      active: cl?.bull?.trend_63d ?? false,
      deviation: Math.min(Math.max((data?.ret_63d ?? 0) / 0.16, 0), 1),
    },
    {
      cat: 'Volatility', signal: 'Recent vol vs baseline', condition: 'vol_21d < vol_63d',
      value: data?.vol_21d != null ? `${(data.vol_21d * 100).toFixed(1)}%` : '—',
      active: cl?.bull?.vol_stability ?? false,
      deviation: data?.vol_21d != null && data?.vol_63d != null
        ? Math.min(Math.max(1 - data.vol_21d / data.vol_63d, 0), 1) : 0,
    },
    {
      cat: 'Volatility', signal: 'Vol ratio (risk cooling)', condition: '< 0.90',
      value: data?.vol_ratio != null ? data.vol_ratio.toFixed(2) : '—',
      active: cl?.bull?.risk_cooling ?? false,
      deviation: data?.vol_ratio != null ? Math.min(Math.max(1 - data.vol_ratio / 0.9, 0), 1) : 0,
    },
    {
      cat: 'Risk', signal: '21d return skew', condition: '> −0.5',
      value: data?.skew_21d != null ? data.skew_21d.toFixed(2) : '—',
      active: cl?.bull?.tail_risk ?? (data?.skew_21d > -0.5),
      deviation: data?.skew_21d != null ? Math.min(Math.max((data.skew_21d + 0.5) / 2, 0), 1) : 0,
    },
    {
      cat: 'Technical', signal: 'EMA cross (Golden)', condition: 'EMA20 > EMA60',
      value: data?.ema_gap != null ? `gap ${data.ema_gap > 0 ? '+' : ''}${(data.ema_gap * 100).toFixed(2)}%` : '—',
      active: cl?.bull?.ema_cross ?? false,
      deviation: data?.ema_gap != null ? Math.min(Math.max(data.ema_gap / 0.03, 0), 1) : 0,
    },
    {
      cat: 'Technical', signal: 'MACD histogram', condition: '> 0',
      value: data?.macd_h != null ? data.macd_h.toFixed(1) : '—',
      active: cl?.bull?.macd_hist ?? false,
      deviation: data?.macd_h != null ? Math.min(Math.max(data.macd_h / 30, 0), 1) : 0,
    },
    {
      cat: 'Technical', signal: 'RSI-14', condition: '50 – 70',
      value: data?.rsi_14 != null ? data.rsi_14.toFixed(1) : '—',
      active: cl?.bull?.rsi_healthy ?? false,
      deviation: data?.rsi_14 != null ? Math.min(Math.max((data.rsi_14 - 50) / 20, 0), 1) : 0,
    },
    {
      cat: 'ML', signal: 'XGBoost P(Bull)', condition: '≥ 65%',
      value: data?.prob_bull != null ? `${(data.prob_bull * 100).toFixed(1)}%` : '—',
      active: cl?.bull?.ml_confidence ?? false,
      deviation: data?.prob_bull != null ? Math.min(Math.max((data.prob_bull - 0.45) / 0.55, 0), 1) : 0,
    },
  ];

  const bearRows = [
    {
      cat: 'Momentum', signal: 'Monthly return', condition: '< −5%',
      value: data?.ret_21d != null ? `${(data.ret_21d * 100).toFixed(1)}%` : '—',
      active: cl?.bear?.momentum_21d ?? false,
      deviation: Math.min(Math.max(-(data?.ret_21d ?? 0) / 0.10, 0), 1),
    },
    {
      cat: 'Momentum', signal: '3-month trend', condition: '< −8%',
      value: data?.ret_63d != null ? `${(data.ret_63d * 100).toFixed(1)}%` : '—',
      active: cl?.bear?.trend_63d ?? false,
      deviation: Math.min(Math.max(-(data?.ret_63d ?? 0) / 0.16, 0), 1),
    },
    {
      cat: 'Volatility', signal: 'Vol spike (panic)', condition: 'vol_21d > vol_63d × 1.3',
      value: data?.vol_21d != null && data?.vol_63d != null
        ? `ratio ${(data.vol_21d / data.vol_63d).toFixed(2)}×` : '—',
      active: cl?.bear?.panic_vol ?? false,
      deviation: data?.vol_21d != null && data?.vol_63d != null
        ? Math.min(Math.max((data.vol_21d / data.vol_63d - 1) / 0.6, 0), 1) : 0,
    },
    {
      cat: 'Risk', signal: '21d return skew', condition: '< −1.0',
      value: data?.skew_21d != null ? data.skew_21d.toFixed(2) : '—',
      active: cl?.bear?.tail_crash ?? false,
      deviation: data?.skew_21d != null ? Math.min(Math.max((-data.skew_21d - 0.5) / 1, 0), 1) : 0,
    },
    {
      cat: 'Technical', signal: 'EMA cross (Death)', condition: 'EMA20 < EMA60',
      value: data?.ema_gap != null ? `gap ${data.ema_gap > 0 ? '+' : ''}${(data.ema_gap * 100).toFixed(2)}%` : '—',
      active: cl?.bear?.ema_death_cross ?? false,
      deviation: data?.ema_gap != null ? Math.min(Math.max(-data.ema_gap / 0.03, 0), 1) : 0,
    },
    {
      cat: 'Technical', signal: 'MACD histogram', condition: '< 0',
      value: data?.macd_h != null ? data.macd_h.toFixed(1) : '—',
      active: cl?.bear?.macd_hist_neg ?? false,
      deviation: data?.macd_h != null ? Math.min(Math.max(-data.macd_h / 30, 0), 1) : 0,
    },
    {
      cat: 'Technical', signal: 'RSI-14', condition: '< 40',
      value: data?.rsi_14 != null ? data.rsi_14.toFixed(1) : '—',
      active: cl?.bear?.rsi_weak ?? false,
      deviation: data?.rsi_14 != null ? Math.min(Math.max((40 - data.rsi_14) / 30, 0), 1) : 0,
    },
    {
      cat: 'ML', signal: 'XGBoost P(Bear)', condition: '≥ 65%',
      value: data?.prob_bear != null ? `${(data.prob_bear * 100).toFixed(1)}%` : '—',
      active: cl?.bear?.ml_confidence ?? false,
      deviation: data?.prob_bear != null ? Math.min(Math.max((data.prob_bear - 0.45) / 0.55, 0), 1) : 0,
    },
  ];

  const rows = regime === 'bull' ? bullRows : bearRows;
  const passCount = rows.filter(r => r.active).length;
  const accentBull = '#10b981';
  const accentBear = '#f43f5e';
  const accent = regime === 'bull' ? accentBull : accentBear;
  const accentBg = regime === 'bull' ? 'bg-emerald-500' : 'bg-rose-500';
  const accentText = regime === 'bull' ? 'text-emerald-700' : 'text-rose-600';
  const accentLight = regime === 'bull' ? 'bg-emerald-50' : 'bg-rose-50';

  // Group rows by category
  const categories = Array.from(new Set(rows.map(r => r.cat)));

  return (
    <div className="px-6 py-4 pb-6">
      {/* Score summary bar */}
      <div className="flex items-center gap-4 mb-5 p-4 rounded-2xl bg-zinc-50 border border-zinc-200/70">
        <div className="flex-1">
          <div className="flex justify-between items-center mb-1.5">
            <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Signal Confirmation Score</span>
            <span className={`text-[13px] font-black ${passCount >= 6 ? accentText : passCount >= 4 ? 'text-amber-600' : 'text-zinc-500'}`}>
              {passCount} / {rows.length}
            </span>
          </div>
          <div className="h-2 bg-zinc-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${accentBg}`}
              style={{ width: `${(passCount / rows.length) * 100}%` }}
            />
          </div>
        </div>
        <div className={`text-center px-4 py-2 rounded-xl ${accentLight} border ${regime === 'bull' ? 'border-emerald-200' : 'border-rose-200'}`}>
          <div className={`text-[22px] font-black ${accentText}`}>{Math.round((passCount / rows.length) * 100)}%</div>
          <div className="text-[8px] text-zinc-400 font-bold uppercase tracking-wider">Conviction</div>
        </div>
      </div>

      {/* Signal rows, grouped */}
      <div className="flex flex-col">
        {/* Table header */}
        <div className="grid gap-2 mb-2 px-1" style={{ gridTemplateColumns: '90px 1fr 90px 80px 100px' }}>
          {['Category', 'Signal', 'Condition', 'Live Value', 'Strength'].map(h => (
            <span key={h} className="text-[9px] font-bold text-zinc-400 uppercase tracking-widest">{h}</span>
          ))}
        </div>

        <div className="flex flex-col gap-1.5">
          {rows.map((row, i) => {
            const isFirst = i === 0 || rows[i - 1].cat !== row.cat;
            const barColor = row.active ? accent : '#d4d4d8';
            return (
              <div key={i}>
                {isFirst && i !== 0 && <div className="border-t border-zinc-100 my-1.5" />}
                <div
                  className={`grid gap-2 items-center px-3 py-2.5 rounded-xl transition-all ${
                    row.active
                      ? regime === 'bull' ? 'bg-emerald-50/60' : 'bg-rose-50/60'
                      : 'bg-white hover:bg-zinc-50'
                  }`}
                  style={{ gridTemplateColumns: '90px 1fr 90px 80px 100px' }}
                >
                  {/* Category pill */}
                  <span className="text-[8px] font-bold text-zinc-400 uppercase tracking-widest truncate">
                    {isFirst ? row.cat : ''}
                  </span>

                  {/* Signal name */}
                  <div className="flex items-center gap-2 min-w-0">
                    <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${row.active ? accentBg : 'bg-zinc-300'}`} />
                    <span className={`text-[11px] font-semibold truncate ${row.active ? 'text-zinc-800' : 'text-zinc-400'}`}>
                      {row.signal}
                    </span>
                  </div>

                  {/* Condition */}
                  <span className="text-[9px] font-mono text-zinc-400 truncate">{row.condition}</span>

                  {/* Live value */}
                  <span className={`text-[12px] font-black tabular-nums ${row.active ? accentText : 'text-zinc-400'}`}>
                    {row.value}
                  </span>

                  {/* Deviation bar */}
                  <div className="flex items-center gap-1.5">
                    <div className="flex-1 h-2 bg-zinc-100 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{ width: `${row.deviation * 100}%`, backgroundColor: barColor }}
                      />
                    </div>
                    <span className="text-[8px] font-bold text-zinc-400 w-6 text-right tabular-nums">
                      {Math.round(row.deviation * 100)}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function SvgMiniChart({ realizedPath, predictedRegime }: { realizedPath: { day: number; price: number }[]; predictedRegime: string }) {
  if (!realizedPath || realizedPath.length < 2) {
    return (
      <svg width="100%" height="100%" viewBox="0 0 360 110" preserveAspectRatio="none">
        <text x="180" y="58" textAnchor="middle" fill="#a1a1aa" fontSize="10" fontFamily="monospace">NO DATA</text>
      </svg>
    );
  }

  const W = 360;
  const H = 110;
  const PT = 10; // pad top
  const PB = 18; // pad bottom (for labels)
  const PL = 4;
  const PR = 4;
  const cW = W - PL - PR;
  const cH = H - PT - PB;

  const prices = realizedPath.map(p => p.price);
  const minP = Math.min(...prices);
  const maxP = Math.max(...prices);
  const range = maxP - minP || 1;
  const n = realizedPath.length;
  const startPrice = prices[0];

  const toX = (i: number) => PL + (n > 1 ? (i / (n - 1)) : 0.5) * cW;
  const toY = (p: number) => PT + cH - ((p - minP) / range) * cH;

  const points = realizedPath.map((p, i) => `${toX(i).toFixed(1)},${toY(p.price).toFixed(1)}`).join(' ');
  const entryY = toY(startPrice);

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ display: 'block' }}>
      {/* Segment fills — pastel green/red per day */}
      {realizedPath.slice(1).map((pt, idx) => {
        const x1 = toX(idx);
        const x2 = toX(idx + 1);
        const cp = pt.price;
        let ok = false;
        if (predictedRegime === 'bull') ok = cp > startPrice;
        else if (predictedRegime === 'bear') ok = cp < startPrice;
        else ok = Math.abs((cp - startPrice) / startPrice) < 0.02;
        return (
          <rect
            key={idx}
            x={x1.toFixed(1)} y={PT}
            width={Math.max(x2 - x1, 1).toFixed(1)} height={cH}
            fill={ok ? '#d1fae5' : '#fee2e2'}
          />
        );
      })}

      {/* Subtle grid lines */}
      {[0.33, 0.66].map(f => (
        <line key={f} x1={PL} y1={(PT + cH * f).toFixed(1)} x2={W - PR} y2={(PT + cH * f).toFixed(1)}
          stroke="#e4e4e7" strokeWidth="0.6" />
      ))}

      {/* Entry price dashed reference */}
      <line x1={PL} y1={entryY.toFixed(1)} x2={W - PR} y2={entryY.toFixed(1)}
        stroke="#94a3b8" strokeWidth="1" strokeDasharray="4 3" opacity="0.5" />

      {/* Price path drawn on top */}
      <polyline points={points} fill="none" stroke="#18181b" strokeWidth="2.5"
        strokeLinejoin="round" strokeLinecap="round" />

      {/* Start dot */}
      <circle cx={toX(0).toFixed(1)} cy={toY(prices[0]).toFixed(1)} r="3" fill="#18181b" />

      {/* End dot — colored by outcome */}
      <circle
        cx={toX(n - 1).toFixed(1)} cy={toY(prices[n - 1]).toFixed(1)} r="4"
        fill={prices[n - 1] > startPrice ? '#10b981' : '#f43f5e'}
        stroke="white" strokeWidth="1.5"
      />

      {/* Day labels */}
      {[0, 5, 10, 15, 20].filter(d => d < n).map(d => (
        <text key={d} x={toX(d).toFixed(1)} y={H - 3} textAnchor="middle"
          fontSize="7" fill="#94a3b8" fontFamily="monospace" fontWeight="600">
          T+{d}
        </text>
      ))}
    </svg>
  );
}

function CustomHud({ active, payload, data: fallbackData, history }: any) {
  const data = active && payload && payload.length ? payload[0].payload : fallbackData;
  if (!data || !history) return null;

  const currentIndex = history.findIndex((h: any) => h.date === data.date);
  const realizedPath: { day: number; price: number }[] = history
    .slice(currentIndex, currentIndex + 22)
    .map((h: any, i: number) => ({ day: i, price: h.close }));

  const isHit = data.regime === data.regime_actual;

  return (
    <div
      className="bg-white border border-zinc-200/80 rounded-2xl shadow-2xl flex flex-col gap-0 overflow-hidden pointer-events-none"
      style={{ width: 400, fontFamily: 'monospace' }}
    >
      {/* Top colour bar */}
      <div className={`h-1 w-full ${data.regime === 'bull' ? 'bg-emerald-400' : data.regime === 'bear' ? 'bg-rose-400' : 'bg-zinc-400'}`} />

      {/* Header */}
      <div className="px-4 pt-3 pb-2.5 border-b border-zinc-100 flex justify-between items-center">
        <div>
          <div className="text-[9px] font-bold text-zinc-400 uppercase tracking-widest">{data.date}</div>
          <div className={`text-[10px] font-bold mt-0.5 ${isHit ? 'text-emerald-600' : 'text-rose-500'}`}>
            {isHit ? '✓ SIGNAL VALIDATED' : '✗ REGIME DRIFT'}
          </div>
        </div>
        <div className={`text-[10px] font-black px-2.5 py-1 rounded-lg ${data.regime === 'bull' ? 'bg-emerald-500 text-white' : data.regime === 'bear' ? 'bg-rose-500 text-white' : 'bg-zinc-800 text-white'}`}>
          {String(data.regime).toUpperCase()}
        </div>
      </div>

      {/* Metrics 4-up */}
      <div className="grid grid-cols-4 gap-px bg-zinc-100">
        {[
          { k: 'PRICE', v: data.close ? Math.floor(data.close).toLocaleString() : '—' },
          { k: 'CONF', v: `${((data.confidence || 0) * 100).toFixed(0)}%` },
          { k: 'TII_21', v: `${data.tii_21 ? (data.tii_21 * 100).toFixed(0) : '0'}%` },
          { k: 'VOL_STB', v: data.vol_of_vol_21 ? (1 - data.vol_of_vol_21).toFixed(2) : '—' },
        ].map(m => (
          <div key={m.k} className="bg-white px-3 py-2">
            <div className="text-[7px] text-zinc-400 font-bold uppercase tracking-widest">{m.k}</div>
            <div className="text-[13px] text-zinc-900 font-black mt-0.5 tabular-nums">{m.v}</div>
          </div>
        ))}
      </div>

      {/* Chart area */}
      <div className="px-3 pt-2.5 pb-1">
        <div className="flex justify-between items-center mb-1">
          <span className="text-[7.5px] text-zinc-400 font-bold uppercase tracking-widest">Realized Path T+1→T+21</span>
          <div className="flex gap-2.5 items-center">
            <div className="flex items-center gap-1"><div className="w-3 h-2 rounded-sm bg-emerald-100 border border-emerald-300"/><span className="text-[7px] text-zinc-400">Correct</span></div>
            <div className="flex items-center gap-1"><div className="w-3 h-2 rounded-sm bg-red-100 border border-red-300"/><span className="text-[7px] text-zinc-400">Drift</span></div>
          </div>
        </div>
        <div className="rounded-xl overflow-hidden border border-zinc-100 bg-zinc-50" style={{ height: 120 }}>
          <SvgMiniChart realizedPath={realizedPath} predictedRegime={data.regime} />
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 pb-3 pt-1.5 flex justify-between items-center">
        <span className="text-[7.5px] font-bold text-zinc-400 uppercase tracking-widest">RSI_14</span>
        <span className="text-[11px] font-black text-zinc-700 tabular-nums">{data.rsi_14 ? data.rsi_14.toFixed(1) : '—'}</span>
        <span className="text-[7.5px] font-bold text-zinc-400 uppercase tracking-widest">RET_21D</span>
        <span className={`text-[11px] font-black tabular-nums ${(data.ret_21d || 0) >= 0 ? 'text-emerald-600' : 'text-rose-500'}`}>
          {data.ret_21d ? `${(data.ret_21d * 100).toFixed(1)}%` : '—'}
        </span>
      </div>
    </div>
  );
}

import { ScatterChart, Scatter, ZAxis } from "recharts";

function ModelVisualizer({ data }: { data: any }) {
  const [activeTab, setActiveTab] = useState<'importance' | 'impact' | 'tree' | 'predictions'>('importance');
  
  if (!data) return null;

  // Calculate cumulative importance for overlay
  let cumulative = 0;
  const topFeatures = (data.feature_importance?.slice(0, 12).reverse() || []).map((f: any, idx: number, arr: any[]) => {
    // Reverse iterating for bottom-up charting, so we calculate cumulative from the top down (array end)
    return { ...f };
  });
  
  // Actually, for vertical rendering, Recharts places the first element at the bottom.
  // Let's sort them ascending so largest is at the top of the chart.
  const chartFeatures = [...(data.feature_importance?.slice(0, 12) || [])].sort((a,b) => a.gain - b.gain).map(f => ({
      ...f,
      gain_pct: (f.gain / data.feature_importance.reduce((acc:any, curr:any) => acc + curr.gain, 0)) * 100
  }));

  const maxGain = Math.max(...chartFeatures.map(f => f.gain));
  const topFeature = chartFeatures[chartFeatures.length - 1];

  return (
    <div className="bg-[#0c0c0c] text-[#a1a1aa] font-mono rounded-xl overflow-hidden border border-zinc-800 shadow-2xl flex flex-col h-[700px]">
      
      {/* Top Header / Sub-nav */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800/80 bg-[#121212]">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="text-zinc-500 font-bold uppercase tracking-widest text-[9px]">Engine</span>
            <span className="text-zinc-200 font-bold text-xs uppercase tracking-wide">XGBOOST</span>
          </div>
          <div className="h-4 w-px bg-zinc-800" />
          <nav className="flex items-center gap-1">
            <button 
              onClick={() => setActiveTab('importance')}
              className={`px-4 py-1.5 text-[11px] font-medium rounded-lg transition-colors ${activeTab === 'importance' ? 'text-zinc-200 bg-zinc-800' : 'hover:text-zinc-300'}`}
            >
              Feature Importance
            </button>
            <button 
              onClick={() => setActiveTab('impact')}
              className={`px-4 py-1.5 text-[11px] font-medium rounded-lg transition-colors ${activeTab === 'impact' ? 'text-zinc-200 bg-zinc-800' : 'hover:text-zinc-300'}`}
            >
              SHAP / Beeswarm
            </button>
            <button 
              onClick={() => setActiveTab('tree')}
              className={`px-4 py-1.5 text-[11px] font-medium rounded-lg transition-colors ${activeTab === 'tree' ? 'text-zinc-200 bg-zinc-800' : 'hover:text-zinc-300'}`}
            >
              Decision Tree
            </button>
            <button 
              onClick={() => setActiveTab('predictions')}
              className={`px-4 py-1.5 text-[11px] font-medium rounded-lg transition-colors ${activeTab === 'predictions' ? 'text-zinc-200 bg-zinc-800' : 'hover:text-zinc-300'}`}
            >
              Predictions
            </button>
          </nav>
        </div>
      </div>

      <div className="flex flex-1 min-h-0 bg-[#0c0c0c]">
        {/* Left Sidebar */}
        <div className="w-[220px] shrink-0 border-r border-zinc-800/80 flex flex-col p-4 gap-4 overflow-y-auto">
          <div>
            <span className="block text-[9px] font-bold text-zinc-600 uppercase tracking-widest mb-2">Objective Function</span>
            <div className="bg-[#1a1a1a] border border-zinc-800 rounded-lg px-3 py-2 text-[11px] text-zinc-300">
              {data.complexity?.objective || "multi:softprob"}
            </div>
          </div>
          <div className="pt-2 border-t border-zinc-800/50">
             <div className="flex items-center gap-2 mb-3">
               <span className="block text-[9px] font-bold text-zinc-600 uppercase tracking-widest">Hyperparameters</span>
             </div>
             <div className="flex flex-col gap-1.5 text-[10px]">
               {Object.entries(data.hyperparameters || {}).map(([k, v]) => (
                 <div key={k} className="flex justify-between p-1.5 hover:bg-zinc-800/30 rounded">
                   <span className="text-zinc-500">{k}</span>
                   <span className="text-zinc-300 font-bold">{String(v)}</span>
                 </div>
               ))}
             </div>
          </div>
        </div>

        {/* Main Chart Area */}
        <div className="flex-1 min-w-0 p-8 flex flex-col relative">
          
          {activeTab === 'importance' && (
            <>
              <div className="text-center mb-6">
                <h3 className="text-zinc-200 font-semibold text-sm">Feature Importance Analysis</h3>
                <p className="text-zinc-500 text-[10px] mt-1">Sorted by Information gain | {data.feature_importance?.length} features</p>
              </div>
              <div className="flex-1 min-h-0 relative">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartFeatures} layout="vertical" barCategoryGap="25%">
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} vertical={true} stroke="#27272a" opacity={0.5} />
                    <XAxis type="number" hide domain={[0, maxGain * 1.1]} />
                    <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false} tick={{fontSize: 10, fill: '#a1a1aa'}} width={120} />
                    <Tooltip 
                      cursor={{fill: '#18181b'}} 
                      contentStyle={{backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '8px', color: '#e4e4e7', fontSize: '11px'}}
                      formatter={(value: any, name: any, props: any) => [
                        `${Number(value).toFixed(2)} (${props.payload.gain_pct.toFixed(1)}%)`, 'Gain'
                      ]}
                    />
                    <Bar dataKey="gain" radius={[0, 4, 4, 0]}>
                      {chartFeatures.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={index === chartFeatures.length - 1 ? '#d4d4d8' : '#3f3f46'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {activeTab === 'impact' && (
            <>
               <div className="text-center mb-6">
                <h3 className="text-zinc-200 font-semibold text-sm">Feature impact distribution (Beeswarm)</h3>
                <p className="text-zinc-500 text-[10px] mt-1">Each dot = one prediction | X-Axis = Bull Probability | Color = Final Outcome</p>
              </div>
              <div className="flex-1 min-h-0 relative">
                 <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 0, right: 10, bottom: 20, left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} horizontal={true} stroke="#27272a" opacity={0.5} />
                    <XAxis 
                      type="number" dataKey="prob_bull" name="Bull Conviction" domain={[0, 1]} 
                      tick={{fontSize: 10, fill: '#71717a'}} axisLine={false} tickLine={false} 
                    />
                    <YAxis 
                      dataKey="feature" type="category" allowDuplicatedCategory={false}
                      axisLine={false} tickLine={false} tick={{fontSize: 10, fill: '#a1a1aa'}} width={120} 
                    />
                    <ZAxis dataKey="prob_bull" range={[10, 20]} />
                    <Tooltip 
                      cursor={{strokeDasharray: '3 3', stroke: '#52525b'}}
                      contentStyle={{backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '8px', color: '#e4e4e7', fontSize: '11px'}}
                    />
                    <ReferenceLine x={0.5} stroke="#52525b" strokeDasharray="3 3" />
                    <Scatter name="Bull" data={(data.scatter_analysis || []).filter((d:any) => d.regime === 'bull')} fill="#10b981" fillOpacity={0.7} />
                    <Scatter name="Bear" data={(data.scatter_analysis || []).filter((d:any) => d.regime === 'bear')} fill="#f43f5e" fillOpacity={0.7} />
                    <Scatter name="Sideways" data={(data.scatter_analysis || []).filter((d:any) => d.regime === 'sideways')} fill="#71717a" fillOpacity={0.5} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {activeTab === 'tree' && (
            <>
               <div className="text-center mb-6 shrink-0">
                <h3 className="text-zinc-200 font-semibold text-sm">Decision Tree Topology</h3>
                <p className="text-zinc-500 text-[10px] mt-1">Single tree from ensemble (Tree [0])</p>
              </div>
              <div className="flex-1 overflow-auto bg-[#0a0a0a] rounded-xl border border-zinc-800 p-8 shadow-inner cursor-grab active:cursor-grabbing">
                <div className="min-w-max w-full flex justify-center items-start pt-8 pb-32">
                   {data.tree_graph ? <SimpleTree node={data.tree_graph} /> : <p className="text-zinc-600 text-xs">No tree data found</p>}
                </div>
              </div>
            </>
          )}

          {activeTab === 'predictions' && (
            <>
               <div className="text-center mb-6">
                <h3 className="text-zinc-200 font-semibold text-sm">Prediction Probability Matrix</h3>
                <p className="text-zinc-500 text-[10px] mt-1">Sample of 500 Out-Of-Sample & In-Sample points | Expected value vs Actual</p>
              </div>
              <div className="flex-1 min-h-0 relative bg-white/5 rounded-2xl p-6 border border-zinc-200/5">
                 <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={true} horizontal={true} stroke="#27272a" opacity={0.5} />
                    <XAxis 
                      type="number" dataKey="predicted" name="Expected Value" domain={[0, 1]} 
                      tick={{fontSize: 10, fill: '#71717a'}} axisLine={true} tickLine={true} 
                      label={{ value: "Predicted Value (0=Bear, 0.5=Side, 1=Bull)", position: 'insideBottom', offset: -10, fill: '#71717a', fontSize: 10 }}
                    />
                    <YAxis 
                      type="number" dataKey="actual" name="Actual" domain={[-0.1, 1.1]} 
                      axisLine={true} tickLine={true} tick={{fontSize: 10, fill: '#a1a1aa'}} 
                      label={{ value: "Actual Regime", angle: -90, position: 'insideLeft', offset: 0, fill: '#71717a', fontSize: 10 }}
                      ticks={[0.0, 0.5, 1.0]} tickFormatter={(val) => val === 1 ? "Bull (1.0)" : val === 0 ? "Bear (0.0)" : "Side (0.5)"} width={80} 
                    />
                    <ZAxis dataKey="predicted" range={[20, 20]} />
                    <Tooltip 
                      cursor={{strokeDasharray: '3 3', stroke: '#52525b'}}
                      contentStyle={{backgroundColor: '#18181b', border: '1px solid #3f3f46', borderRadius: '8px', color: '#e4e4e7', fontSize: '11px'}}
                      formatter={(val: any, name: any) => [Number(val).toFixed(3), name]}
                    />
                    <ReferenceLine x={0.0} y={0.0} stroke="#10b981" strokeDasharray="3 3" opacity={0.4} />
                    <Scatter name="Correct Bull" data={(data.predictions_analysis || []).filter((d:any) => d.actual === 1.0 && d.predicted > 0.6)} fill="#10b981" fillOpacity={0.7} />
                    <Scatter name="Correct Bear" data={(data.predictions_analysis || []).filter((d:any) => d.actual === 0.0 && d.predicted < 0.4)} fill="#f43f5e" fillOpacity={0.7} />
                    <Scatter name="Correct Side" data={(data.predictions_analysis || []).filter((d:any) => d.actual === 0.5 && d.predicted >= 0.4 && d.predicted <= 0.6)} fill="#71717a" fillOpacity={0.7} />
                    <Scatter name="Misses" data={(data.predictions_analysis || []).filter((d:any) => {
                      if (d.actual === 1.0 && d.predicted <= 0.6) return true;
                      if (d.actual === 0.0 && d.predicted >= 0.4) return true;
                      if (d.actual === 0.5 && (d.predicted < 0.4 || d.predicted > 0.6)) return true;
                      return false;
                    })} fill="#fbbf24" fillOpacity={0.8} shape="cross" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
          
        </div>

        {/* Right Sidebar Stats */}
        <div className="w-[200px] shrink-0 border-l border-zinc-800/80 p-4 flex flex-col gap-6 overflow-y-auto bg-[#0a0a0a]">
           
           <div>
             <span className="block text-[8px] font-bold text-zinc-600 uppercase tracking-widest mb-3">Features</span>
             <div className="flex flex-col gap-2 text-[10px]">
                <div className="flex justify-between items-end">
                  <span className="text-zinc-500">Top Feature</span>
                  <span className="text-zinc-300 font-bold truncate max-w-[90px]">{topFeature?.feature}</span>
                </div>
                <div className="flex justify-between items-end mt-1">
                  <span className="text-zinc-500">Top Share</span>
                  <span className="text-zinc-300 font-bold">{topFeature?.gain_pct?.toFixed(1)}%</span>
                </div>
                <div className="h-1 w-full bg-zinc-800 rounded-full mt-1 overflow-hidden">
                   <div className="h-full bg-zinc-400" style={{width: `${topFeature?.gain_pct || 0}%`}} />
                </div>
                <div className="flex justify-between items-end mt-2">
                  <span className="text-zinc-500">Total Features</span>
                  <span className="text-zinc-300 font-bold">{data.feature_importance?.length}</span>
                </div>
             </div>
           </div>

           <div>
             <span className="block text-[8px] font-bold text-zinc-600 uppercase tracking-widest mb-3">Complexity</span>
             <div className="flex flex-col gap-2 text-[10px]">
                <div className="flex justify-between items-end">
                  <span className="text-zinc-500">Est. Nodes</span>
                  <span className="text-zinc-300 font-bold">{data.complexity?.est_nodes?.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-end">
                  <span className="text-zinc-500">Trees Build</span>
                  <span className="text-zinc-300 font-bold">{data.complexity?.total_trees}</span>
                </div>
                <div className="flex justify-between items-end">
                  <span className="text-zinc-500">Overfit Risk</span>
                  <span className="text-emerald-500 font-bold">Low</span>
                </div>
             </div>
           </div>
           
           <div className="mt-auto opacity-40">
             <div className="text-[8px] leading-relaxed">
               Gain = Information per split.<br/>
               Impact = Probability shift.<br/>
               <span className="text-zinc-400 font-bold mt-2 block">London Strategic Edge</span>
               <span className="text-zinc-500 text-[7px]">Inspired Dashboard</span>
             </div>
           </div>
        </div>
      </div>
    </div>
  );
}

function SimpleTree({ node }: { node: any }) {
  if (!node) return null;
  return (
    <div className="flex flex-col items-center">
      <div className={`border px-3 py-1.5 rounded-lg text-[10px] whitespace-nowrap shadow-lg z-10 tracking-tight ${node.is_leaf ? 'bg-[#10b981]/10 border-[#10b981]/30 text-[#34d399] font-bold' : 'bg-[#18181b] border-zinc-700 text-zinc-300'}`}>
        {node.name}
      </div>
      {node.children && node.children.length > 0 && (
        <div className="flex relative mt-0 pt-6">
          {/* horizontal line connecting children */}
          <div className="absolute top-0 left-1/4 right-1/4 h-px bg-zinc-700" />
          {/* vertical drop from parent */}
          <div className="absolute -top-0 left-1/2 w-px h-6 bg-zinc-700 -translate-y-[23px]" />
          <div className="flex gap-12">
            {node.children.map((child: any, i: number) => (
              <div key={i} className="flex flex-col items-center relative">
                {/* vertical line dropping to child */}
                <div className="absolute -top-6 left-1/2 w-px h-6 bg-zinc-700" />
                <SimpleTree node={child} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}