"use client";

import { useState } from "react";

export default function Dashboard() {
  const [data, setData] = useState<any>(null);

  const handleFileUpload = (event: any) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = (e: any) => {
      try {
        const json = JSON.parse(e.target.result);

        // handle array vs object
        if (Array.isArray(json)) {
          setData(json[json.length - 1]);
        } else {
          setData(json);
        }

      } catch {
        alert("Invalid JSON file");
      }
    };

    reader.readAsText(file);
  };

  const getColor = () => {
    if (!data || !data.regime) return "bg-gray-400";
    if (data.regime.includes("bull")) return "bg-black";
    if (data.regime.includes("bear")) return "bg-gray-800";
    return "bg-gray-500";
  };

  return (
    <div className="flex h-screen bg-[#f5f5f7] text-black">

      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 p-4 flex flex-col">

        <h1 className="text-xl font-semibold mb-6">Regime Engine</h1>

        <label>
          <input
            type="file"
            accept=".json"
            onChange={handleFileUpload}
            className="hidden"
          />
          <div className="bg-black text-white p-2 rounded text-center cursor-pointer mb-4">
            Upload JSON
          </div>
        </label>

        <div className="p-2 bg-gray-100 rounded text-sm">Dashboard</div>

      </div>

      {/* Main */}
      <div className="flex-1 p-6 space-y-6">

        {!data ? (
          <div className="text-gray-500 text-lg">
            Upload your JSON file to start 🚀
          </div>
        ) : (
          <>
            {/* Regime Card */}
            <div className={`p-6 rounded-xl text-white ${getColor()}`}>
              <h2 className="text-xs opacity-80 mb-1">CURRENT REGIME</h2>
              <p className="text-2xl font-medium">
                {data?.regime ?? "—"}
              </p>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-3 gap-4">
              <Card title="Price" value={data?.close} />
              <Card title="21D Return" value={data?.ret_21d} />
              <Card title="Vol Ratio" value={data?.vol_ratio} />
            </div>

            {/* 🔥 Probability Bars */}
            {data?.prob_bull !== undefined && (
              <div className="bg-white border border-gray-200 rounded-xl p-4">

                <div className="text-sm text-gray-500 mb-3">
                  Model Probabilities
                </div>

                <div className="space-y-3">
                  <Bar label="Bull" value={data.prob_bull} />
                  <Bar label="Sideways" value={data.prob_sideways} />
                  <Bar label="Bear" value={data.prob_bear} />
                </div>

              </div>
            )}

            {/* 🔥 Confidence Meter */}
            {data?.confidence !== undefined && (
              <div className="bg-white border border-gray-200 rounded-xl p-4">

                <div className="text-sm text-gray-500 mb-2">
                  Confidence
                </div>

                <div className="w-full h-2 bg-gray-200 rounded">
                  <div
                    className="h-2 bg-black rounded"
                    style={{ width: `${data.confidence * 100}%` }}
                  />
                </div>

                <div className="text-xs text-gray-500 mt-1">
                  {(data.confidence * 100).toFixed(1)}%
                </div>
              </div>
            )}

            {/* Placeholder Chart */}
            <div className="bg-white border border-gray-200 p-4 rounded-xl h-80">
              <p className="text-gray-500 text-sm">
                Chart coming next…
              </p>
            </div>
          </>
        )}

      </div>

      {/* Right Panel */}
      <div className="w-72 bg-white border-l border-gray-200 p-4">
        <h2 className="font-semibold mb-4">Details</h2>

        {data ? (
          <div className="space-y-2 text-sm">
            <p><b>Date:</b> {data.date}</p>
            <p><b>Regime:</b> {data.regime}</p>
          </div>
        ) : (
          <p className="text-gray-400 text-sm">No data loaded</p>
        )}
      </div>

    </div>
  );
}

/* ---------- Components ---------- */

function Card({ title, value }: any) {
  return (
    <div className="bg-white border border-gray-200 p-4 rounded-xl">
      <p className="text-gray-500 text-xs">{title}</p>
      <p className="text-lg font-medium">
        {format(value)}
      </p>
    </div>
  );
}

function Bar({ label, value }: any) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span>{label}</span>
        <span>{(value * 100).toFixed(1)}%</span>
      </div>

      <div className="w-full h-2 bg-gray-200 rounded">
        <div
          className="h-2 bg-black rounded"
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );
}

/* ---------- Utils ---------- */

function format(val: any) {
  if (val === null || val === undefined) return "—";
  if (typeof val === "number") return val.toFixed(4);
  return val;
}