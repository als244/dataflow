import { useMemo, useState, type ChangeEvent } from "react";
import type { MeasuredRun, TaskInterval } from "../types";
import { ComputeTimeline } from "./ComputeTimeline";
import { MemoryTimelinePanel } from "./MemoryTimelinePanel";

/** Real hardware runs: import a `dataflow-measured-run/v1` file (exported by
 * the runtime's export_measured_run tool) and render the MEASURED timeline
 * with the same panels used for simulations, side-by-side with the
 * simulator's prediction for the identical annotated plan. Everything is
 * client-side — the file embeds both event logs. */

function fmtMs(us: number): string {
  return us >= 1_000_000 ? `${(us / 1_000_000).toFixed(2)} s` : `${(us / 1_000).toFixed(1)} ms`;
}

function Delta({ real, sim, invert }: { real: number; sim: number; invert?: boolean }) {
  if (!sim) return null;
  const pct = (real / sim - 1) * 100;
  const good = invert ? pct <= 0 : pct >= 0;
  return (
    <span className={good ? "measured-delta-good" : "measured-delta-bad"}>
      {pct >= 0 ? "+" : ""}
      {pct.toFixed(1)}% vs sim
    </span>
  );
}

interface Row {
  taskId: string;
  simUs: number;
  realUs: number;
}

function taskDeltas(real: TaskInterval[], sim: TaskInterval[]): Row[] {
  const simDur = new Map<string, number>();
  for (const iv of sim) {
    if (iv.track === "compute") simDur.set(iv.task_id, iv.end - iv.start);
  }
  const rows: Row[] = [];
  for (const iv of real) {
    if (iv.track !== "compute") continue;
    const s = simDur.get(iv.task_id);
    if (s === undefined) continue;
    rows.push({ taskId: iv.task_id, simUs: s, realUs: iv.end - iv.start });
  }
  rows.sort((a, b) => Math.abs(b.realUs - b.simUs) - Math.abs(a.realUs - a.simUs));
  return rows.slice(0, 20);
}

export function MeasuredRunImport({
  run,
  onLoad,
}: {
  run: MeasuredRun | null;
  onLoad: (run: MeasuredRun | null) => void;
}) {
  const [error, setError] = useState<string | null>(null);

  async function onImport(e: ChangeEvent<HTMLInputElement>) {
    const input = e.currentTarget;
    const file = input.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text()) as MeasuredRun;
      if (parsed.format !== "dataflow-measured-run/v1") {
        setError('expected "format": "dataflow-measured-run/v1"');
        return;
      }
      setError(null);
      onLoad(parsed);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      input.value = "";
    }
  }

  return (
    <section className="panel">
      <div className="panel-title-row">
        <h2>Measured Run (real hardware)</h2>
        <label className="reset-btn schema-import-btn">
          Import measured run
          <input type="file" accept="application/json,.json" onChange={onImport} />
        </label>
        {run && (
          <button className="reset-btn" onClick={() => onLoad(null)}>
            Clear
          </button>
        )}
      </div>
      {error && <div className="input-error">measured run: {error}</div>}
      {!run && !error && (
        <p className="dim">
          Upload a <code>*.measured.json</code> exported by the dataflow runtime
          (<code>tools/export_measured_run.py</code>) to see real timings rendered like a
          simulation, next to the simulator&apos;s prediction of the same plan. Results
          open in the main window.
        </p>
      )}
      {run && (
        <p className="dim">
          Loaded: {String(run.meta.config ?? run.meta.name ?? "run")}
          {run.meta.budget_gib ? ` @ ${run.meta.budget_gib} GiB` : ""} — see the main
          window for the summary and timeline comparison.
        </p>
      )}
    </section>
  );
}

export function MeasuredRunResults({ run }: { run: MeasuredRun }) {
  const [hoverTaskId, setHoverTaskId] = useState<string | null>(null);

  const totalDuration = useMemo(() => {
    const ends = [...run.log.task_intervals, ...run.sim_log.task_intervals].map((iv) => iv.end);
    return Math.max(...ends, 1);
  }, [run]);

  const deltas = useMemo(
    () => taskDeltas(run.log.task_intervals, run.sim_log.task_intervals),
    [run],
  );

  return (
    <section className="panel">
      <div className="panel-title-row">
        <h2>Measured Run vs Simulation</h2>
      </div>
      <p className="dim">
            {String(run.meta.config ?? run.meta.name ?? "run")}
            {run.meta.budget_gib ? ` @ ${run.meta.budget_gib} GiB` : ""}
            {run.meta.kernel_set ? ` · kernels: ${run.meta.kernel_set}` : ""}
            {run.meta.device ? ` · ${run.meta.device}` : ""}
          </p>

          <div className="summary-grid">
            <div className="summary-stat">
              <div className="summary-stat-label">Real makespan</div>
              <div className="summary-stat-value">{fmtMs(run.summary.makespan_us)}</div>
              <div className="summary-stat-sub">
                sim {fmtMs(run.sim_summary.makespan_us)}{" "}
                <Delta real={run.summary.makespan_us} sim={run.sim_summary.makespan_us} invert />
              </div>
            </div>
            <div className="summary-stat">
              <div className="summary-stat-label">Real tokens/s</div>
              <div className="summary-stat-value">
                {run.summary.tokens_per_second.toFixed(0)}
              </div>
              <div className="summary-stat-sub">
                sim {run.sim_summary.tokens_per_second.toFixed(0)}{" "}
                <Delta
                  real={run.summary.tokens_per_second}
                  sim={run.sim_summary.tokens_per_second}
                />
              </div>
            </div>
            <div className="summary-stat">
              <div className="summary-stat-label">Peak fast memory</div>
              <div className="summary-stat-value">
                {run.summary.peak_fast_memory_gb.toFixed(2)} GB
              </div>
              <div className="summary-stat-sub">
                sim {run.sim_summary.peak_fast_memory_gb.toFixed(2)} GB
              </div>
            </div>
            <div className="summary-stat">
              <div className="summary-stat-label">Compute idle</div>
              <div className="summary-stat-value">{run.summary.idle_pct.toFixed(1)}%</div>
              <div className="summary-stat-sub">sim {run.sim_summary.idle_pct.toFixed(1)}%</div>
            </div>
            <div className="summary-stat">
              <div className="summary-stat-label">PCIe util (in / out)</div>
              <div className="summary-stat-value">
                {run.summary.from_slow_util_pct.toFixed(0)}% /{" "}
                {run.summary.to_slow_util_pct.toFixed(0)}%
              </div>
              <div className="summary-stat-sub">
                sim {run.sim_summary.from_slow_util_pct.toFixed(0)}% /{" "}
                {run.sim_summary.to_slow_util_pct.toFixed(0)}%
              </div>
            </div>
          </div>

          <h3 className="measured-subhead">Measured timeline</h3>
          <ComputeTimeline
            intervals={run.log.task_intervals}
            currentT={0}
            totalDuration={totalDuration}
            activeTaskId={null}
            hoverTaskId={hoverTaskId}
            onHoverTask={setHoverTaskId}
          />

          <details className="collapsible-panel" open>
            <summary className="collapsible-summary">
              Simulator prediction (same plan, same time scale)
            </summary>
            <div className="collapsible-content">
              <ComputeTimeline
                intervals={run.sim_log.task_intervals}
                currentT={0}
                totalDuration={totalDuration}
                activeTaskId={null}
                hoverTaskId={hoverTaskId}
                onHoverTask={setHoverTaskId}
              />
            </div>
          </details>

          {run.log.memory_trace.length > 0 && (
            <MemoryTimelinePanel
              log={run.log}
              fastMemoryCapacityGb={
                typeof run.meta.budget_gib === "number"
                  ? run.meta.budget_gib * 1.073741824
                  : run.summary.peak_fast_memory_gb
              }
              currentT={null}
            />
          )}

          {deltas.length > 0 && (
            <details className="collapsible-panel">
              <summary className="collapsible-summary">
                Largest per-task deviations from prediction (top {deltas.length})
              </summary>
              <div className="collapsible-content">
                <table className="measured-delta-table">
                  <thead>
                    <tr>
                      <th>task</th>
                      <th>sim</th>
                      <th>measured</th>
                      <th>Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {deltas.map((r) => (
                      <tr key={r.taskId}>
                        <td>{r.taskId}</td>
                        <td>{fmtMs(r.simUs)}</td>
                        <td>{fmtMs(r.realUs)}</td>
                        <td>
                          {r.realUs >= r.simUs ? "+" : ""}
                          {((r.realUs / r.simUs - 1) * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
    </section>
  );
}
