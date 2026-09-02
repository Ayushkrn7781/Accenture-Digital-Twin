import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, AlertTriangle, CheckCircle2, Cpu, FlaskConical, GitBranch, Info,
  Radio, ShieldCheck, Sigma, Thermometer, TrendingDown, Upload, Wrench, X
} from 'lucide-react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Line, LineChart,
  ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis
} from 'recharts';
import './styles.css';

const palette = { healthy: '#35d49a', warning: '#ffc65d', alert: '#fa667a', accent: '#62a5ff' };
const TICK_MS = 900;

/* Every value shown comes from the API. Where the backend reports null - a manual
   station has no health score, an upload arrived without ground truth - the UI says
   so rather than substituting a plausible-looking number. */
const Value = ({ v, unit = '', digits = 1, absent = 'n/a' }) =>
  v === null || v === undefined
    ? <span className="absent">{absent}</span>
    : <>{typeof v === 'number' ? Number(v.toFixed(digits)) : v}{unit && <span className="unit">{unit}</span>}</>;

const useJson = (url, deps = [], skip = false) => {
  const [data, setData] = useState(null);
  useEffect(() => {
    if (skip || !url) return;
    let live = true;
    fetch(url).then(r => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
      .then(d => live && setData(d)).catch(() => live && setData(null));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return data;
};

function App() {
  const [tab, setTab] = useState('floor');
  const [selected, setSelected] = useState(null);
  const [current, setCurrent] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [replayId, setReplayId] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [health, setHealth] = useState(null);
  const [evidence, setEvidence] = useState(null);

  const qs = replayId ? `&replay_id=${replayId}` : '';

  useEffect(() => {
    let stop = false;
    const poll = () => fetch('/api/health').then(r => r.json()).then(h => {
      if (stop) return;
      setHealth(h);
      if (!h.ready) setTimeout(poll, 700);
    }).catch(() => !stop && setTimeout(poll, 1200));
    poll();
    return () => { stop = true; };
  }, []);

  const ready = health?.ready;
  const state = useJson(ready ? `/api/state?at=${current}${qs}` : null, [current, replayId, ready], !ready);
  const validation = useJson(ready ? '/api/validation' : null, [ready], !ready);
  const economics = useJson(ready ? '/api/economics' : null, [ready], !ready);

  const stationId = selected || state?.stations?.find(s => s.instrumented)?.id;
  const series = useJson(
    ready && stationId ? `/api/stations/${stationId}/series?at=${current}&n=140${qs}` : null,
    [stationId, current, replayId, ready], !ready);

  useEffect(() => {
    if (!playing || !state) return;
    const t = setTimeout(
      () => setCurrent(v => (v < state.line.vehicle_count ? v + 1 : (setPlaying(false), v))), TICK_MS);
    return () => clearTimeout(t);
  }, [playing, state]);

  const upload = useCallback(async e => {
    e.preventDefault();
    const form = e.currentTarget;
    if (!form.data_file.files[0]) return setMessage('Choose a line-data CSV first.');
    setUploading(true); setMessage('');
    const body = new FormData();
    body.append('data_file', form.data_file.files[0]);
    if (form.ground_truth_file.files[0]) body.append('ground_truth_file', form.ground_truth_file.files[0]);
    try {
      const res = await fetch('/api/datasets/upload', { method: 'POST', body });
      const p = await res.json();
      if (!res.ok) throw new Error(p.detail || 'Upload failed');
      setReplayId(p.replay_id); setCurrent(1); setPlaying(false); setSelected(null);
      setMessage(`Replaying ${p.vehicle_count} vehicles${p.ground_truth_loaded
        ? ' with ground-truth validation.' : ' — no ground truth, so validation metrics stay unavailable.'}`);
    } catch (err) { setMessage(err.message); } finally { setUploading(false); }
  }, []);

  if (!ready) return (
    <main className="loading">
      <div className="spin" />
      <b>Warming the twin</b>
      <span>{health?.stage || 'connecting to backend…'}</span>
      <small>Simulating the line, calibrating detectors against fault-free runs, and scoring the risk model.</small>
    </main>
  );
  if (!state) return <main className="loading"><b>No state returned</b><span>Check the backend is running.</span></main>;

  const station = state.stations.find(s => s.id === stationId) || state.stations[0];
  const L = state.line;

  return (
    <main>
      <nav>
        <div className="brand">
          <span className="brand-mark">D</span>
          <span>DigitalTwin<span className="blue">.ai</span></span>
          {state.ml?.model && <span className="model-tag"><Cpu size={12} /> {state.ml.model}</span>}
        </div>
        <div className="tabs">
          {[['floor', 'Floor Supervisor'], ['manager', 'Plant Manager'], ['leadership', 'Leadership']]
            .map(([id, label]) => (
              <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{label}</button>
            ))}
        </div>
        <div className="live">
          <Radio size={14} className={playing ? 'pulsing' : ''} />
          {playing ? 'REPLAY RUNNING' : 'REPLAY PAUSED'}
        </div>
      </nav>

      <section className="hero">
        <div>
          <p className="eyebrow">{L.source} · {L.name}</p>
          <h1>Catch the constraint <em>before</em> it starves the line.</h1>
          <p className="sub">
            Contextual SPC, a physics-informed oven model, and a calibrated risk model across{' '}
            {L.total_stations} stations — {L.instrumented_stations} instrumented,{' '}
            {L.total_stations - L.instrumented_stations} on manual checks only.
            Control limits are derived from a measured false-alarm budget, not chosen.
          </p>
        </div>
        <div className="hero-stats-group">
          <Stat label="Line throughput" sub={L.constraint_station ? `Constraint: ${L.constraint_station}` : '—'}>
            <Value v={L.line_uph} unit=" UPH" />
          </Stat>
          <Stat label="Validated precision" sub="Anomaly, vs ground truth">
            <Value v={state.metrics?.precision} unit="%" />
          </Stat>
          <Stat label="False alarms" sub={`ARL0 ${validation?.false_alarms?.arl0_vehicles ?? '—'} vehicles`}
                tone={(validation?.false_alarms?.total_false_alarms_per_run ?? 0) > 2 ? 'warning' : 'healthy'}>
            <Value v={validation?.false_alarms?.total_false_alarms_per_run} digits={2} />
            <span className="unit"> /run</span>
          </Stat>
        </div>
      </section>

      <section className="replay card">
        <div className="replay-left">
          <div className="replay-header">
            <p className="eyebrow">Replay timeline</p>
            <b>Vehicle {L.current_vehicle} of {L.vehicle_count}</b>
            {L.baseline_warmup_vehicles && L.current_vehicle <= L.baseline_warmup_vehicles &&
              <span className="warm-tag">Phase&nbsp;I baseline window — detectors not yet armed</span>}
          </div>
          <input type="range" aria-label="Replay position" min="1" max={L.vehicle_count}
                 value={L.current_vehicle}
                 onChange={e => { setPlaying(false); setCurrent(Number(e.target.value)); }} />
        </div>
        <div className="replay-actions">
          <button className="primary" onClick={() => setPlaying(p => !p)}>{playing ? 'Pause' : 'Play'}</button>
          <button onClick={() => { setPlaying(false); setCurrent(1); }}>Reset</button>
        </div>
      </section>

      <form className="upload-bar" onSubmit={upload}>
        <Upload size={16} />
        <label>Line data CSV<input name="data_file" type="file" accept=".csv" /></label>
        <label>Ground truth <small>(optional)</small><input name="ground_truth_file" type="file" accept=".csv" /></label>
        <button className="primary" disabled={uploading}>{uploading ? 'Processing…' : 'Upload & replay'}</button>
        {replayId && <button type="button" onClick={() => { setReplayId(null); setCurrent(1); setMessage(''); }}>
          Back to simulation</button>}
        <span>{message}</span>
      </form>

      {tab === 'floor' && <Floor {...{ state, station, series, setSelected, setEvidence }} />}
      {tab === 'manager' && <Manager {...{ state, validation, setEvidence }} />}
      {tab === 'leadership' && <Leadership {...{ state, validation, economics }} />}

      {evidence && <EvidenceDrawer item={evidence} onClose={() => setEvidence(null)} thresholds={state.thresholds} />}

      <footer>
        <b>How to read this:</b> every number is computed from the run in view — none are
        hardcoded. Thresholds come from <code>calibrate.py</code>, which sweeps each limit
        against injection-free simulated lines until it meets a stated false-alarm budget.
        The twin is a read-only listener; nothing here writes to a PLC.
      </footer>
    </main>
  );
}

const Stat = ({ label, sub, tone, children }) => (
  <div className="hero-stat">
    <strong className={tone ? `text-${tone}` : ''}>{children}</strong>
    <span>{label}</span><small>{sub}</small>
  </div>
);

/* ---------------------------------------------------------------- Floor */

function Floor({ state, station, series, setSelected, setEvidence }) {
  const th = state.thresholds;
  const params = series?.series ? Object.keys(series.series) : [];
  const [param, setParam] = useState(null);
  const active = param && params.includes(param) ? param : params[0];
  const rows = active ? series.series[active] : [];
  const zs = rows.map(r => r.z);
  const bound = Math.max(th.alert_abs_z + 1, ...zs.map(Math.abs).concat([0])) * 1.1;
  const flagAt = new Set((series?.flags || []).map(f => f.detected_at));

  return (
    <section className="view">
      <div className="section-head">
        <div>
          <p className="eyebrow">Floor supervisor — right now</p>
          <h2>Line state at vehicle {state.line.current_vehicle}</h2>
        </div>
        <div className="legend">
          <i className="healthy" />Healthy<i className="warning" />Warning<i className="alert" />Alert
        </div>
      </div>

      <div className="line-map">
        {['Body construction', 'Paint', 'Final assembly'].map(area => (
          <div className="area" key={area}>
            <div className="area-title">
              <h3>{area}</h3>
              <span className="badge-count">{state.stations.filter(s => s.area === area).length} stations</span>
            </div>
            <div className="stations">
              {state.stations.filter(s => s.area === area).map(s => (
                <button key={s.id} onClick={() => setSelected(s.id)}
                        className={`station ${s.status}${station?.id === s.id ? ' selected' : ''}`}>
                  <div className="station-top"><b>{s.id}</b><span className="station-dot" /></div>
                  <span className="station-type">{s.instrumented ? 'Telemetry' : 'Manual QA'}</span>
                  {s.defect_risk_pct !== null && s.defect_risk_pct !== undefined && s.defect_risk_pct >= 20 &&
                    <span className="mini-risk-badge">{s.defect_risk_pct}%</span>}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="detail-grid">
        <article className="card">
          <p className="eyebrow">Selected station</p>
          <h2>{station.name} <span className="mono-id">({station.id})</span>
            <span className={`pill ${station.status}`}>{station.status}</span></h2>
          <p className="station-desc">
            {station.instrumented
              ? <>Continuous telemetry. Baselines are contextual per station, parameter and variant,
                  estimated from the first {state.line.baseline_warmup_vehicles} vehicles.</>
              : <>No continuous telemetry here. Status is derived via <code>{station.gap_method}</code>.</>}
          </p>

          <div className="risk-box">
            <div className="risk-head">
              <Sigma size={15} className="blue" />
              <b>Defect risk, next {state.ml?.horizon_vehicles ?? '—'} vehicles</b>
              <span className={`risk-val ${(station.defect_risk_pct ?? 0) > 60 ? 'high'
                : (station.defect_risk_pct ?? 0) > 25 ? 'med' : 'low'}`}>
                <Value v={station.defect_risk_pct} unit="%" absent="not scored" />
              </span>
            </div>
            {station.risk_drivers?.length > 0 ? (
              <div className="drivers">
                <p className="drivers-title">{state.ml?.attribution_method}</p>
                {station.risk_drivers.map(d => (
                  <div className="driver" key={d.feature}>
                    <span className="driver-label">{d.label}</span>
                    <span className="driver-family">{d.family}</span>
                    <span className={`driver-bar ${d.contribution >= 0 ? 'up' : 'down'}`}>
                      <i style={{ width: `${Math.min(100, Math.abs(d.contribution) * 45)}%` }} />
                    </span>
                    <span className="driver-num">{d.contribution >= 0 ? '+' : ''}{d.contribution}</span>
                  </div>
                ))}
              </div>
            ) : <p className="muted small">No attribution available for this cell.</p>}
          </div>

          <div className="mini-kpis">
            <Kpi label="Cycle time" v={station.operation?.cycle_time} unit="s" />
            <Kpi label="Buffer queue" v={station.operation?.queue} unit="" digits={0} />
            <Kpi label="Utilisation" v={station.operation?.utilization} unit="%" />
            <Kpi label="Health score" v={station.health_score} unit="" />
          </div>

          {station.open_flags?.length > 0 && (
            <div className="open-flags">
              <p className="eyebrow">Open alarms in the last 20 vehicles</p>
              {station.open_flags.map((f, i) => (
                <button className="flag-row" key={i} onClick={() => setEvidence(f)}>
                  <span className={`dot ${f.severity}`} />
                  <b>{f.type}</b><span>{f.parameter}</span>
                  <span className="mono">v{f.detected_at}</span>
                  <Info size={13} />
                </button>
              ))}
            </div>
          )}
        </article>

        <article className="card">
          <div className="chart-header">
            <p className="eyebrow">Control chart</p>
            <div className="param-tabs">
              {params.map(p => (
                <button key={p} className={p === active ? 'on' : ''} onClick={() => setParam(p)}>{p}</button>
              ))}
            </div>
          </div>
          {station.instrumented && rows.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={230}>
                <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
                  <CartesianGrid stroke="#1c2940" strokeDasharray="3 3" />
                  <XAxis dataKey="vehicle" stroke="#486581" fontSize={10} />
                  <YAxis domain={[-bound, bound]} stroke="#486581" fontSize={10} />
                  <Tooltip contentStyle={{ background: '#0b1322', borderColor: '#233854', borderRadius: 6 }}
                           formatter={(v, n) => [v, n === 'z' ? 'z-score' : n]} />
                  <ReferenceArea y1={-th.green_abs_z} y2={th.green_abs_z} fill="#35d49a" fillOpacity={0.05} />
                  {[th.alert_abs_z, -th.alert_abs_z].map(y => (
                    <ReferenceLine key={y} y={y} stroke={palette.alert} strokeDasharray="4 3"
                                   label={{ value: `${y > 0 ? '+' : ''}${y}σ`, fill: palette.alert, fontSize: 10 }} />
                  ))}
                  {[th.green_abs_z, -th.green_abs_z].map(y => (
                    <ReferenceLine key={y} y={y} stroke={palette.warning} strokeDasharray="2 4" strokeOpacity={0.6} />
                  ))}
                  <Line type="monotone" dataKey="z" stroke={palette.accent} strokeWidth={2} isAnimationActive={false}
                        dot={({ cx, cy, payload }) => flagAt.has(payload.vehicle)
                          ? <circle key={payload.vehicle} cx={cx} cy={cy} r={3.5} fill={palette.alert} />
                          : null} />
                </LineChart>
              </ResponsiveContainer>
              <p className="chart-sub">
                Limits from config: warning ±{th.green_abs_z}σ, alert ±{th.alert_abs_z}σ.
                Red points mark where a rule fired. Baseline: {rows[0]?.baseline_source}.
              </p>
            </>
          ) : (
            <div className="no-signal">
              <ShieldCheck size={30} />
              <b>{station.instrumented ? 'No readings yet' : 'Manual QA station'}</b>
              <span>{station.instrumented
                ? 'Replay further to accumulate readings.'
                : <>No continuous telemetry. Gap handled via <code>{station.gap_method}</code>.</>}</span>
            </div>
          )}
        </article>
      </div>
    </section>
  );
}

const Kpi = ({ label, v, unit, digits }) => (
  <div className="kpi-item"><small>{label}</small><b><Value v={v} unit={unit} digits={digits} /></b></div>
);

/* -------------------------------------------------------------- Manager */

function Manager({ state, validation, setEvidence }) {
  const bn = state.bottlenecks || [];
  const loss = state.throughput_loss || [];
  const imp = Object.entries(state.ml?.feature_importances || {})
    .map(([k, v]) => ({ name: state.ml?.feature_labels?.[k] || k, v }))
    .filter(d => d.v > 0).sort((a, b) => b.v - a.v).slice(0, 8);

  // Real station x parameter matrix, from the alarms actually open.
  const params = [...new Set(state.flags.map(f => f.parameter).filter(p => p && !p.includes('+')))];
  const stations = [...new Set(state.flags.map(f => f.station))];
  const cell = {};
  state.flags.forEach(f => {
    const k = `${f.station}|${f.parameter}`;
    cell[k] = Math.max(cell[k] || 0, Math.abs(f.z ?? 0));
  });

  return (
    <section className="view">
      <div className="section-head">
        <div>
          <p className="eyebrow">Plant manager — shift &amp; capacity</p>
          <h2>Where the line is losing throughput</h2>
        </div>
      </div>

      <div className="manager-grid">
        <article className="card wide">
          <div className="chart-header">
            <p className="eyebrow">Confirmed constraints</p>
            <span className="tag">{bn.length} detected · CUSUM-confirmed</span>
          </div>
          {bn.length ? (
            <div className="bn-list">
              {bn.map((b, i) => (
                <button className="bn-row" key={i} onClick={() => setEvidence(b)}>
                  <span className="bn-station">{b.station}</span>
                  <span className="bn-detail">
                    cycle <b>{b.cycle_time?.toFixed(1)}s</b> vs baseline {b.baseline_cycle}s
                  </span>
                  <span className="bn-chain">
                    <TrendingDown size={13} /> starves {b.starving_downstream?.join(', ') || '—'}
                  </span>
                  <span className="mono">v{b.detected_at}</span>
                  <span className="bn-stat">CUSUM {b.cusum}σ · t={b.slope_t}</span>
                </button>
              ))}
            </div>
          ) : <p className="muted">No constraint confirmed yet at this point in the replay.</p>}

          {loss.length > 0 && (
            <>
              <p className="eyebrow mt">Cycle time above nominal</p>
              <ResponsiveContainer width="100%" height={170}>
                <BarChart data={loss} margin={{ top: 4, right: 8, bottom: 4, left: -20 }}>
                  <CartesianGrid stroke="#1c2940" strokeDasharray="3 3" />
                  <XAxis dataKey="station" stroke="#486581" fontSize={10} />
                  <YAxis stroke="#486581" fontSize={10} unit="s" />
                  <Tooltip contentStyle={{ background: '#0b1322', borderColor: '#233854' }} />
                  <Bar dataKey="excess_s" radius={[3, 3, 0, 0]}>
                    {loss.map((d, i) => (
                      <Cell key={i} fill={bn.some(b => b.station === d.station) ? palette.alert : palette.warning} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="caption">
                Line rate is set by the slowest station, so this is the queue of candidates
                for the constraint — not an average across the line.
              </p>
            </>
          )}
        </article>

        <article className="card">
          <p className="eyebrow">Risk model drivers</p>
          <p className="feat-sub">{state.ml?.attribution_method || 'Attribution unavailable'}</p>
          {imp.length ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={imp} layout="vertical" margin={{ left: 8, right: 14 }}>
                <CartesianGrid stroke="#1c2940" horizontal={false} />
                <XAxis type="number" stroke="#486581" unit="%" fontSize={10} />
                <YAxis dataKey="name" type="category" stroke="#90a2bd" width={150} fontSize={10} />
                <Tooltip contentStyle={{ background: '#0b1322', borderColor: '#233854' }} />
                <Bar dataKey="v" fill={palette.accent} radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <p className="muted">Model unavailable.</p>}
        </article>
      </div>

      <div className="manager-grid-bottom">
        <article className="card">
          <p className="eyebrow">Station health</p>
          <div className="health-list">
            {state.stations.map(s => (
              <div className="health-row" key={s.id}>
                <span className="station-lbl">{s.id}</span>
                <div className="bar-container">
                  {s.health_score === null
                    ? <span className="bar-absent">no telemetry</span>
                    : <i style={{ width: `${s.health_score}%`, background: palette[s.status] }} />}
                </div>
                <b className="health-score"><Value v={s.health_score} absent="—" /></b>
                <span className="risk-pill"><Value v={s.defect_risk_pct} unit="%" absent="—" /></span>
              </div>
            ))}
          </div>
        </article>

        <article className="card">
          <p className="eyebrow">Station × parameter deviation</p>
          {stations.length ? (
            <div className="heat-wrap">
              <table className="heat-table">
                <thead><tr><th />{params.map(p => <th key={p}>{p}</th>)}</tr></thead>
                <tbody>
                  {stations.map(st => (
                    <tr key={st}>
                      <th>{st}</th>
                      {params.map(p => {
                        const v = cell[`${st}|${p}`];
                        const a = v ? Math.min(1, v / 6) : 0;
                        return <td key={p} title={v ? `|z| ${v.toFixed(2)}` : 'no alarm'}
                                   style={{ background: v ? `rgba(250,102,122,${0.15 + a * 0.75})` : 'transparent' }}>
                          {v ? v.toFixed(1) : ''}</td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p className="muted">No alarms open yet.</p>}
          <p className="caption">Cell intensity is peak |z| at the alarm. Blank means no rule fired.</p>
        </article>
      </div>

      {validation?.propagation?.available && (
        <article className="card">
          <p className="eyebrow">Latent defect back-trace</p>
          <h3>A defect that never alarmed where it was created</h3>
          <p className="scale-p">
            {validation.propagation.failures_analysed} vehicles failed at{' '}
            <b>{validation.propagation.inspection_station}</b>. No upstream station broke its own
            control limit, so per-vehicle SPC cannot attribute this. Comparing upstream signal
            populations for failing versus time-matched passing vehicles can.
          </p>
          <div className="trace-list">
            {validation.propagation.ranking.slice(0, 5).map((r, i) => (
              <div className={`trace-row${i === 0 ? ' top' : ''}`} key={`${r.station}${r.parameter}`}>
                <span className="rank">{i + 1}</span>
                <b>{r.station}</b><span>{r.parameter}</span>
                <span className="mono">t = {r.t_statistic}</span>
                <span className="muted small">effect {r.effect_size_sigma}σ</span>
                {i === 0 && validation.propagation.origin_identified &&
                  <span className="ok-tag"><CheckCircle2 size={12} /> matches true origin</span>}
              </div>
            ))}
          </div>
          <p className="caption">{validation.propagation.method}</p>
        </article>
      )}
    </section>
  );
}

/* ----------------------------------------------------------- Leadership */

function Leadership({ state, validation, economics }) {
  const bt = validation?.backtest || state.metrics || {};
  const fa = validation?.false_alarms;
  const mlv = validation?.ml;
  const ec = economics?.economics;
  const ro = economics?.rollout;
  const rf = economics?.retrofit;
  const gap = validation?.sensor_gap;
  const cal = validation?.calibration;
  const money = n => n === null || n === undefined ? '—' : `$${Math.round(n).toLocaleString()}`;

  return (
    <section className="view">
      <div className="section-head">
        <div>
          <p className="eyebrow">Leadership — evidence &amp; investment case</p>
          <h2>What is measured, and what it is worth</h2>
        </div>
      </div>

      {!bt.validation_available && (
        <div className="notice"><AlertTriangle size={16} />
          <span>{bt.reason || 'Validation metrics unavailable for this dataset.'}</span></div>
      )}

      <div className="metrics">
        <Metric v={bt.precision} unit="%" label="Anomaly precision" note="Flagged stations that were genuinely faulty" />
        <Metric v={bt.recall} unit="%" label="Anomaly recall" note="Injected faults recovered" />
        <Metric v={bt.bottleneck_precision} unit="%" label="Bottleneck precision" note="Constraints confirmed, not guessed" />
        <Metric v={bt.median_detection_lag} unit=" veh" label="Median lead time" note={`p90 ${bt.p90_detection_lag ?? '—'} vehicles`} />
      </div>

      <div className="detail-grid">
        <article className="card">
          <p className="eyebrow">False-alarm floor</p>
          <h3>Measured, not assumed</h3>
          {fa ? (
            <>
              <div className="fa-grid">
                <div><b><Value v={fa.total_false_alarms_per_run} digits={2} /></b><span>per {fa.vehicles_per_run}-vehicle run</span></div>
                <div><b><Value v={fa.arl0_vehicles} /></b><span>ARL₀ (vehicles to first false alarm)</span></div>
                <div><b><Value v={fa.false_alarms_per_1000_vehicles} digits={2} /></b><span>per 1,000 vehicles</span></div>
              </div>
              <ul className="fa-split">
                <li>Anomaly <b><Value v={fa.anomaly_false_alarms_per_run} digits={2} /></b></li>
                <li>Bottleneck <b><Value v={fa.bottleneck_false_alarms_per_run} digits={2} /></b></li>
                <li>Physics <b><Value v={fa.physics_false_alarms_per_run} digits={2} /></b></li>
              </ul>
              <p className="caption">{fa.method} Measured over {fa.seeds} independent runs.</p>
            </>
          ) : <p className="muted">Unavailable.</p>}
          {cal && (
            <div className="calib">
              <p className="eyebrow">Calibrated limits</p>
              <p className="caption">
                EWMA L = <b>{cal.ewma_L}</b> · CUSUM h = <b>{cal.cusum_h_sigma}σ</b> ·
                oven τ z = <b>{cal.tau_fit_z_crit}</b> — chosen by sweeping each limit against{' '}
                {cal.seeds} fault-free lines to a budget of {cal.budget_false_alarms_per_run}/run,
                then confirmed on {cal.validation_seeds} lines held out of that sweep.
              </p>
            </div>
          )}
        </article>

        <article className="card">
          <p className="eyebrow">Risk model validation</p>
          <h3>Held-out lines only</h3>
          {mlv ? (
            <>
              <div className="ml-grid">
                <div><b><Value v={mlv.pr_auc} digits={3} /></b><span>PR-AUC</span></div>
                <div><b><Value v={mlv.pr_auc_baseline} digits={3} /></b><span>No-skill baseline</span></div>
                <div><b>{mlv.lift_over_baseline}×</b><span>Lift</span></div>
                <div><b><Value v={mlv.brier_score} digits={4} /></b><span>Brier score</span></div>
              </div>
              <p className="caption">
                {mlv.held_out_rows?.toLocaleString()} rows, {mlv.positive_rate_pct}% positive.
                Precision@40 = {mlv.precision_at_40}. ROC-AUC ({mlv.roc_auc}) is shown for
                completeness only — at a {mlv.positive_rate_pct}% positive rate it flatters
                weak models, which is why PR-AUC is the headline.
              </p>
              {mlv.reliability?.length > 0 && (
                <>
                  <p className="eyebrow mt">Calibration reliability</p>
                  <ResponsiveContainer width="100%" height={150}>
                    <LineChart data={mlv.reliability} margin={{ top: 6, right: 10, bottom: 2, left: -22 }}>
                      <CartesianGrid stroke="#1c2940" strokeDasharray="3 3" />
                      <XAxis dataKey="predicted" stroke="#486581" fontSize={10} domain={[0, 1]} type="number" />
                      <YAxis stroke="#486581" fontSize={10} domain={[0, 1]} />
                      <Tooltip contentStyle={{ background: '#0b1322', borderColor: '#233854' }} />
                      <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#486581" strokeDasharray="4 4" />
                      <Line dataKey="observed" stroke={palette.healthy} strokeWidth={2} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="caption">Predicted probability vs observed frequency. Diagonal is perfect.</p>
                </>
              )}
              <p className="prov">{validation?.ml_provenance?.training_provenance}</p>
            </>
          ) : <p className="muted">Model unavailable.</p>}
        </article>
      </div>

      {gap?.levels && (
        <article className="card">
          <p className="eyebrow">Sensor coverage study</p>
          <h3>How much detection costs when instrumentation is missing</h3>
          <div className="table-wrapper">
            <table className="roadmap-table">
              <thead><tr>
                <th>Coverage</th><th>Instrumented</th><th>Manual</th><th>Anomaly recall</th>
                <th>Precision</th><th>Bottleneck recall</th><th>Median lead</th><th>Back-trace</th>
              </tr></thead>
              <tbody>
                {gap.levels.map(r => (
                  <tr key={r.instrumented_ratio}>
                    <td><b>{Math.round(r.instrumented_ratio * 100)}%</b></td>
                    <td>{r.instrumented_stations}</td><td>{r.manual_stations}</td>
                    <td><Value v={r.recall} unit="%" /></td>
                    <td><Value v={r.precision} unit="%" /></td>
                    <td><Value v={r.bottleneck_recall} unit="%" /></td>
                    <td><Value v={r.median_detection_lag} unit=" veh" /></td>
                    <td>{r.latent_origin_identified
                      ? <span className="ok-tag"><CheckCircle2 size={12} /> found</span>
                      : <span className="absent">lost</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="caption">{gap.note}</p>
        </article>
      )}

      <div className="detail-grid">
        <article className="card">
          <p className="eyebrow">Business case</p>
          <h3>Computed from the metrics above</h3>
          {ec ? (
            <>
              <div className="econ-head">
                <div><b>{money(ec.annual.net_benefit_usd)}</b><span>Net annual benefit, one line</span></div>
                <div className="band">
                  {money(ec.sensitivity.net_benefit_low_usd)} – {money(ec.sensitivity.net_benefit_high_usd)}
                  <small>±{ec.sensitivity.band_pct}% on {ec.sensitivity.dominant_assumptions.join(' and ')}</small>
                </div>
              </div>
              <table className="econ-table"><tbody>
                <tr><td>Defects prevented</td><td>{ec.annual.defects_prevented.toLocaleString()}</td></tr>
                <tr><td>Rework avoided</td><td>{money(ec.annual.rework_avoided_usd)}</td></tr>
                <tr><td>Scrap avoided</td><td>{money(ec.annual.scrap_avoided_usd)}</td></tr>
                <tr><td>Throughput recovered</td><td>{money(ec.annual.throughput_value_usd)}</td></tr>
                <tr className="sub"><td>False-alarm investigations</td><td>−{ec.annual.false_alarm_investigations.toLocaleString()} events</td></tr>
                <tr className="sub"><td>Investigation cost</td><td>−{money(ec.annual.investigation_cost_usd)}</td></tr>
              </tbody></table>
              <p className="caption">
                Benefit scales with the measured recall ({ec.derived_from_measurement.anomaly_recall_pct}%)
                and median lead time ({ec.derived_from_measurement.median_detection_lead_vehicles} vehicles),
                of which {Math.round(ec.derived_from_measurement.actionable_fraction * 100)}% is treated as
                actionable. Improving the detector moves this number; a noisy one is charged for its noise.
              </p>
            </>
          ) : <p className="muted">Unavailable.</p>}
        </article>

        <article className="card">
          <p className="eyebrow">Rollout</p>
          <h3>Phased deployment</h3>
          {ro ? (
            <>
              <ol className="roadmap">
                {ro.phases.map(p => (
                  <li key={p.phase}>
                    <b>{String(p.phase).padStart(2, '0')} · {p.scope}</b>
                    <span>{p.duration_weeks} weeks · {money(p.cost_estimate_usd)}</span>
                  </li>
                ))}
              </ol>
              <div className="roi-highlight">
                <Activity size={16} className="blue" />
                <span>
                  <b>{money(ro.total_implementation_cost_usd)}</b> over {ro.total_duration_weeks} weeks ·
                  payback <b>{ro.payback_months ?? '—'} months</b> · first-year ROI{' '}
                  <b>{ro.first_year_roi_pct ?? '—'}%</b>
                </span>
              </div>
              <div className="constraint">
                <AlertTriangle size={18} />
                <div><b>Operational constraint:</b> read-only listener, no PLC writes. Instrumentation
                  changes are confined to the {state.line.scheduled_sensor_change_windows_per_year} scheduled
                  maintenance windows per year.</div>
              </div>
            </>
          ) : <p className="muted">Unavailable.</p>}
        </article>
      </div>

      {rf?.items && (
        <article className="card">
          <p className="eyebrow">Sensor retrofit plan</p>
          <h3>Priced against the coverage study, not asserted</h3>
          <div className="table-wrapper">
            <table className="roadmap-table">
              <thead><tr>
                <th>Stations</th><th>Sensor</th><th>Capex</th><th>Window</th>
                <th>Est. annual value</th><th>Payback</th><th>Benefit</th>
              </tr></thead>
              <tbody>
                {rf.items.map((r, i) => (
                  <tr key={i}>
                    <td><b>{r.target_stations.join(', ')}</b></td>
                    <td><span className="sensor-badge">{r.sensor_type}</span></td>
                    <td>{money(r.total_cost_usd)}</td>
                    <td><span className="window-badge">{r.install_window}</span></td>
                    <td>{money(r.estimated_annual_value_usd)}</td>
                    <td>{r.payback_months ? `${r.payback_months} mo` : '—'}</td>
                    <td className="benefit-cell">{r.benefit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="caption">{rf.basis} Total capex {money(rf.total_capex_usd)}.</p>
        </article>
      )}

      <article className="card limitations">
        <p className="eyebrow">Assumptions &amp; limitations</p>
        <ul>
          <li><b>This is a prototype on simulated data.</b> It validates the <em>method</em>, not the model
            on a real plant. Every metric here comes from synthetic lines generated by this repository.</li>
          <li><b>Unit economics are stated assumptions</b>, declared in <code>config/economics.json</code>,
            not observations from any specific company. Only the detector performance feeding them is measured.</li>
          <li><b>The risk model can learn the simulator.</b> Fault station, kind, onset and magnitude are
            randomised per training line to suppress memorisation, but synthetic-to-real transfer is unproven.</li>
          <li><b>Baselines need a fault-free Phase&nbsp;I window.</b> The first{' '}
            {state.line.baseline_warmup_vehicles} vehicles establish contextual limits; a fault present
            throughout that window would be absorbed into the baseline.</li>
          <li><b>No PLC or OT system is connected.</b> Ingestion is read-only CSV replay.</li>
        </ul>
      </article>
    </section>
  );
}

const Metric = ({ v, unit, label, note }) => (
  <article className="metric card">
    <strong><Value v={v} unit={unit} absent="n/a" /></strong>
    <b>{label}</b><span>{note}</span>
  </article>
);

/* ------------------------------------------------------------- Evidence */

function EvidenceDrawer({ item, onClose, thresholds }) {
  const isBn = 'cusum' in item;
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={e => e.stopPropagation()}>
        <div className="drawer-head">
          <div>
            <p className="eyebrow">Why this fired</p>
            <h3>{item.station} · {isBn ? 'bottleneck' : item.type}</h3>
          </div>
          <button className="icon" onClick={onClose} aria-label="Close"><X size={18} /></button>
        </div>
        <dl className="drawer-rows">
          <dt>Detected at</dt><dd>Vehicle {item.detected_at}</dd>
          {item.parameter && <><dt>Parameter</dt><dd>{item.parameter}</dd></>}
          {item.severity && <><dt>Severity</dt><dd><span className={`pill ${item.severity}`}>{item.severity}</span></dd></>}
          {isBn && <>
            <dt>CUSUM</dt><dd>{item.cusum}σ against limit {item.cusum_limit}σ</dd>
            <dt>Slope test</dt><dd>t = {item.slope_t}</dd>
            <dt>Queue</dt><dd>z = {item.queue_z}</dd>
            <dt>Starves</dt><dd>{item.starving_downstream?.join(', ') || '—'}</dd>
          </>}
          {!isBn && item.z !== undefined && <><dt>Statistic</dt><dd>{item.z}</dd></>}
          {!isBn && thresholds && <><dt>Limits</dt>
            <dd>warning ±{thresholds.green_abs_z}σ · alert ±{thresholds.alert_abs_z}σ</dd></>}
        </dl>
        <p className="drawer-evidence">{item.evidence}</p>
        <p className="caption">
          Limits are set by <code>calibrate.py</code> from a measured false-alarm budget on
          fault-free runs — not chosen by hand.
        </p>
      </aside>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
