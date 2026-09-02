# DigitalTwin.ai

A predictive digital twin for a mixed-model vehicle assembly line, built to work with
the uneven sensor coverage a real plant actually has rather than an ideal instrumented
factory. It detects emerging constraints before they starve downstream stations,
attributes defects that surface far from where they were created, and prices its own
business case from measured detector performance.

**Every number in the dashboard is computed from the run in view.** No metric, risk
score, ROI figure or control limit is hardcoded, and a test suite enforces that.

---

## What it does

| Capability | How |
|---|---|
| **Anomaly detection** | Contextual SPC over (station, parameter, variant) baselines: confirmed point rule, EWMA for sustained shifts, monotone trend, cross-parameter correlation. All alarms latch, so one fault is one event. |
| **Constraint detection** | Page CUSUM on cycle time, confirmed by a slope *t*-test or a baseline-relative queue excursion, with the starvation chain from line topology. |
| **Physics-informed check** | Fits the paint oven's thermal time constant across all paint stations and charts its drift — catching a fouling oven whose individual stations all stay inside their own control limits. |
| **Defect risk** | Gradient-boosted model over 17 causal features, calibrated to a real probability, with exact TreeSHAP attribution per prediction. |
| **Latent-defect back-trace** | Attributes inspection failures to the upstream station that caused them, using time-detrended signal populations of failing vs time-matched passing vehicles. |
| **Sensor-gap handling** | Manual-only stations report outcomes, never invented readings; status falls back to adjacent-signal inference, and the cost of missing instrumentation is measured, not assumed. |
| **Business case** | Computed from measured recall, lead time and false-alarm rate — including the investigation cost of the false alarms the system itself generates. |

---

## Current measured results

Regenerate with `python run_backtest.py`. These are outputs, not claims.

**Detection** (seed 42, against injected ground truth)

| | |
|---|---|
| Anomaly precision / recall | **90.0% / 100.0%** |
| Bottleneck precision / recall | **50.0% / 100.0%** |
| Median / p90 detection lead | **17.5 / 31.0 vehicles** |
| Events detected | **7 of 7** |
| Latent-defect origin | **BIW-09, correctly identified** |

**False-alarm floor** — measured on injection-free runs, where every alarm is false by
construction:

| | |
|---|---|
| Total | **1.88 per 420-vehicle run** |
| ARL₀ | **224 vehicles** to first false alarm |
| Per 1,000 vehicles | **4.46** |

**Risk model** — `XGBoost 3.2.0`, evaluated only on lines used for neither training
nor calibration:

| | |
|---|---|
| PR-AUC | **0.659** against a no-skill baseline of **0.014** (47× lift) |
| Brier score | **0.0071** |
| Precision@40 | **1.00** |

**Throughput** — bottleneck-governed: **42.6 UPH** against a nominal 62.1, constraint
at **PNT-08**.

**Latent defect exposure** — 14 vehicles shipped carrying the defect before local SPC
raised anything at the origin; the back-trace names the station from the failure
population instead of waiting.

---

## Why the thresholds are what they are

No control limit is chosen by hand. `calibrate.py` sweeps each one against
injection-free simulated lines and picks the tightest limit that still meets a stated
false-alarm budget — then re-checks it on lines held out of the sweep, because a limit
fitted to the same runs it is scored on is optimistically biased.

```bash
python calibrate.py --seeds 14 --budget 1.0
```

Current calibrated limits: EWMA `L = 4.5`, CUSUM `h = 7.75σ`, oven-τ `z = 3.3`, with
the full provenance record written into `config/line_config.json`.

This is the direct answer to the operational risk that false alarms erode floor trust:
here that risk is a measured number with a stated method, and the thresholds are set
by it.

---

## Run it

Backend (from `backend/`):

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

First start trains the risk model (~70 s) and caches it under `backend/.cache/`;
subsequent starts are fast. `/api/health` reports warm-up progress.

Frontend (from `frontend/`):

```bash
npm install   # or: corepack pnpm install
npm run dev
```

Open the URL Vite prints (normally `http://localhost:5173`).

### Validation report

```bash
cd backend
python run_backtest.py              # human-readable
python run_backtest.py --json       # machine-readable
python run_backtest.py --config config/line_config_plant_b.json
```

### Tests

```bash
cd backend && python -m pytest
```

63 tests covering causality invariants, detector behaviour, generalisation to a second
plant, and guards against hardcoded values returning.

---

## Replaying your own data

Choose a **line data CSV**, optionally a **ground truth CSV**, and press *Upload &
replay*. Use [`line_data_template.csv`](backend/sample_data/line_data_template.csv) as
the column contract; `outputs/benchmark_line_data.csv` is a full 420-vehicle example
generated by `python export_benchmark_dataset.py`.

Manual-only stations leave `parameter` and `value` blank and supply `manual_outcome`
(`pass` / `fail` / `rework`). Station IDs and parameters must exist in the line
configuration — unknown identifiers are rejected rather than guessed.

Without a ground-truth file the dashboard marks validation metrics **unavailable**
rather than borrowing the simulation's figures. Uploaded data stays in process memory
and the integration is read-only.

---

## Architecture

```
config/line_config.json
        │
        ├── twin.simulate()            synthetic line, latent causal factors, injected faults
        │   or ingestion.load_*_csv()  read-only replay of a historical extract
        ▼
   twin.build_baselines()              contextual (station, parameter, variant) limits
        ▼
   twin.analyze()                      SPC · CUSUM constraints · physics · back-trace
        ▼
   metrics · ml_risk · economics       backtest, false-alarm floor, risk model, business case
        ▼
   state.build_state()                 causal slicing by detected_at
        ▼
   main.py (FastAPI)  →  React dashboard
```

### Two invariants the code guarantees

1. **Causality.** Every derived artifact carries `detected_at` — the vehicle at which
   an online detector could first have emitted it — and all slicing is bounded on both
   sides. `state(t)` can never contain something detected after `t`, and CI asserts it
   at eleven checkpoints across the replay.
2. **Nothing displayed that isn't computed.** Where the backend has no answer (a manual
   station's health score, an upload without ground truth) the UI says so rather than
   substituting a plausible number. Tests fail the build on numeric fallbacks, currency
   literals, hardcoded chart domains, and unrendered computed data.

---

## Scaling to another line

`config/line_config_plant_b.json` is a structurally different plant — 28 stations
instead of 40, two variants instead of three, 50–85% instrumentation instead of ~70%,
its own faults and its own oven geometry. The same code runs it with no changes, and
`test_generalises_to_a_different_line` asserts it still recovers every injected fault.

Adding a line means writing a config and running `calibrate.py` against it.

---

## Assumptions and limitations

Stated plainly, because the problem statement asks for assumptions to be explicit:

- **This is a prototype on simulated data.** It validates the *method*, not the model
  on a real plant. Synthetic-to-real transfer is unproven.
- **Unit economics are assumptions**, declared in `config/economics.json`. Only the
  detector performance feeding the business case is measured.
- **The risk model could learn the simulator.** Fault station, kind, onset and
  magnitude are randomised across training lines to suppress memorisation, and the demo
  line is held out of training and calibration entirely — but this is mitigation, not
  proof.
- **Calibration is imperfect in the mid-range.** The reliability diagram on the
  leadership view is measured on lines the calibrator never saw, and shows it.
- **Baselines need a fault-free Phase I window.** The first 200 vehicles establish
  contextual limits; a fault present throughout that window would be absorbed.
- **No PLC or OT system is connected.** Ingestion is read-only CSV replay. Nothing in
  this codebase can write to a control system.
- **Bottleneck precision is 50% on the demo run** — two constraints confirmed, two
  stations flagged that were not injected. Reported rather than hidden.

---

