"""Derive detector thresholds from a measured false-alarm budget.

No control limit in this system is chosen by hand. Each one is swept against
injection-free simulated lines - where every alarm raised is false by construction -
and the tightest limit that still meets an explicit budget is written back into
``line_config.json`` with its provenance.

This is the direct, quantitative answer to the operational risk the problem
statement raises: false alarms about defects that never materialise erode floor
trust. Here that risk is a number with a stated measurement method, and the
thresholds are set by it rather than by taste.

Run:  python calibrate.py [--seeds N] [--budget F]
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from datetime import date

import numpy as np

from twin import (CONFIG, ROOT, build_baselines, detect_bottlenecks, fit_oven_tau,
                  oven_dwell_seconds, score_readings, simulate, thermal_model_C)

CONFIG_PATH = ROOT / "config" / "line_config.json"


def _clean(config: dict) -> dict:
    clean = copy.deepcopy(config)
    clean["injections"] = []
    return clean


def _ewma_peaks(raw: dict, lam: float) -> dict[str, float]:
    """Peak |EWMA| per station on a fault-free run.

    Because the rule is latched, a station alarms for a given limit L if and only if
    its peak exceeds L. Recording peaks once turns the whole threshold sweep into a
    comparison instead of a re-run per candidate.
    """
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in raw["readings"]:
        by_key[(r["station"], r["parameter"])].append(r)
    peaks: dict[str, float] = {}
    for (station, _param), rows in by_key.items():
        rows.sort(key=lambda r: r["vehicle"])
        e, peak = 0.0, 0.0
        for r in rows:
            e = lam * r["z"] + (1 - lam) * e
            peak = max(peak, abs(e))
        peaks[station] = max(peaks.get(station, 0.0), peak)
    return peaks


def _tau_peak(raw: dict, config: dict) -> float:
    stations = {s["id"]: s for s in raw["stations"]}
    dwell_of = {sid: oven_dwell_seconds(s, config) for sid, s in stations.items()}
    per_vehicle: dict[int, list] = defaultdict(list)
    for r in raw["readings"]:
        if r["parameter"] == "temperature" and dwell_of.get(r["station"]) is not None:
            per_vehicle[r["vehicle"]].append((dwell_of[r["station"]], r["value"]))
    win = int(config["physics"].get("tau_fit_window_vehicles", 15))
    vehicles = sorted(per_vehicle)
    if len(vehicles) < win * 3:
        return 0.0
    fitted = {}
    for i in range(win - 1, len(vehicles)):
        pooled = [x for v in vehicles[i - win + 1:i + 1] for x in per_vehicle[v]]
        tau = fit_oven_tau(pooled, config)
        if tau is not None:
            fitted[vehicles[i]] = tau
    warmup = min(config["thresholds"]["baseline_warmup_vehicles"], len(vehicles) // 2)
    base = [t for v, t in fitted.items() if v <= warmup]
    if len(base) < 20:
        return 0.0
    mu, sd = float(np.mean(base)), max(float(np.std(base, ddof=1)), 1e-6)
    return max(abs((t - mu) / sd) for t in fitted.values())


def _bottleneck_peaks(raw: dict, config: dict) -> dict[str, float]:
    """Peak CUSUM (in sigma) per station among positions where the corroborating
    test also passed - the quantity the h limit is compared against."""
    cfg = config["bottleneck"]
    probe = copy.deepcopy(config)
    probe["bottleneck"] = {**cfg, "cusum_h_sigma": 0.0, "calibrated": None}
    peaks: dict[str, float] = {}
    for ev in detect_bottlenecks(raw, probe):
        peaks[ev["station"]] = max(peaks.get(ev["station"], 0.0), ev["cusum"])
    return peaks


def calibrate(config: dict = CONFIG, seeds: int = 12, budget: float = 1.0) -> dict:
    """Sweep every detector limit against the fault-free false-alarm budget."""
    clean = _clean(config)
    lam = float(config["thresholds"]["ewma_lambda"])

    def collect(base_seed: int, n: int):
        ew, bn, tau = [], [], []
        for seed in range(base_seed, base_seed + n):
            raw = simulate(clean, seed=seed)
            raw["baselines"] = build_baselines(raw, clean)
            score_readings(raw, raw["baselines"])
            ew.append(_ewma_peaks(raw, lam))
            bn.append(_bottleneck_peaks(raw, clean))
            tau.append(_tau_peak(raw, clean))
        return ew, bn, tau, raw

    # Two disjoint seed sets. A limit chosen on the same runs it is scored against is
    # optimistically biased - it is fitted to that sample's particular tail. The limit
    # has to clear the budget on runs it was not selected on.
    fit_ew, fit_bn, fit_tau, sample = collect(2000, seeds)
    val_ew, val_bn, val_tau, _ = collect(5000, max(4, seeds // 2))

    share = budget / 3.0

    def rate_of(runs: list[dict[str, float]], limit: float) -> float:
        return float(np.mean([sum(1 for p in run.values() if p > limit) for run in runs]))

    def sweep(fit_runs, val_runs, grid):
        for limit in grid:
            if rate_of(fit_runs, limit) <= share and rate_of(val_runs, limit) <= share:
                return float(limit), rate_of(val_runs, limit)
        return float(grid[-1]), rate_of(val_runs, grid[-1])

    n_base = max(int(np.median([r.get("baseline_n", 1) or 1 for r in sample["readings"]])), 1)
    sigma_e = float(np.sqrt(lam / (2.0 - lam) + 1.0 / n_base))
    scale = lambda runs: [{k: v / sigma_e for k, v in run.items()} for run in runs]

    ewma_L, ewma_rate = sweep(scale(fit_ew), scale(val_ew), np.arange(2.5, 14.01, 0.1))
    bn_h, bn_rate = sweep(fit_bn, val_bn, np.arange(2.0, 60.01, 0.25))

    tau_grid = np.arange(2.0, 14.01, 0.1)
    tau_crit, tau_rate = float(tau_grid[-1]), 0.0
    for limit in tau_grid:
        f = float(np.mean([1.0 if p > limit else 0.0 for p in fit_tau]))
        w = float(np.mean([1.0 if p > limit else 0.0 for p in val_tau]))
        if f <= share and w <= share:
            tau_crit, tau_rate = float(limit), w
            break

    n_runs = seeds
    total = ewma_rate + bn_rate + tau_rate
    vehicles = config["line"]["vehicle_count"]
    return {
        "method": ("Threshold sweep against injection-free simulated lines; every alarm "
                   "raised on those runs is a false alarm by construction."),
        "seeds": n_runs,
        "validation_seeds": max(4, seeds // 2),
        "vehicles_per_run": vehicles,
        "budget_false_alarms_per_run": budget,
        "ewma_L": round(ewma_L, 2),
        "cusum_h_sigma": round(bn_h, 2),
        "tau_fit_z_crit": round(tau_crit, 2),
        "measured_false_alarms_per_run_heldout": {
            "anomaly": round(ewma_rate, 3),
            "bottleneck": round(bn_rate, 3),
            "physics": round(tau_rate, 3),
            "total": round(total, 3),
        },
        "arl0_vehicles": round(vehicles / total, 1) if total > 0 else None,
        "calibrated_on": date.today().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--budget", type=float, default=1.0,
                        help="Target false alarms per run, line-wide.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = calibrate(CONFIG, seeds=args.seeds, budget=args.budget)
    print(json.dumps(result, indent=2))
    if args.dry_run:
        return

    config = json.loads(CONFIG_PATH.read_text())
    config["thresholds"]["ewma_L"] = result["ewma_L"]
    config["bottleneck"]["cusum_h_sigma"] = result["cusum_h_sigma"]
    config["physics"]["tau_fit_z_crit"] = result["tau_fit_z_crit"]
    config["bottleneck"]["calibrated"] = result
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    print(f"\nWritten to {CONFIG_PATH.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
