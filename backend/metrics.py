"""Deterministic validation: backtest, false-alarm floor, and sensor-coverage study.

Every figure the dashboard and the business case quote about detector quality is
produced here. Two properties matter and are deliberately enforced:

* Precision is reported for *both* anomalies and bottlenecks. Reporting recall
  alone lets a detector that fires constantly score perfectly.
* The false-alarm floor is measured on injection-free runs, not asserted. That is
  the number the ROI model charges investigation cost against.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from twin import CONFIG, analyze, simulate

ANOMALY_KINDS = {"trend", "point", "cross_parameter", "thermal_tau_drift", "anomaly"}
MATCH_LEAD, MATCH_TAIL = 2, 6


def _span(scenario: dict) -> tuple[int, int]:
    vehicles = scenario.get("vehicles")
    if vehicles:
        return min(vehicles), max(vehicles)
    return scenario.get("start_vehicle", 0), scenario.get("end_vehicle", 0)


def _pct(numerator: float, denominator: float) -> float | None:
    return round(100.0 * numerator / denominator, 1) if denominator else None


def backtest(raw: dict, result: dict, config: dict = CONFIG) -> dict[str, Any]:
    """Scenario-level backtest against injected or uploaded ground truth."""
    if raw.get("source") == "uploaded CSV" and not raw.get("validation_scenarios"):
        return {"validation_available": False,
                "reason": "No ground-truth CSV supplied with this upload.",
                "precision": None, "recall": None, "false_positives": None,
                "mean_detection_lag": None, "bottleneck_recall": None,
                "bottleneck_precision": None, "evaluated_events": 0}

    scenarios = raw.get("validation_scenarios") or config["injections"]
    flags, bottlenecks = result["flags"], result["bottlenecks"]

    anomaly_scen = [s for s in scenarios if s.get("kind") in ANOMALY_KINDS]
    bn_scen = [s for s in scenarios if s.get("kind") == "bottleneck"]
    latent_scen = [s for s in scenarios if s.get("kind") == "latent_defect"]

    # --- anomalies -------------------------------------------------------
    matched, lags = [], []
    for scen in anomaly_scen:
        start, end = _span(scen)
        hits = [f for f in flags
                if f["station"] == scen["station"] and start - MATCH_LEAD <= f["detected_at"] <= end + MATCH_TAIL]
        if hits:
            first = min(h["detected_at"] for h in hits)
            matched.append(scen)
            lags.append(first - start)

    # A station is "truly faulty" if ground truth says a fault was active there -
    # not merely if it is named in a scenario header. A line-wide scenario such as
    # oven fouling ("PNT-*") affects every paint station, and a latent defect affects
    # both its origin and the inspection station where it surfaces. Counting those as
    # false positives would understate precision for detections that are correct.
    truth_stations = {s["station"] for s in anomaly_scen}
    truth_stations |= {g["station"] for g in raw.get("ground_truth", [])}
    for scen in scenarios:
        station = scen.get("station", "")
        if station.endswith("-*"):
            prefix = station[:-1]
            truth_stations |= {s["id"] for s in raw["stations"] if s["id"].startswith(prefix)}
        if scen.get("kind") == "latent_defect" and scen.get("inspection_station"):
            truth_stations.add(scen["inspection_station"])
    candidate_stations = sorted({f["station"] for f in flags})
    true_pos = [s for s in candidate_stations if s in truth_stations]
    false_pos = [s for s in candidate_stations if s not in truth_stations]

    # --- bottlenecks -----------------------------------------------------
    bn_matched, bn_lags = [], []
    for scen in bn_scen:
        hits = [b for b in bottlenecks
                if b["station"] == scen["station"] and b["detected_at"] >= scen["start_vehicle"] - MATCH_LEAD]
        if hits:
            first = min(h["detected_at"] for h in hits)
            bn_matched.append(scen)
            bn_lags.append(first - scen["start_vehicle"])

    bn_truth = {s["station"] for s in bn_scen}
    bn_candidates = sorted({b["station"] for b in bottlenecks})
    bn_true = [s for s in bn_candidates if s in bn_truth]
    bn_false = [s for s in bn_candidates if s not in bn_truth]

    # --- latent defects (recovered by back-tracing, not by SPC) ----------
    prop = result.get("propagation", {})
    latent_found = bool(latent_scen) and bool(prop.get("origin_identified"))

    all_lags = lags + bn_lags
    return {
        "validation_available": True,
        "precision": _pct(len(true_pos), len(candidate_stations)),
        "recall": _pct(len(matched), len(anomaly_scen)),
        "false_positives": len(false_pos),
        "false_positive_stations": false_pos,
        "bottleneck_precision": _pct(len(bn_true), len(bn_candidates)),
        "bottleneck_recall": _pct(len(bn_matched), len(bn_scen)),
        "bottleneck_false_positives": len(bn_false),
        "mean_detection_lag": round(float(np.mean(all_lags)), 1) if all_lags else None,
        "median_detection_lag": round(float(np.median(all_lags)), 1) if all_lags else None,
        "p90_detection_lag": round(float(np.percentile(all_lags, 90)), 1) if all_lags else None,
        "latent_defect_scenarios": len(latent_scen),
        "latent_origin_identified": latent_found,
        "latent_origin_station": prop.get("ranking", [{}])[0].get("station") if prop.get("ranking") else None,
        "evaluated_events": len(anomaly_scen) + len(bn_scen) + len(latent_scen),
        "detected_events": len(matched) + len(bn_matched) + (1 if latent_found else 0),
    }


def false_alarm_profile(config: dict = CONFIG, seeds: int = 12) -> dict[str, Any]:
    """Measure the false-alarm floor on fault-free lines.

    With no injections present every alarm is by construction a false one, so this
    yields an honest ARL0 (average run length to a false alarm) rather than an
    assumed one.
    """
    clean = copy.deepcopy(config)
    clean["injections"] = []
    vehicles = clean["line"]["vehicle_count"]

    anomaly_runs, bottleneck_runs, physics_runs = [], [], []
    for seed in range(1000, 1000 + seeds):
        raw = simulate(clean, seed=seed)
        res = analyze(raw, clean)
        anomaly_stations = {f["station"] for f in res["flags"] if f["type"] != "physics_thermal_violation"}
        anomaly_runs.append(len(anomaly_stations))
        bottleneck_runs.append(len({b["station"] for b in res["bottlenecks"]}))
        physics_runs.append(len(res["physics_flags"]))

    total = float(np.mean(anomaly_runs) + np.mean(bottleneck_runs) + np.mean(physics_runs))
    return {
        "seeds": seeds,
        "vehicles_per_run": vehicles,
        "anomaly_false_alarms_per_run": round(float(np.mean(anomaly_runs)), 2),
        "bottleneck_false_alarms_per_run": round(float(np.mean(bottleneck_runs)), 2),
        "physics_false_alarms_per_run": round(float(np.mean(physics_runs)), 2),
        "total_false_alarms_per_run": round(total, 2),
        "false_alarms_per_1000_vehicles": round(total * 1000.0 / vehicles, 2),
        "arl0_vehicles": round(vehicles / total, 1) if total > 0 else None,
        "method": "Injection-free simulation; every alarm raised is a false alarm by construction.",
    }


def sensor_gap_study(config: dict = CONFIG, coverage_levels=(0.70, 0.50, 0.30),
                     seed: int = 42) -> dict[str, Any]:
    """Quantify how detection degrades as instrumentation coverage falls.

    Answers the problem statement's sensor-gap question with a number instead of a
    claim, and gives the retrofit business case a measured benefit to price: the
    recall recovered per point of coverage regained.
    """
    rows = []
    for level in coverage_levels:
        cfg = copy.deepcopy(config)
        for group in cfg["station_groups"]:
            group["instrumented_ratio"] = level
        raw = simulate(cfg, seed=seed)
        res = analyze(raw, cfg)
        bt = backtest(raw, res, cfg)
        instrumented = sum(1 for s in raw["stations"] if s["instrumented"])
        rows.append({
            "instrumented_ratio": level,
            "instrumented_stations": instrumented,
            "manual_stations": len(raw["stations"]) - instrumented,
            "recall": bt["recall"],
            "precision": bt["precision"],
            "bottleneck_recall": bt["bottleneck_recall"],
            "median_detection_lag": bt["median_detection_lag"],
            "latent_origin_identified": bt["latent_origin_identified"],
        })
    baseline, worst = rows[0], rows[-1]
    delta = None
    if baseline["recall"] is not None and worst["recall"] is not None:
        span = baseline["instrumented_ratio"] - worst["instrumented_ratio"]
        delta = round((baseline["recall"] - worst["recall"]) / span / 100.0, 3) if span else None
    return {
        "levels": rows,
        "recall_lost_per_coverage_point": delta,
        "note": ("Bottleneck and cycle-time detection survive instrumentation loss because "
                 "they read operational metrics available at every station; parameter-level "
                 "anomaly detection is what degrades."),
    }
