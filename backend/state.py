"""Dashboard state assembly.

The one rule this module exists to enforce: state(t) contains only what a live system
could have known at vehicle t. Every artifact is selected by ``detected_at`` through a
sorted index with *both* bounds, and station risk is re-sliced per request rather than
computed once over the whole run. The look-ahead that made the line map disagree with
its own alarm list is structurally impossible here.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from twin import ArtifactIndex, CONFIG

RECENT_WINDOW = 20
INFERENCE_WINDOW = 10


def line_throughput(operations: list[dict], stations: list[dict]) -> dict[str, Any]:
    """Bottleneck-governed line rate.

    Serial line throughput is set by the slowest station, not the average one.
    Averaging cycle time across forty stations is what made a genuine bottleneck
    invisible in the headline KPI.
    """
    if not operations:
        return {"line_uph": None, "takt_s": None, "constraint_station": None,
                "nominal_uph": None, "recoverable_uph": 0.0}

    by_station: dict[str, float] = {}
    for o in operations:
        by_station[o["station"]] = max(by_station.get(o["station"], 0.0), o["cycle_time"])
    constraint = max(by_station, key=by_station.get)
    takt = by_station[constraint]

    nominal_takt = max(s["base_cycle"] for s in stations)
    line_uph = 3600.0 / takt if takt > 0 else None
    nominal_uph = 3600.0 / nominal_takt if nominal_takt > 0 else None

    return {
        "line_uph": round(line_uph, 1) if line_uph else None,
        "takt_s": round(takt, 2),
        "constraint_station": constraint,
        "nominal_uph": round(nominal_uph, 1) if nominal_uph else None,
        "recoverable_uph": round(max(0.0, (nominal_uph or 0) - (line_uph or 0)), 2),
        "basis": "Throughput = 3600 / slowest station cycle time (bottleneck-governed).",
    }


def station_throughput_loss(operations: list[dict], stations: list[dict]) -> list[dict]:
    """Throughput each station costs the line, in units per hour."""
    if not operations:
        return []
    worst: dict[str, float] = {}
    for o in operations:
        worst[o["station"]] = max(worst.get(o["station"], 0.0), o["cycle_time"])
    nominal = {s["id"]: s["base_cycle"] for s in stations}
    rows = []
    for sid, cycle in worst.items():
        base = nominal.get(sid, cycle)
        if cycle > base:
            rows.append({"station": sid,
                         "cycle_s": round(cycle, 2), "nominal_s": round(base, 2),
                         "excess_s": round(cycle - base, 2)})
    rows.sort(key=lambda r: -r["excess_s"])
    return rows[:8]


def build_state(raw: dict, analysis: dict, metrics: dict, ml: dict | None = None,
                upto_vehicle: int | None = None, config: dict = CONFIG,
                indices: dict | None = None) -> dict[str, Any]:
    count = raw.get("vehicle_count") or max(o["vehicle"] for o in raw["operations"])
    latest = max(1, min(upto_vehicle or count, count))

    idx = indices or {}
    flag_idx: ArtifactIndex = idx.get("flags") or ArtifactIndex(analysis["flags"])
    bn_idx: ArtifactIndex = idx.get("bottlenecks") or ArtifactIndex(analysis["bottlenecks"])

    ops_upto = [o for o in raw["operations"] if o["vehicle"] <= latest]
    latest_ops = {o["station"]: o for o in ops_upto}

    ml_scores = (ml or {}).get("station_scores", {})
    ml_drivers = (ml or {}).get("station_drivers", {})

    stations_out = []
    for s in raw["stations"]:
        sid = s["id"]
        recent = flag_idx.window(sid, latest - RECENT_WINDOW, latest)
        recent_bn = bn_idx.window(sid, latest - RECENT_WINDOW, latest)
        op = latest_ops.get(sid)

        if recent_bn:
            status = "alert"
        elif any(f["severity"] == "alert" for f in recent):
            status = "alert"
        elif recent:
            status = "warning"
        else:
            status = "healthy"

        if s["instrumented"]:
            gap = "instrumented"
        elif op and op.get("manual_outcome") in {"fail", "rework"}:
            gap, status = "manual outcome", "alert"
        elif op and op.get("manual_outcome") == "pass" and status == "healthy":
            gap = "manual outcome"
        else:
            # No local signal: infer from instrumented neighbours, strictly within
            # the causal window.
            gap = "adjacent-signal inference"
            neighbours = [n for n in (s.get("upstream"), s.get("downstream")) if n]
            inferred = [f for n in neighbours
                        for f in flag_idx.window(n, latest - INFERENCE_WINDOW, latest)]
            if status == "healthy" and inferred:
                status = "alert" if any(f["severity"] == "alert" for f in inferred) else "warning"

        # Health score from the causal window only.
        window_readings = [r for r in raw["readings"]
                           if r["station"] == sid and latest - RECENT_WINDOW <= r["vehicle"] <= latest]
        if window_readings:
            weighted = [abs(r["z"]) * s["parameters"].get(r["parameter"], {}).get("weight", 1.0)
                        for r in window_readings]
            health = round(max(0.0, 100.0 - 16.0 * float(np.mean(weighted))), 1)
        else:
            health = None

        risk = ml_scores.get(sid, {}).get(latest)
        drivers = ml_drivers.get(sid, {}).get(latest, [])

        stations_out.append({
            "id": sid, "name": s["name"], "area": s["area"], "status": status,
            "instrumented": s["instrumented"], "gap_method": gap,
            "health_score": health,
            "downstream": s.get("downstream"),
            "operation": op,
            "open_flags": recent,
            "defect_risk_pct": risk,
            "risk_drivers": drivers,
        })

    throughput = line_throughput(ops_upto, raw["stations"])
    bottlenecks_now = bn_idx.upto(latest)
    throughput["bottleneck_events_per_run"] = len({b["station"] for b in analysis["bottlenecks"]})

    line = {
        "name": raw.get("config_name", config["line"]["name"]),
        "vehicle_count": count,
        "current_vehicle": latest,
        "source": raw.get("source", "synthetic simulation"),
        "scheduled_sensor_change_windows_per_year": config["line"]["scheduled_sensor_change_windows_per_year"],
        "instrumented_stations": sum(1 for s in raw["stations"] if s["instrumented"]),
        "total_stations": len(raw["stations"]),
        "baseline_warmup_vehicles": analysis.get("baseline_warmup"),
        **throughput,
    }

    return {
        "line": line,
        "thresholds": config["thresholds"],
        "calibration": config["bottleneck"].get("calibrated"),
        "stations": stations_out,
        "flags": flag_idx.upto(latest),
        "bottlenecks": bottlenecks_now,
        "physics_flags": [f for f in analysis.get("physics_flags", []) if f["detected_at"] <= latest],
        "throughput_loss": station_throughput_loss(ops_upto, raw["stations"]),
        "metrics": metrics,
        "propagation": analysis.get("propagation", {}),
        "ml": {k: ml.get(k) for k in
               ("model", "backend", "attribution_method", "target", "training_provenance",
                "calibration", "feature_importances", "feature_labels", "validation",
                "horizon_vehicles")} if ml else None,
        "sensor_retrofit_plan": config.get("sensor_retrofit_plan", []),
        "scalability": config.get("scalability", {}),
    }
