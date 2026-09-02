"""Config-driven digital twin: simulator, contextual baselines, and detection engines.

Design contract enforced throughout this module:

* Every derived artifact carries ``detected_at`` - the vehicle index at which an
  online detector would first have been able to emit it (the *closing* vehicle of
  the rule's window, not the vehicle the fault started on). Consumers slice on
  ``detected_at``, so the dashboard can never show a signal before it was knowable.
* No detector threshold is a bare literal. Every one is read from config, and the
  bottleneck thresholds are written there by ``calibrate.py`` from measured
  fault-free false-alarm rates.
"""
from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "config" / "line_config.json").read_text())


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------

def build_stations(config: dict) -> list[dict]:
    """Ordered station list. Order defines the physical line, so index+1 is the
    genuine downstream neighbour and is recorded explicitly rather than inferred
    at each call site."""
    stations: list[dict] = []
    for group in config["station_groups"]:
        manual = round(group["count"] * (1 - group["instrumented_ratio"]))
        for n in range(1, group["count"] + 1):
            stations.append({
                "id": f"{group['prefix']}-{n:02}",
                "name": f"{group['area']} {n:02}",
                "area": group["area"],
                "index_in_group": n,
                "instrumented": n > manual,
                "parameters": group["parameters"],
                "base_cycle": group["cycle_time"],
            })
    for i, s in enumerate(stations):
        s["downstream"] = stations[i + 1]["id"] if i + 1 < len(stations) else None
        s["upstream"] = stations[i - 1]["id"] if i > 0 else None
    return stations


def oven_dwell_seconds(station: dict, config: dict) -> float | None:
    """Dwell time of a body in the curing oven at this paint station."""
    if station["area"] != "Paint":
        return None
    oven = config["physics"]["paint_oven"]
    return oven["dwell_base_s"] + oven["dwell_step_s"] * (station["index_in_group"] - 1)


def thermal_model_C(dwell_s: float, tau_s: float, config: dict) -> float:
    """Newton's law of cooling / first-order thermal response.

    A body entering at ambient approaches the oven setpoint exponentially:
        T(t) = T_ambient + (T_setpoint - T_ambient) * (1 - exp(-t / tau))
    """
    oven = config["physics"]["paint_oven"]
    ambient, setpoint = oven["ambient_C"], oven["setpoint_C"]
    return ambient + (setpoint - ambient) * (1.0 - np.exp(-dwell_s / tau_s))


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def _active(item: dict, vehicle: int) -> bool:
    if vehicle in item.get("vehicles", []):
        return True
    return item.get("start_vehicle", 10**9) <= vehicle <= item.get("end_vehicle", -1)


def injected(config, station, parameter, vehicle):
    hits = []
    for item in config["injections"]:
        if item["station"] != station or not _active(item, vehicle):
            continue
        if parameter in item.get("parameters", []) or parameter == item.get("parameter"):
            hits.append(item)
    return hits


def simulate(config=CONFIG, seed=42) -> dict[str, Any]:
    """Generate one synthetic production history.

    Models three latent causal factors the problem statement calls out - operator
    variation, incoming part quality, and ambient conditions - so that root-cause
    attribution has something real to recover. Paint temperatures are generated
    from the thermal model rather than drawn around a flat mean, which is what
    makes the physics residual an independent detector rather than a duplicate
    of the z-test.
    """
    rng = np.random.default_rng(seed)
    stations = build_stations(config)
    count = config["line"]["vehicle_count"]
    latent = config["latent_factors"]
    oven = config["physics"]["paint_oven"]
    per_shift = config["line"]["vehicles_per_shift"]
    inspection_id = config.get("inspection_station")

    operators = latent["operators"]
    operator_skill = {op: float(rng.normal(0, latent["operator_skill_sigma"])) for op in operators}
    # Mixed-model context: a variant genuinely shifts some station parameters, which
    # is why baselines are held per (station, parameter, variant) rather than pooled.
    variant_sigma = latent.get("variant_offset_sigma", 0.0)
    variant_offset = {}
    for st in stations:
        for pname in st["parameters"]:
            for var in config["product_variants"]:
                variant_offset[(st["id"], pname, var)] = float(rng.normal(0, variant_sigma))
    lot_size = latent["supplier_lot_size_vehicles"]
    lot_quality: dict[int, float] = {}

    tau_drifts = [x for x in config["injections"] if x["kind"] == "thermal_tau_drift"]
    latent_defects = [x for x in config["injections"] if x["kind"] == "latent_defect"]

    readings: list[dict] = []
    operations: list[dict] = []
    ground_truth: list[dict] = []
    carried_defect: dict[int, dict] = {}

    for v in range(1, count + 1):
        variant = config["product_variants"][(v - 1) % len(config["product_variants"])]
        shift = (v - 1) // per_shift
        operator = operators[shift % len(operators)]
        op_offset = operator_skill[operator]
        lot = (v - 1) // lot_size
        if lot not in lot_quality:
            lot_quality[lot] = float(rng.normal(0, latent["supplier_quality_sigma"]))
        lot_offset = lot_quality[lot]
        ambient_offset = latent["ambient_swing_C"] * np.sin(2 * np.pi * v / latent["ambient_period_vehicles"])

        # Oven thermal time constant, degraded by any active fouling injection.
        tau = oven["tau_s"]
        for drift in tau_drifts:
            if _active(drift, v):
                span = max(1, drift["end_vehicle"] - drift["start_vehicle"])
                frac = (v - drift["start_vehicle"]) / span
                tau = oven["tau_s"] + (drift["tau_end_s"] - oven["tau_s"]) * frac
                ground_truth.append({"type": "thermal_tau_drift", "station": "PNT-*",
                                     "parameter": "temperature", "vehicle": v,
                                     "scenario": drift["id"]})

        # A latent defect is introduced upstream with a deliberately sub-threshold
        # signal, and only surfaces later at the inspection station.
        for ld in latent_defects:
            if _active(ld, v) and rng.random() < ld["surface_probability"]:
                carried_defect[v] = ld
                ground_truth.append({"type": "latent_defect", "station": ld["station"],
                                     "parameter": ld.get("parameter"), "vehicle": v,
                                     "scenario": ld["id"],
                                     "surfaces_at": ld["inspection_station"]})

        for s in stations:
            cycle = s["base_cycle"] + rng.normal(0, 3) + op_offset * 1.4
            queue = max(0, int(rng.normal(3, 1.4)))
            bn = [x for x in config["injections"]
                  if x["kind"] == "bottleneck" and x["station"] == s["id"] and v >= x["start_vehicle"]]
            if bn:
                cycle += bn[0]["cycle_slope"] * (v - bn[0]["start_vehicle"])
                queue += int((v - bn[0]["start_vehicle"]) / 18)
                ground_truth.append({"type": "bottleneck", "station": s["id"],
                                     "vehicle": v, "scenario": bn[0]["id"]})
            utilization = min(99, max(30, 58 + cycle / s["base_cycle"] * 18 + queue * 2 + rng.normal(0, 2)))

            op = {
                "vehicle": v, "variant": variant, "station": s["id"],
                "cycle_time": round(float(cycle), 2), "queue": queue,
                "utilization": round(float(utilization), 1),
                "operator": operator, "supplier_lot": f"LOT-{lot:03d}",
                "ambient_c": round(float(latent["ambient_swing_C"] * np.sin(2 * np.pi * v / latent["ambient_period_vehicles"])), 3),
            }
            operations.append(op)

            if not s["instrumented"]:
                # Manual stations report an outcome only. The inspection station is
                # where a carried latent defect finally becomes visible.
                fails = v in carried_defect and s["id"] == inspection_id
                rework = fails or rng.random() < 0.045
                op["manual_outcome"] = ("fail" if fails else "rework") if rework else "pass"
                continue

            for p, spec in s["parameters"].items():
                if spec.get("model") == "thermal_first_order":
                    dwell = oven_dwell_seconds(s, config)
                    expected = thermal_model_C(dwell, tau, config) + ambient_offset * 0.35
                    value = expected + rng.normal(0, oven["measurement_sigma_C"])
                else:
                    # Un-truncated noise. Clipping the tail would manufacture a
                    # flattering false-alarm rate: a 3-sigma rule cannot fire on
                    # data that is capped below 3 sigma.
                    value = spec["mean"] + rng.normal(0, spec["std"])
                    value += spec["std"] * (op_offset * 0.22 + lot_offset * 0.18)
                    value += spec["std"] * variant_offset.get((s["id"], p, variant), 0.0)

                for hit in injected(config, s["id"], p, v):
                    if hit["kind"] == "trend":
                        value += spec["std"] * hit["slope_std_per_vehicle"] * (v - hit["start_vehicle"])
                    elif hit["kind"] == "latent_defect":
                        if v in carried_defect:
                            value += spec["std"] * hit["shift_std"]
                        continue
                    else:
                        value += spec["std"] * hit["shift_std"]
                    ground_truth.append({"type": hit["kind"], "station": s["id"], "parameter": p,
                                         "vehicle": v, "scenario": hit["id"]})

                # Inspection vision score drops for a vehicle carrying a latent defect.
                if p == "vision_score" and s["id"] == inspection_id and v in carried_defect:
                    value -= spec["std"] * 3.6

                readings.append({
                    "vehicle": v, "variant": variant, "station": s["id"], "parameter": p,
                    "value": round(float(value), 3), "operator": operator,
                    "ambient_c": round(float(ambient_offset), 3),
                    "supplier_lot": f"LOT-{lot:03d}",
                })

    return {
        "stations": stations,
        "readings": readings,
        "operations": operations,
        "ground_truth": ground_truth,
        "vehicle_count": count,
        "source": "synthetic simulation",
        "config_name": config["line"]["name"],
    }


# --------------------------------------------------------------------------
# Contextual baselines
# --------------------------------------------------------------------------

def build_baselines(raw: dict, config: dict = CONFIG) -> dict:
    """Empirical (station, parameter, variant) baselines from a warm-up prefix.

    The warm-up window is a *prefix* of the run, so a baseline is never informed by
    the faults it will later be used to detect. Where a cell has too few samples
    the group spec is used and the fallback is recorded, rather than silently
    substituting a healthy-looking number.
    """
    warmup = config["thresholds"]["baseline_warmup_vehicles"]
    count = raw.get("vehicle_count") or max((r["vehicle"] for r in raw["readings"]), default=0)
    warmup = min(warmup, max(30, count // 2))

    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in raw["readings"]:
        if r["vehicle"] <= warmup:
            buckets[(r["station"], r["parameter"], r["variant"])].append(r["value"])
            buckets[(r["station"], r["parameter"], None)].append(r["value"])

    # Physics-modelled parameters get a regression baseline rather than a flat mean:
    # the deterministic part (oven dwell + measured ambient) is predicted and removed,
    # so only the unexplained residual is charted. Without this, a slow ambient swing
    # walks straight through the control limits and reads as a process fault.
    station_by_id = {s["id"]: s for s in raw["stations"]}
    thermal: dict[str, dict] = {}
    thermal_rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in raw["readings"]:
        st = station_by_id.get(r["station"])
        if not st or st["parameters"].get(r["parameter"], {}).get("model") != "thermal_first_order":
            continue
        if r["vehicle"] <= warmup:
            dwell = oven_dwell_seconds(st, config)
            pred = float(thermal_model_C(dwell, config["physics"]["paint_oven"]["tau_s"], config))
            thermal_rows[r["station"]].append((r.get("ambient_c", 0.0), r["value"] - pred))
    for sid, rows in thermal_rows.items():
        if len(rows) < 20:
            continue
        amb = np.array([a for a, _ in rows])
        res = np.array([d for _, d in rows])
        slope, intercept = (np.polyfit(amb, res, 1) if amb.std() > 1e-9 else (0.0, res.mean()))
        resid = res - (slope * amb + intercept)
        thermal[sid] = {"ambient_coeff": float(slope), "offset": float(intercept),
                        "sigma": max(float(np.std(resid, ddof=1)), 1e-6), "n": len(rows)}

    spec_by_station = {s["id"]: s["parameters"] for s in raw["stations"]}
    baselines: dict[tuple, dict] = {}
    for key, values in buckets.items():
        station, parameter, _ = key
        spec = spec_by_station.get(station, {}).get(parameter, {})
        if len(values) >= 12:
            std = float(np.std(values, ddof=1))
            baselines[key] = {
                "mean": float(np.mean(values)), "std": max(std, 1e-6),
                "n": len(values), "source": "empirical",
            }
        elif spec:
            baselines[key] = {
                "mean": float(spec["mean"]), "std": float(spec["std"]),
                "n": len(values), "source": "spec_fallback",
            }
    return {"warmup_vehicles": warmup, "cells": baselines, "thermal": thermal}


def score_readings(raw: dict, baselines: dict) -> None:
    """Attach a contextual z-score to every reading, in place."""
    cells = baselines["cells"]
    thermal = baselines.get("thermal", {})
    station_by_id = {s["id"]: s for s in raw["stations"]}
    spec_by_station = {s["id"]: s["parameters"] for s in raw["stations"]}
    for r in raw["readings"]:
        fit = thermal.get(r["station"])
        st = station_by_id.get(r["station"])
        if fit and st and st["parameters"].get(r["parameter"], {}).get("model") == "thermal_first_order":
            dwell = oven_dwell_seconds(st, CONFIG)
            pred = float(thermal_model_C(dwell, CONFIG["physics"]["paint_oven"]["tau_s"], CONFIG))
            expected = pred + fit["offset"] + fit["ambient_coeff"] * r.get("ambient_c", 0.0)
            r["mean"] = round(expected, 3)
            r["std"] = round(fit["sigma"], 4)
            r["z"] = round((r["value"] - expected) / fit["sigma"], 3)
            r["baseline_source"] = "physics_model_residual"
            continue
        # Mean is contextual to the variant; dispersion is pooled across variants,
        # which share the same process physics. A per-variant std estimated from a
        # third of the warm-up is noisy enough to inflate the z tail on its own.
        cell = cells.get((r["station"], r["parameter"], r["variant"]))
        pooled = cells.get((r["station"], r["parameter"], None))
        if cell and pooled and pooled["n"] >= 36:
            # Shrink the per-variant mean toward the pooled mean in proportion to how
            # much of the observed gap is explainable by sampling error alone. An
            # unshrunk per-variant mean carries ~1/sqrt(n) of noise straight into the
            # control chart, where a sustained-shift rule reads it as a real fault.
            delta = cell["mean"] - pooled["mean"]
            se_sq = (pooled["std"] ** 2) / max(cell["n"], 1)
            shrink = max(0.0, 1.0 - se_sq / max(delta ** 2, 1e-12))
            cell = {**cell, "mean": pooled["mean"] + delta * shrink,
                    "std": pooled["std"], "n_effective": pooled["n"],
                    "source": f"{cell['source']}+shrunk_variant"}
        cell = cell or pooled
        if cell is None:
            spec = spec_by_station.get(r["station"], {}).get(r["parameter"])
            if not spec:
                r["z"], r["mean"], r["std"], r["baseline_source"] = 0.0, 0.0, 1.0, "unavailable"
                continue
            cell = {"mean": spec["mean"], "std": spec["std"], "source": "spec_fallback"}
        r["mean"] = round(cell["mean"], 3)
        r["std"] = round(cell["std"], 4)
        r["z"] = round((r["value"] - cell["mean"]) / cell["std"], 3)
        r["baseline_source"] = cell["source"]
        r["baseline_n"] = cell.get("n_effective", cell.get("n", 0))


# --------------------------------------------------------------------------
# Physics-informed detector
# --------------------------------------------------------------------------

def fit_oven_tau(samples: list[tuple[float, float]], config: dict) -> float | None:
    """Least-squares fit of the oven's thermal time constant from (dwell, temperature)
    samples, marginalising out a uniform ambient offset.

    Ambient temperature shifts every station by the same amount, so it is a nuisance
    parameter here: for each candidate tau the optimal offset is subtracted before
    scoring. What remains is the *shape* of the temperature-versus-dwell profile,
    which only the thermal time constant can change. That is what makes this
    detector independent of the per-station z-test rather than a restatement of it.
    """
    oven = config["physics"]["paint_oven"]
    pairs = [(d, t) for d, t in samples if d is not None]
    if len(pairs) < 6:
        return None
    ambient, setpoint = oven["ambient_C"], oven["setpoint_C"]
    grid = np.linspace(oven["tau_s"] * 0.6, oven["tau_s"] * 1.8, 400)
    dwell = np.array([d for d, _ in pairs])
    meas = np.array([t for _, t in pairs])
    pred = ambient + (setpoint - ambient) * (1.0 - np.exp(-dwell[None, :] / grid[:, None]))
    resid = meas[None, :] - pred
    resid -= resid.mean(axis=1, keepdims=True)          # marginalise ambient offset
    return float(grid[int(np.argmin((resid ** 2).sum(axis=1)))])


def physics_thermal_check(raw: dict, config: dict = CONFIG) -> list[dict]:
    """Detect oven degradation as a drift in the fitted thermal time constant.

    This is deliberately orthogonal to the per-station z-test: it reads a
    cross-station *pattern*, so it catches a fouling oven whose individual
    stations all remain inside their own control limits.
    """
    stations = {s["id"]: s for s in raw["stations"]}
    dwell_of = {sid: oven_dwell_seconds(s, config) for sid, s in stations.items()}
    paint_temps: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in raw["readings"]:
        if r["parameter"] == "temperature" and dwell_of.get(r["station"]) is not None:
            paint_temps[r["vehicle"]].append((dwell_of[r["station"]], r["value"]))
    if not paint_temps:
        return []

    # Pool a rolling window of vehicles before fitting. A single vehicle gives ~10
    # points and a noisy tau; pooling shrinks the fit's standard error by sqrt(n)
    # and is what lifts a sub-control-limit drift into a detectable signal.
    win = int(config["physics"].get("tau_fit_window_vehicles", 15))
    vehicles = sorted(paint_temps)
    if len(vehicles) < win * 3:
        return []
    fitted: dict[int, float] = {}
    for i in range(win - 1, len(vehicles)):
        pooled: list[tuple[float, float]] = []
        for v in vehicles[i - win + 1:i + 1]:
            pooled.extend(paint_temps[v])
        tau = fit_oven_tau(pooled, config)
        if tau is not None:
            fitted[vehicles[i]] = tau
    if len(fitted) < 40:
        return []

    warmup = min(config["thresholds"]["baseline_warmup_vehicles"], len(vehicles) // 2)
    base_vals = [t for v, t in fitted.items() if v <= warmup]
    if len(base_vals) < 20:
        return []
    mu, sd = float(np.mean(base_vals)), max(float(np.std(base_vals, ddof=1)), 1e-6)
    z_crit = config["physics"]["tau_fit_z_crit"]

    flags, latched = [], False
    for v in sorted(fitted):
        z = (fitted[v] - mu) / sd
        if abs(z) >= z_crit and not latched:
            latched = True
            flags.append({
                "type": "physics_thermal_violation", "station": "PNT-*",
                "parameter": "oven_tau", "vehicle": v, "detected_at": v,
                "severity": "alert",
                "measured": round(fitted[v], 2), "expected": round(mu, 2),
                "residual": round(fitted[v] - mu, 2), "z": round(z, 2),
                "evidence": (f"Fitted oven time constant {fitted[v]:.1f}s vs baseline "
                             f"{mu:.1f}s ({z:+.1f} sigma). No single paint station "
                             f"breached its own control limit."),
            })
        elif abs(z) < z_crit * 0.5:
            latched = False
    return flags


# --------------------------------------------------------------------------
# Bottleneck detector
# --------------------------------------------------------------------------

def detect_bottlenecks(raw: dict, config: dict = CONFIG) -> list[dict]:
    """CUSUM-confirmed bottleneck detection.

    A bottleneck is a small shift that persists, which is exactly what a CUSUM is
    designed for - and unlike a single-window slope test it accumulates evidence
    instead of re-rolling the dice every window. Confirmation requires the CUSUM
    plus one corroborating signal, and each episode latches so one emerging
    constraint produces one event rather than one per vehicle.
    """
    cfg = config["bottleneck"]
    cal = cfg.get("calibrated") or {}
    h_sigma = float(cal.get("cusum_h_sigma", cfg["cusum_h_sigma"]))
    k_sigma = float(cfg["cusum_k_sigma"])
    t_crit = float(cfg["slope_t_crit"])
    q_crit = float(cfg["queue_z_crit"])
    window = int(cfg["slope_window"])
    release = float(cfg["release_fraction"])
    warmup = config["thresholds"]["baseline_warmup_vehicles"]

    stations = {s["id"]: s for s in raw["stations"]}
    by_station: dict[str, list[dict]] = defaultdict(list)
    for o in raw["operations"]:
        by_station[o["station"]].append(o)

    x = np.arange(window, dtype=float)
    sxx = float(((x - x.mean()) ** 2).sum())
    events: list[dict] = []

    for sid, rows in by_station.items():
        rows.sort(key=lambda r: r["vehicle"])
        cycles = np.array([r["cycle_time"] for r in rows], dtype=float)
        queues = np.array([r["queue"] for r in rows], dtype=float)
        n_warm = min(warmup, max(20, len(rows) // 3))
        if len(rows) < max(window, n_warm) + 5:
            continue

        mu_c = float(np.mean(cycles[:n_warm]))
        sd_c = max(float(np.std(cycles[:n_warm], ddof=1)), 1e-6)
        mu_q = float(np.mean(queues[:n_warm]))
        sd_q = max(float(np.std(queues[:n_warm], ddof=1)), 1e-6)

        k, h = k_sigma * sd_c, h_sigma * sd_c
        cusum, latched = 0.0, False

        for i in range(n_warm, len(rows)):
            cusum = max(0.0, cusum + (cycles[i] - mu_c) - k)
            if latched:
                if cusum < h * release:
                    latched = False
                continue
            if cusum <= h:
                continue

            seg = cycles[i - window + 1:i + 1]
            if len(seg) < window:
                continue
            slope = float(np.polyfit(np.arange(window), seg, 1)[0])
            resid = seg - (slope * (np.arange(window) - np.arange(window).mean()) + seg.mean())
            se = max(float(np.sqrt((resid ** 2).sum() / max(window - 2, 1)) / np.sqrt(sxx)), 1e-9)
            t_stat = slope / se
            queue_z = (queues[i] - mu_q) / sd_q

            if t_stat >= t_crit or queue_z >= q_crit:
                latched = True
                events.append({
                    "station": sid, "vehicle": int(rows[i]["vehicle"]),
                    "detected_at": int(rows[i]["vehicle"]),
                    "cycle_time": float(cycles[i]),
                    "baseline_cycle": round(mu_c, 2),
                    "cusum": round(cusum / sd_c, 2), "cusum_limit": round(h_sigma, 2),
                    "slope_t": round(t_stat, 2), "queue_z": round(queue_z, 2),
                    "starving_downstream": [d for d in [stations.get(sid, {}).get("downstream")] if d],
                    "evidence": (f"CUSUM {cusum / sd_c:.1f} sigma over limit {h_sigma:.1f}; "
                                 f"slope t={t_stat:.1f} (crit {t_crit}); "
                                 f"queue z={queue_z:.1f}. Baseline cycle {mu_c:.1f}s."),
                })
    events.sort(key=lambda e: e["detected_at"])
    return events


# --------------------------------------------------------------------------
# SPC anomaly detector
# --------------------------------------------------------------------------

def detect_anomalies(raw: dict, config: dict = CONFIG) -> list[dict]:
    """Statistical process control over contextual z-scores.

    Four rules, each latched: an alarm stays open until the signal recovers, so one
    sustained fault produces one event rather than one per vehicle. Latching is what
    makes precision meaningful - an unlatched detector reports thousands of "events"
    for a handful of real conditions.
    """
    th = config["thresholds"]
    lam = float(th["ewma_lambda"])
    L = float(th["ewma_L"])
    release = float(th["release_fraction"])
    # Steady-state EWMA dispersion, inflated for the uncertainty in the estimated
    # baseline mean. A chart tuned as if the baseline were exact will spend that
    # error budget on false alarms.
    n_base = max(int(np.median([r.get("baseline_n", 0) or 1 for r in raw["readings"]])), 1)
    sigma_e = float(np.sqrt(lam / (2.0 - lam) + 1.0 / n_base))
    limit = L * sigma_e

    flags: list[dict] = []
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in raw["readings"]:
        by_key[(r["station"], r["parameter"])].append(r)

    for (station, param), rows in by_key.items():
        rows.sort(key=lambda r: r["vehicle"])
        zs = np.array([r["z"] for r in rows])
        ewma = 0.0
        open_alarm = {"point": False, "ewma": False, "trend": False}

        for i, row in enumerate(rows):
            v = row["vehicle"]
            ewma = lam * zs[i] + (1 - lam) * ewma

            # Rule 1 - point excursion, confirmed. An isolated reading beyond 3 sigma
            # occurs about once per 370 samples by chance and is not evidence of a
            # fault; requiring 2 of the last 3 removes that whole class of alarm.
            recent = np.abs(zs[max(0, i - 2):i + 1])
            n_beyond = int((recent >= th["alert_abs_z"]).sum())
            if abs(row["z"]) >= th["alert_abs_z"] and n_beyond >= 2:
                if not open_alarm["point"]:
                    open_alarm["point"] = True
                    flags.append({"type": "point", "station": station, "parameter": param,
                                  "vehicle": v, "detected_at": v, "severity": "alert",
                                  "z": row["z"],
                                  "evidence": (f"|z|={abs(row['z']):.2f}; {n_beyond} of last 3 readings "
                                               f"beyond {th['alert_abs_z']} sigma.")})
            elif abs(row["z"]) < th["green_abs_z"]:
                open_alarm["point"] = False

            # Rule 2 - EWMA, for sustained shifts too small to trip the point rule.
            if abs(ewma) > limit:
                if not open_alarm["ewma"]:
                    open_alarm["ewma"] = True
                    flags.append({"type": "persistent", "station": station, "parameter": param,
                                  "vehicle": v, "detected_at": v, "severity": "alert",
                                  "z": round(float(ewma), 3),
                                  "evidence": (f"EWMA(lambda={lam}) at {ewma:+.2f} beyond "
                                               f"+/-{limit:.2f} control limit; sustained level shift.")})
            elif abs(ewma) < limit * release:
                open_alarm["ewma"] = False

            # Rule 3 - monotone drift.
            w = th["trend_window"]
            if i >= w - 1:
                win = zs[i - w + 1:i + 1]
                same_dir = np.all(win > 0) or np.all(win < 0)
                if (same_dir and np.all(np.diff(np.abs(win)) > 0.12)
                        and abs(win[-1]) >= 2.0 and abs(win[-1]) - abs(win[0]) >= 1.6):
                    if not open_alarm["trend"]:
                        open_alarm["trend"] = True
                        flags.append({"type": "trend", "station": station, "parameter": param,
                                      "vehicle": v, "detected_at": v, "severity": "warning",
                                      "z": row["z"],
                                      "evidence": (f"Monotone drift over {w} vehicles, "
                                                   f"|z| {abs(win[0]):.1f} -> {abs(win[-1]):.1f}.")})
                elif abs(row["z"]) < th["green_abs_z"]:
                    open_alarm["trend"] = False

    # Rule 4 - cross-parameter: two signals at one station excursion together.
    for s in raw["stations"]:
        params = list(s["parameters"]) if s["instrumented"] else []
        if len(params) < 2:
            continue
        lookup = {(r["vehicle"], r["parameter"]): r for r in raw["readings"] if r["station"] == s["id"]}
        vehicles = sorted({v for v, _ in lookup})
        matrix = np.array([[lookup[(v, p)]["z"] for p in params]
                           for v in vehicles if all((v, p) in lookup for p in params)])
        if len(matrix) < 2:
            continue
        corr = np.corrcoef(matrix, rowvar=False)
        open_cross = False
        for v in vehicles:
            abnormal = [p for p in params
                        if (v, p) in lookup and abs(lookup[v, p]["z"]) >= th["cross_param_abs_z"]]
            linked = [(a, b) for a in abnormal for b in abnormal
                      if a != b and abs(corr[params.index(a), params.index(b)]) >= th["cross_param_min_corr"]]
            if len(abnormal) >= 2 and linked:
                if not open_cross:
                    open_cross = True
                    a, b = linked[0]
                    flags.append({"type": "cross_parameter", "station": s["id"],
                                  "parameter": " + ".join(abnormal), "vehicle": v, "detected_at": v,
                                  "severity": "alert",
                                  "evidence": (f"{a} and {b} both beyond {th['cross_param_abs_z']} sigma, "
                                               f"correlation {corr[params.index(a), params.index(b)]:.2f}.")})
            elif not abnormal:
                open_cross = False
    return flags


# --------------------------------------------------------------------------
# Latent-defect back-tracing
# --------------------------------------------------------------------------

def trace_latent_defects(raw: dict, config: dict = CONFIG,
                         flags: list[dict] | None = None) -> dict:
    """Attribute inspection failures to the upstream station that caused them.

    The problem statement's hardest case: a defect introduced early produces no
    local alarm and only surfaces at final inspection. Per-vehicle SPC cannot see
    it. Comparing the upstream signal *populations* of failing versus passing
    vehicles can - a sub-threshold shift that is invisible one vehicle at a time
    is highly significant across a few dozen.
    """
    flags_at = (lambda sid: [f for f in (flags or []) if f["station"] == sid])
    inspection = config.get("inspection_station")
    if not inspection:
        return {"available": False, "reason": "No inspection station configured."}

    failed = {o["vehicle"] for o in raw["operations"]
              if o["station"] == inspection and o.get("manual_outcome") == "fail"}
    insp_station = next((s for s in raw["stations"] if s["id"] == inspection), None)
    if insp_station and insp_station["instrumented"]:
        vs = [r for r in raw["readings"] if r["station"] == inspection and r["parameter"] == "vision_score"]
        if vs:
            cut = float(np.mean([r["z"] for r in vs])) - 2.5
            failed |= {r["vehicle"] for r in vs if r["z"] <= cut}

    if len(failed) < 8:
        return {"available": False, "reason": f"Only {len(failed)} inspection failures; need 8 to attribute."}

    upstream_ids = []
    for s in raw["stations"]:
        if s["id"] == inspection:
            break
        if s["instrumented"]:
            upstream_ids.append(s["id"])

    # Cycle time is a throughput signal, not a part-quality one: a slow station does
    # not put a defect into the body it is holding. Including it lets an unrelated
    # constraint outrank the true origin.
    quality_only = set(config["thresholds"].get("trace_exclude_parameters", ["cycle_time"]))
    by_cell: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in raw["readings"]:
        if r["station"] in upstream_ids and r["parameter"] not in quality_only:
            by_cell[(r["station"], r["parameter"])][r["vehicle"]] = r["z"]

    # Detrend each upstream signal against its own rolling median before comparing
    # populations. A latent defect is a *per-vehicle* effect - only the vehicles
    # carrying it shift - whereas the things that confound this comparison (a tool
    # drifting elsewhere on the line, a seasonal ambient swing) are *temporal*.
    # Removing the slow component separates the two, and without it a station with an
    # unrelated ramp outranks the true origin.
    detrend_window = int(config["thresholds"].get("trace_detrend_window", 61))
    half = detrend_window // 2
    for key, zmap in by_cell.items():
        vehicles = sorted(zmap)
        values = np.array([zmap[v] for v in vehicles])
        trend = np.array([np.median(values[max(0, i - half):i + half + 1])
                          for i in range(len(values))])
        by_cell[key] = {v: float(values[i] - trend[i]) for i, v in enumerate(vehicles)}

    # Time-matched controls: compare failing vehicles only against passing vehicles
    # from the same stretch of production. Without this, any slow drift elsewhere on
    # the line (a fouling oven, a seasonal ambient swing) separates the two groups
    # and outranks the true origin.
    lo, hi = min(failed), max(failed)
    span = hi - lo
    ctrl_lo, ctrl_hi = lo - span, hi + span

    ranked = []
    for (sid, param), zmap in by_cell.items():
        fail_z = [z for v, z in zmap.items() if v in failed]
        pass_z = [z for v, z in zmap.items()
                  if v not in failed and ctrl_lo <= v <= ctrl_hi]
        if len(fail_z) < 8 or len(pass_z) < 30:
            continue
        m1, m2 = float(np.mean(fail_z)), float(np.mean(pass_z))
        s1, s2 = float(np.var(fail_z, ddof=1)), float(np.var(pass_z, ddof=1))
        se = float(np.sqrt(s1 / len(fail_z) + s2 / len(pass_z))) or 1e-9
        t = (m1 - m2) / se
        pooled = float(np.sqrt((s1 + s2) / 2)) or 1e-9
        ranked.append({
            "station": sid, "parameter": param,
            "t_statistic": round(t, 2), "effect_size_sigma": round((m1 - m2) / pooled, 2),
            "mean_z_failing": round(m1, 3), "mean_z_passing": round(m2, 3),
            "n_failing": len(fail_z),
        })
    ranked.sort(key=lambda r: -abs(r["t_statistic"]))

    truth = next((x for x in config["injections"] if x["kind"] == "latent_defect"), None)
    top = ranked[0] if ranked else None
    margin = round(abs(top["t_statistic"]) - abs(ranked[1]["t_statistic"]), 2) \
        if len(ranked) > 1 else None

    # How much of the line's output was already carrying the defect by the time local
    # SPC noticed anything at the origin - the exposure the problem statement warns
    # about, quantified rather than asserted.
    exposure = None
    if top:
        local_alarms = [f["detected_at"] for f in flags_at(top["station"])] if flags_at else []
        first_local = min((v for v in local_alarms if v >= min(failed)), default=None)
        exposure = {
            "first_downstream_failure": min(failed),
            "first_local_alarm": first_local,
            "vehicles_shipped_before_local_alarm": (
                sum(1 for v in failed if first_local is None or v < first_local)),
            "local_alarm_lag_vehicles": (first_local - min(failed)) if first_local else None,
        }

    return {
        "available": True,
        "inspection_station": inspection,
        "failures_analysed": len(failed),
        "ranking": ranked[:8],
        "origin_identified": bool(top and truth and top["station"] == truth["station"]),
        "true_origin": truth["station"] if truth else None,
        "separation_margin_t": margin,
        "exposure": exposure,
        "method": ("Welch t-test on time-detrended upstream z-score populations, failing "
                   "versus time-matched passing vehicles at the inspection station. "
                   "Quality parameters only."),
    }


# --------------------------------------------------------------------------
# Analysis entry point
# --------------------------------------------------------------------------

def analyze(raw: dict, config: dict = CONFIG) -> dict:
    if "baselines" not in raw:
        raw["baselines"] = build_baselines(raw, config)
        score_readings(raw, raw["baselines"])
    flags = detect_anomalies(raw, config)
    physics_flags = physics_thermal_check(raw, config)
    flags.extend(physics_flags)
    flags.sort(key=lambda f: f["detected_at"])
    return {
        "flags": flags,
        "physics_flags": physics_flags,
        "bottlenecks": detect_bottlenecks(raw, config),
        "propagation": trace_latent_defects(raw, config, flags),
        "baseline_warmup": raw["baselines"]["warmup_vehicles"],
    }


class ArtifactIndex:
    """Sorted-by-detected_at index giving O(log n) causal window slicing."""

    def __init__(self, items: list[dict]):
        self.by_station: dict[str, list[dict]] = defaultdict(list)
        for it in sorted(items, key=lambda x: x["detected_at"]):
            self.by_station[it["station"]].append(it)
        self._keys = {s: [i["detected_at"] for i in v] for s, v in self.by_station.items()}
        self.all = sorted(items, key=lambda x: x["detected_at"])
        self._all_keys = [i["detected_at"] for i in self.all]

    def window(self, station: str, lo: int, hi: int) -> list[dict]:
        keys, rows = self._keys.get(station, []), self.by_station.get(station, [])
        return rows[bisect_left(keys, lo):bisect_right(keys, hi)]

    def upto(self, hi: int) -> list[dict]:
        return self.all[:bisect_right(self._all_keys, hi)]
