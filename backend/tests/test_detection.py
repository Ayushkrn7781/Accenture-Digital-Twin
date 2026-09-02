"""Detector behaviour: does it find real faults, and stay quiet otherwise."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from metrics import backtest, false_alarm_profile
from twin import CONFIG, analyze, simulate

BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def demo():
    raw = simulate(CONFIG, seed=42)
    return raw, analyze(raw, CONFIG)


def test_all_injected_faults_are_recovered(demo):
    raw, analysis = demo
    bt = backtest(raw, analysis, CONFIG)
    assert bt["recall"] == 100.0, f"missed anomaly scenarios: {bt}"
    assert bt["bottleneck_recall"] == 100.0
    assert bt["latent_origin_identified"], "latent-defect origin not recovered"


def test_bottleneck_detector_is_specific(demo):
    """It once fired on 40 of 40 stations. Only injected constraints should confirm."""
    raw, analysis = demo
    injected = {i["station"] for i in CONFIG["injections"] if i["kind"] == "bottleneck"}
    detected = {b["station"] for b in analysis["bottlenecks"]}
    assert injected <= detected, f"missed {injected - detected}"
    assert len(detected) <= len(injected) + 2, f"over-firing on {detected}"


def test_one_event_per_episode(demo):
    """Alarms latch. A sustained fault is one event, not one per vehicle."""
    _, analysis = demo
    for station in {b["station"] for b in analysis["bottlenecks"]}:
        n = sum(1 for b in analysis["bottlenecks"] if b["station"] == station)
        assert n <= 3, f"{station} raised {n} bottleneck events for one condition"


def test_false_alarm_floor_meets_budget():
    """The headline trust number, measured on runs the thresholds were not fitted to."""
    fa = false_alarm_profile(CONFIG, seeds=6)
    budget = CONFIG["bottleneck"]["calibrated"]["budget_false_alarms_per_run"]
    assert fa["total_false_alarms_per_run"] <= budget * 3, (
        f"false-alarm floor {fa['total_false_alarms_per_run']}/run far exceeds the "
        f"{budget}/run budget — re-run calibrate.py")
    assert fa["arl0_vehicles"] > CONFIG["line"]["vehicle_count"] / 6


def test_noise_is_not_truncated():
    """The original simulator clipped noise at 2.7 sigma, so a 3-sigma rule could not
    fire at all. That manufactures a flattering false-alarm rate."""
    raw = simulate(CONFIG, seed=7)
    analysis = analyze(raw, CONFIG)
    del analysis
    zs = [abs(r["z"]) for r in raw["readings"]]
    assert max(zs) > 3.5, "noise distribution appears truncated below the control limit"


def test_physics_detector_is_orthogonal_to_spc(demo):
    """Oven fouling must be caught by the thermal model while every individual paint
    station stays inside its own control limits - otherwise the physics check is just
    a second copy of the z-test."""
    raw, analysis = demo
    assert analysis["physics_flags"], "thermal tau drift was not detected"

    drift = next(i for i in CONFIG["injections"] if i["kind"] == "thermal_tau_drift")
    window = range(drift["start_vehicle"], drift["end_vehicle"] + 1)
    paint = {s["id"] for s in raw["stations"] if s["area"] == "Paint"}
    point_stations = {i["station"] for i in CONFIG["injections"] if i["kind"] == "point"}

    # The systematic shift the drift imposes on each station - isolated noise
    # excursions are what the control limits are there to tolerate.
    per_station: dict[str, list[float]] = {}
    for r in raw["readings"]:
        if (r["station"] in paint and r["station"] not in point_stations
                and r["parameter"] == "temperature" and r["vehicle"] in window):
            per_station.setdefault(r["station"], []).append(r["z"])
    worst = max(abs(sum(v) / len(v)) for v in per_station.values())
    assert worst < CONFIG["thresholds"]["green_abs_z"], (
        f"tau drift shifts a station by {worst:.2f} sigma, past the warning limit - "
        f"SPC would have caught it and the two detectors are not independent")


def test_latent_defect_back_trace_beats_local_spc(demo):
    """The problem statement's hardest case, made concrete.

    A defect introduced upstream produces a sub-threshold local signal and only
    surfaces at final inspection. Local SPC does eventually notice - but only after a
    run of vehicles has already shipped carrying it. The back-trace names the origin
    from the failure population instead of waiting for the origin to alarm.
    """
    raw, analysis = demo
    ld = next(i for i in CONFIG["injections"] if i["kind"] == "latent_defect")
    prop = analysis["propagation"]

    assert prop["available"], prop.get("reason")
    assert prop["origin_identified"], f"ranked {prop['ranking'][0]} instead of {ld['station']}"
    assert prop["ranking"][0]["station"] == ld["station"]
    assert prop["separation_margin_t"] > 0.5, (
        f"origin is only {prop['separation_margin_t']} t ahead of the next candidate")

    exp = prop["exposure"]
    assert exp["first_downstream_failure"] >= ld["start_vehicle"]
    assert exp["vehicles_shipped_before_local_alarm"] > 0, (
        "local SPC caught this immediately, so the back-trace adds nothing")


def test_latent_defect_signal_is_sub_threshold_at_origin(demo):
    """No point alarm may fire at the origin - the signal is deliberately below the
    per-reading control limit, which is what makes population attribution necessary."""
    raw, analysis = demo
    ld = next(i for i in CONFIG["injections"] if i["kind"] == "latent_defect")
    window = range(ld["start_vehicle"], ld["end_vehicle"] + 1)
    points = [f for f in analysis["flags"]
              if f["station"] == ld["station"] and f["type"] == "point"
              and f.get("parameter") == ld["parameter"] and f["detected_at"] in window]
    assert not points, f"origin breached the point control limit: {points}"


def test_uploaded_data_without_ground_truth_reports_unavailable():
    """It must not borrow the simulation's metrics."""
    raw = simulate(CONFIG, seed=42)
    raw["source"] = "uploaded CSV"
    bt = backtest(raw, analyze(raw, CONFIG), CONFIG)
    assert bt["validation_available"] is False
    assert bt["precision"] is None and bt["recall"] is None


def test_generalises_to_a_different_line():
    """Same code, a structurally different plant: fewer stations, different mix,
    poorer instrumentation, its own faults. Configuration, not custom code."""
    cfg = json.loads((BACKEND / "config" / "line_config_plant_b.json").read_text())
    raw = simulate(cfg, seed=42)
    analysis = analyze(raw, cfg)
    bt = backtest(raw, analysis, cfg)

    assert len(raw["stations"]) == 28 and raw["vehicle_count"] == 360
    assert bt["recall"] == 100.0, f"plant B anomaly recall {bt['recall']}"
    assert bt["bottleneck_recall"] == 100.0, f"plant B bottleneck recall {bt['bottleneck_recall']}"
    assert bt["precision"] is not None and bt["precision"] >= 50.0


def test_manual_stations_never_report_invented_readings():
    raw = simulate(CONFIG, seed=42)
    manual = {s["id"] for s in raw["stations"] if not s["instrumented"]}
    assert manual, "config has no manual stations to test"
    assert not [r for r in raw["readings"] if r["station"] in manual], \
        "a manual-only station produced continuous sensor readings"
