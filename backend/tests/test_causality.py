"""Causality invariants.

These are the tests that make the look-ahead defect structurally impossible rather
than merely fixed once. A twin that shows vehicle t must only know what was knowable
at t; if that ever stops holding, the build fails here.
"""
from __future__ import annotations

import pytest

from metrics import backtest
from state import build_state
from twin import CONFIG, ArtifactIndex, analyze, simulate


@pytest.fixture(scope="module")
def run():
    raw = simulate(CONFIG, seed=42)
    analysis = analyze(raw, CONFIG)
    metrics = backtest(raw, analysis, CONFIG)
    indices = {"flags": ArtifactIndex(analysis["flags"]),
               "bottlenecks": ArtifactIndex(analysis["bottlenecks"])}
    return raw, analysis, metrics, indices


def _state(run, t):
    raw, analysis, metrics, indices = run
    return build_state(raw, analysis, metrics, None, upto_vehicle=t, indices=indices)


CHECKPOINTS = [1, 5, 25, 100, 180, 220, 260, 300, 340, 380, 420]


def test_every_artifact_carries_detected_at(run):
    _, analysis, _, _ = run
    for item in analysis["flags"] + analysis["bottlenecks"]:
        assert "detected_at" in item, f"missing detected_at: {item}"
        assert isinstance(item["detected_at"], int)


@pytest.mark.parametrize("t", CHECKPOINTS)
def test_no_future_artifact_is_visible(run, t):
    """The core invariant: nothing detected after t may appear in state(t)."""
    st = _state(run, t)
    for f in st["flags"]:
        assert f["detected_at"] <= t, f"flag from the future at t={t}: {f}"
    for b in st["bottlenecks"]:
        assert b["detected_at"] <= t, f"bottleneck from the future at t={t}: {b}"
    for s in st["stations"]:
        for f in s["open_flags"]:
            assert f["detected_at"] <= t, f"station {s['id']} shows a future flag at t={t}"


@pytest.mark.parametrize("t", CHECKPOINTS[:-1])
def test_visible_artifacts_are_monotone(run, t):
    """State accumulates. What was visible at t stays visible at t+1."""
    later = CHECKPOINTS[CHECKPOINTS.index(t) + 1]
    now = {(f["station"], f["type"], f["detected_at"]) for f in _state(run, t)["flags"]}
    then = {(f["station"], f["type"], f["detected_at"]) for f in _state(run, later)["flags"]}
    assert now <= then, f"state(t={t}) is not a subset of state(t={later})"


@pytest.mark.parametrize("t", CHECKPOINTS)
def test_alert_status_is_backed_by_visible_evidence(run, t):
    """A red station must have a reason the operator can also see.

    This is the test that would have caught the original defect, where the line map
    showed five stations in alert while the alarm list was empty.
    """
    st = _state(run, t)
    visible = {f["station"] for f in st["flags"]} | {b["station"] for b in st["bottlenecks"]}
    for s in st["stations"]:
        if s["status"] != "alert":
            continue
        has_own = bool(s["open_flags"]) or s["id"] in visible
        manual = s["operation"] and s["operation"].get("manual_outcome") in {"fail", "rework"}
        inferred = s["gap_method"] == "adjacent-signal inference"
        assert has_own or manual or inferred, (
            f"station {s['id']} is alert at t={t} with no visible evidence")


def test_baseline_window_precedes_every_injected_fault():
    """Baselines must be estimated from fault-free production.

    A fault active during the Phase I window is absorbed into the baseline and becomes
    undetectable - so the warm-up must end before the earliest injection.
    """
    warm = CONFIG["thresholds"]["baseline_warmup_vehicles"]
    for inj in CONFIG["injections"]:
        start = inj.get("start_vehicle") or min(inj.get("vehicles", [10**9]))
        assert start > warm, f"injection {inj['id']} starts at {start}, inside the Phase I window ({warm})"


def test_station_risk_moves_with_the_replay(run):
    """Station risk must be re-sliced per request, not computed once over the run."""
    raw, analysis, metrics, indices = run
    from ml_risk import score
    ml = score(raw, analysis, CONFIG)
    values = set()
    for t in (60, 150, 240, 330, 420):
        st = build_state(raw, analysis, metrics, ml, upto_vehicle=t, indices=indices)
        values.add(tuple(s["defect_risk_pct"] for s in st["stations"]))
    assert len(values) > 1, "station risk is identical at every replay position"


def test_risk_is_not_a_single_constant(run):
    """Guards the original defect where all 40 stations reported exactly 89.0%.

    Risk collapsing toward zero once faults clear is correct behaviour, so the spread
    is asserted across the replay rather than at one arbitrary endpoint.
    """
    raw, analysis, metrics, indices = run
    from ml_risk import score
    ml = score(raw, analysis, CONFIG)
    seen = set()
    peak = 0.0
    for t in (150, 240, 280, 330, 380, 420):
        st = build_state(raw, analysis, metrics, ml, upto_vehicle=t, indices=indices)
        vals = [s["defect_risk_pct"] for s in st["stations"] if s["defect_risk_pct"] is not None]
        seen.update(vals)
        peak = max(peak, max(vals, default=0.0))
    assert len(seen) >= 8, f"only {len(seen)} distinct risk values across the whole replay"
    assert peak > 50.0, f"risk never rises above {peak}% even during an injected fault"
    assert min(seen) < 10.0, "risk never falls to a quiet level"
