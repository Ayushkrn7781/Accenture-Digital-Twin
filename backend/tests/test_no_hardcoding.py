"""Guards against the defect class this rebuild existed to remove.

The original build displayed numbers no computation produced: a constant 89% risk, a
health bar whose width encoded only a three-state status, `?? 5.0` fallbacks that
invented a value whenever data was missing, and a hardcoded ROI that disagreed with
the PDF beside it. A one-time cleanup regresses; these tests make it stay clean.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
FRONTEND = BACKEND.parent / "frontend" / "src"
JSX = FRONTEND / "main.jsx"


# ---------------------------------------------------------------- frontend

def test_no_invented_numeric_fallbacks():
    """`value ?? 5.0` and `value || 68` silently fabricate data when the API returns
    nothing. Absent data must render as absent."""
    src = JSX.read_text()
    offenders = []
    for i, line in enumerate(src.splitlines(), 1):
        if line.lstrip().startswith('//') or line.lstrip().startswith('*'):
            continue
        for m in re.finditer(r"(\?\?|\|\|)\s*(-?\d+\.?\d*)\b", line):
            # A zero default is a neutral identity, not an invented reading.
            if float(m.group(2)) != 0:
                offenders.append(f"{JSX.name}:{i}: {line.strip()}")
    assert not offenders, "numeric fallbacks fabricate data:\n" + "\n".join(offenders)


def test_no_currency_or_roi_literals_in_ui():
    """Money and ROI belong to the economics module, which derives them from measured
    detector performance and shares one source with the PDF proposal."""
    src = JSX.read_text()
    hits = re.findall(r"\$\s?\d[\d,]*\s?[kKmM]?\b", src)
    hits += re.findall(r"\b\d{1,3}%\s*(?:reduction|boost|improvement|savings|ROI)", src, re.I)
    assert not hits, f"hardcoded commercial figures in the UI: {hits}"


def test_control_limits_come_from_the_api():
    """The chart claimed limits it never drew. It must read them from thresholds."""
    src = JSX.read_text()
    assert "ReferenceLine" in src, "control chart draws no control limits"
    assert "th.alert_abs_z" in src and "th.green_abs_z" in src, \
        "control limits are not taken from the API thresholds"
    # A 0-1 axis on the calibration reliability plot is the probability scale itself,
    # not a clipping bound; anything else fixed would clip a real excursion.
    # Only fully-literal domains clip data. "[-bound, bound]" is computed from the
    # series; "[0, 1]" is the probability scale on the reliability plot.
    fixed = [m for m in re.findall(r"domain=\{\[([^\]]+)\]\}", src)
             if re.fullmatch(r"\s*-?[\d.]+\s*,\s*-?[\d.]+\s*", m)
             and m.replace(" ", "") != "0,1"]
    assert not fixed, f"chart y-domain is hardcoded and will clip real excursions: {fixed}"


def test_health_bar_encodes_the_health_score():
    src = JSX.read_text()
    assert "s.health_score}%" in src.replace(" ", ""), \
        "health bar width must be the health score, not a status-derived constant"


def test_computed_panels_are_actually_rendered():
    """Everything the backend computes is either shown or is not computed at all."""
    src = JSX.read_text()
    for key in ["bottlenecks", "throughput_loss", "risk_drivers", "propagation",
                "sensor_gap", "rollout", "retrofit", "reliability"]:
        assert key in src, f"backend computes '{key}' but the UI never reads it"


# ----------------------------------------------------------------- backend

def test_detector_limits_are_config_driven():
    """No detector may compare a statistic against a bare literal."""
    src = (BACKEND / "twin.py").read_text()
    for fn in ("def detect_anomalies", "def detect_bottlenecks"):
        body = src[src.index(fn):]
        body = body[:body.index("\ndef ", 10)]
        bad = re.findall(r"(?:>=|<=|>|<)\s*(\d+\.\d+)", body)
        allowed = {"0.0", "0.12", "1.6", "2.0"}   # loop guards and shape constants
        unexpected = [b for b in bad if b not in allowed]
        assert not unexpected, f"{fn} compares against literals {unexpected}"


def test_calibration_provenance_is_recorded():
    """Thresholds must carry evidence of how they were chosen."""
    cfg = json.loads((BACKEND / "config" / "line_config.json").read_text())
    cal = cfg["bottleneck"].get("calibrated")
    assert cal, "thresholds have no calibration record - run calibrate.py"
    for field in ("method", "seeds", "budget_false_alarms_per_run", "arl0_vehicles",
                  "measured_false_alarms_per_run_heldout", "calibrated_on"):
        assert field in cal, f"calibration record is missing '{field}'"
    assert cfg["thresholds"]["ewma_L"] == cal["ewma_L"], "config drifted from its calibration"
    assert cfg["bottleneck"]["cusum_h_sigma"] == cal["cusum_h_sigma"]


def test_model_identity_is_not_asserted():
    """The UI once displayed 'XGBoost 3.2' as a literal while running a fallback."""
    ml = (BACKEND / "ml_risk.py").read_text()
    assert "xgb.__version__" in ml, "model version must be read from the library"
    assert not re.search(r'"XGBoost \d', ml), "model name/version is hardcoded"
    assert "AI4I" not in ml, "AI4I training claim is not supported by this code"
    assert "AI4I" not in JSX.read_text(), "AI4I training claim appears in the UI"


def test_shap_is_claimed_only_when_computed():
    """'SHAP weights' labelled gain-based importances. Attribution names its method."""
    ml = (BACKEND / "ml_risk.py").read_text()
    src = JSX.read_text()
    assert "pred_contribs" in ml, "no exact TreeSHAP path"
    assert "attribution_method" in ml and "attribution_method" in src, \
        "the UI must render whichever attribution method actually ran"
    assert not re.search(r"SHAP", src), "UI hardcodes a SHAP claim instead of reporting the method used"


def test_physics_model_uses_its_stated_parameters():
    """The old check called itself Newton's cooling and computed abs(T-178) > 12,
    leaving ambient and the cooling constant unused."""
    src = (BACKEND / "twin.py").read_text()
    fn = src[src.index("def thermal_model_C"):]
    fn = fn[:fn.index("\ndef ", 10)]
    assert "np.exp" in fn, "thermal model has no exponential term"
    assert "ambient" in fn and "setpoint" in fn, "thermal model ignores its own parameters"


@pytest.mark.parametrize("field", ["precision", "recall", "bottleneck_precision",
                                   "bottleneck_recall", "median_detection_lag",
                                   "p90_detection_lag"])
def test_backtest_reports_precision_not_just_recall(field):
    """Reporting recall alone let a detector firing on 40/40 stations score 100%."""
    from metrics import backtest
    from twin import CONFIG, analyze, simulate
    raw = simulate(CONFIG, seed=42)
    assert field in backtest(raw, analyze(raw, CONFIG), CONFIG)
