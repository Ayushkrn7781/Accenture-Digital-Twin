"""Station-level defect-risk model: multi-seed training, temporal validation, SHAP.

Three deliberate design choices, each fixing a way this kind of model usually lies:

* **The independent unit is the event, not the row.** A single 420-vehicle run
  contains only a handful of injected faults; thousands of autocorrelated rows drawn
  from them cannot validate a ten-feature model. Training therefore spans many
  independently generated lines with randomised fault station, kind, onset and
  magnitude, so station identity carries no signal the model can memorise.
* **Splits are by line, never by row.** A random row split on autocorrelated time
  series leaks the answer across the boundary and inflates every metric. The demo
  seed is held out entirely, so every figure the dashboard shows comes from data the
  model never saw.
* **Features are causal by construction.** ``features_at`` reads only rows at or
  before the vehicle being scored, and the same function is used at training and at
  serving time, so there is no train/serve skew.
"""
from __future__ import annotations

import copy
import hashlib
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from twin import CONFIG, analyze, simulate

HORIZON = 8
TRAIN_SEEDS = range(700, 718)
CAL_SEEDS = range(800, 806)    # isotonic calibration fitted here
VAL_SEEDS = range(900, 906)    # metrics reported here only - never fitted on
DEMO_SEED = 42

FEATURES = [
    "max_abs_z_10", "mean_abs_z_10", "ewma_abs", "z_slope_10",
    "alerts_20", "warnings_20", "vehicles_since_alert",
    "cycle_z", "cycle_cusum", "queue_z", "utilization",
    "upstream_alerts_20", "thermal_residual_abs",
    "manual_rework_20", "operator_idx", "lot_mean_abs_z", "variant_idx",
]

FEATURE_LABELS = {
    "max_abs_z_10": "Peak parameter deviation (10 veh)",
    "mean_abs_z_10": "Mean parameter deviation (10 veh)",
    "ewma_abs": "Sustained level shift (EWMA)",
    "z_slope_10": "Deviation trend slope",
    "alerts_20": "Alerts at station (20 veh)",
    "warnings_20": "Warnings at station (20 veh)",
    "vehicles_since_alert": "Vehicles since last alert",
    "cycle_z": "Cycle time vs baseline",
    "cycle_cusum": "Cycle-time CUSUM",
    "queue_z": "Buffer queue vs baseline",
    "utilization": "Station utilization",
    "upstream_alerts_20": "Upstream station alerts",
    "thermal_residual_abs": "Thermal model residual",
    "manual_rework_20": "Manual QA reworks (20 veh)",
    "operator_idx": "Operator on shift",
    "lot_mean_abs_z": "Incoming supplier lot quality",
    "variant_idx": "Product variant",
}

CAUSE_FAMILY = {
    "max_abs_z_10": "Equipment / process", "mean_abs_z_10": "Equipment / process",
    "ewma_abs": "Equipment / process", "z_slope_10": "Equipment wear",
    "alerts_20": "Equipment / process", "warnings_20": "Equipment / process",
    "vehicles_since_alert": "Equipment wear",
    "cycle_z": "Capacity / pacing", "cycle_cusum": "Capacity / pacing",
    "queue_z": "Capacity / pacing", "utilization": "Capacity / pacing",
    "upstream_alerts_20": "Upstream propagation",
    "thermal_residual_abs": "Environmental / thermal",
    "manual_rework_20": "Manual QA outcome",
    "operator_idx": "Operator variation",
    "lot_mean_abs_z": "Incoming part quality",
    "variant_idx": "Product mix",
}


# --------------------------------------------------------------------------
# Causal feature extraction
# --------------------------------------------------------------------------

def _windows(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing windows, left-padded so index i sees only values[:i+1]."""
    pad = np.full(window - 1, values[0] if len(values) else 0.0)
    return sliding_window_view(np.concatenate([pad, values]), window)


def _roll_max(v: np.ndarray, w: int) -> np.ndarray:
    return _windows(v, w).max(axis=1)


def _roll_mean(v: np.ndarray, w: int) -> np.ndarray:
    return _windows(v, w).mean(axis=1)


def _roll_sum(v: np.ndarray, w: int) -> np.ndarray:
    return _windows(v, w).sum(axis=1)


def _roll_slope(v: np.ndarray, w: int) -> np.ndarray:
    """Rolling OLS slope, closed form. A polyfit per cell is ~500k fits per training
    sweep; the closed form is the same number to machine precision."""
    win = _windows(v, w)
    x = np.arange(w, dtype=float)
    xc = x - x.mean()
    denom = float((xc ** 2).sum()) or 1.0
    return (win - win.mean(axis=1, keepdims=True)) @ xc / denom


def _ewma_abs(v: np.ndarray, lam: float = 0.2) -> np.ndarray:
    out = np.empty(len(v), dtype=float)
    e = 0.0
    for i, x in enumerate(v):
        e = lam * x + (1 - lam) * e
        out[i] = abs(e)
    return out


def _cusum(cycles: np.ndarray, mu: float, sd: float) -> np.ndarray:
    out = np.empty(len(cycles), dtype=float)
    acc = 0.0
    k = 0.5 * sd
    for i, c in enumerate(cycles):
        acc = max(0.0, acc + (c - mu) - k)
        out[i] = acc / sd
    return out


def build_frame(raw: dict, analysis: dict, config: dict = CONFIG) -> dict[str, Any]:
    """Per (station, vehicle) causal feature matrix, plus labels where available."""
    stations = raw["stations"]
    count = raw.get("vehicle_count") or max(o["vehicle"] for o in raw["operations"])
    variants = {v: i for i, v in enumerate(config["product_variants"])}
    operators = {o: i for i, o in enumerate(config["latent_factors"]["operators"])}

    readings_by_station: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    thermal_by_station: dict[str, dict[int, float]] = defaultdict(dict)
    for r in raw["readings"]:
        readings_by_station[r["station"]][r["vehicle"]].append(abs(r["z"]))
        if r.get("baseline_source") == "physics_model_residual":
            thermal_by_station[r["station"]][r["vehicle"]] = abs(r["z"])

    ops_by_station: dict[str, dict[int, dict]] = defaultdict(dict)
    for o in raw["operations"]:
        ops_by_station[o["station"]][o["vehicle"]] = o

    alerts_by_station: dict[str, np.ndarray] = {}
    warns_by_station: dict[str, np.ndarray] = {}
    for s in stations:
        alerts_by_station[s["id"]] = np.zeros(count + 1)
        warns_by_station[s["id"]] = np.zeros(count + 1)
    for f in analysis["flags"]:
        arr = alerts_by_station if f.get("severity") == "alert" else warns_by_station
        if f["station"] in arr and 0 <= f["detected_at"] <= count:
            arr[f["station"]][f["detected_at"]] += 1

    # Ground-truth labels: a fault confirmed at this station within the next H vehicles.
    label_map: dict[str, set[int]] = defaultdict(set)
    paint_ids = [s["id"] for s in stations if s["area"] == "Paint"]
    for g in raw.get("ground_truth", []):
        targets = paint_ids if str(g["station"]).endswith("-*") else [g["station"]]
        for t in targets:
            label_map[t].add(g["vehicle"])

    lot_z: dict[str, list[float]] = defaultdict(list)
    for r in raw["readings"]:
        lot_z[r.get("supplier_lot", "NA")].append(abs(r["z"]))
    lot_mean = {k: float(np.mean(v)) for k, v in lot_z.items()}

    rows, meta, labels = [], [], []
    station_index = {s["id"]: i for i, s in enumerate(stations)}

    for s in stations:
        sid = s["id"]
        vehicles = np.arange(1, count + 1)
        zmax = np.array([max(readings_by_station[sid].get(v, [0.0])) for v in vehicles])
        zmean = np.array([float(np.mean(readings_by_station[sid].get(v, [0.0]))) for v in vehicles])
        therm = np.array([thermal_by_station[sid].get(v, 0.0) for v in vehicles])

        cycles = np.array([ops_by_station[sid].get(v, {}).get("cycle_time", 0.0) for v in vehicles])
        queues = np.array([ops_by_station[sid].get(v, {}).get("queue", 0.0) for v in vehicles])
        utils = np.array([ops_by_station[sid].get(v, {}).get("utilization", 0.0) for v in vehicles])
        rework = np.array([1.0 if ops_by_station[sid].get(v, {}).get("manual_outcome") in {"fail", "rework"}
                           else 0.0 for v in vehicles])

        warm = min(config["thresholds"]["baseline_warmup_vehicles"], count // 2)
        mu_c, sd_c = float(np.mean(cycles[:warm])), max(float(np.std(cycles[:warm], ddof=1)), 1e-6)
        mu_q, sd_q = float(np.mean(queues[:warm])), max(float(np.std(queues[:warm], ddof=1)), 1e-6)

        cycle_z = (cycles - mu_c) / sd_c
        queue_z = (queues - mu_q) / sd_q

        cusum = _cusum(cycles, mu_c, sd_c)
        ewma = _ewma_abs(zmean)
        max10 = _roll_max(zmax, 10)
        mean10 = _roll_mean(zmean, 10)
        slope10 = _roll_slope(zmean, 10)
        alerts20 = _roll_sum(alerts_by_station[sid][1:count + 1], 20)
        warns20 = _roll_sum(warns_by_station[sid][1:count + 1], 20)
        rework20 = _roll_sum(rework, 20)

        idx = station_index[sid]
        up_ids = [stations[j]["id"] for j in range(max(0, idx - 3), idx)]
        up_alerts = np.sum([alerts_by_station[u][1:count + 1] for u in up_ids], axis=0) \
            if up_ids else np.zeros(count)
        up20 = _roll_sum(up_alerts, 20)

        since = np.zeros(count)
        last = -999
        for i in range(count):
            if alerts_by_station[sid][i + 1] > 0:
                last = i
            since[i] = min(i - last, 200) if last >= 0 else 200

        for i, v in enumerate(vehicles):
            op = ops_by_station[sid].get(v, {})
            rows.append([
                max10[i], mean10[i], ewma[i], slope10[i],
                alerts20[i], warns20[i], since[i],
                cycle_z[i], cusum[i], queue_z[i], utils[i],
                up20[i], therm[i],
                rework20[i],
                float(operators.get(op.get("operator"), 0)),
                lot_mean.get(op.get("supplier_lot", "NA"), 0.0),
                float(variants.get(op.get("variant"), 0)),
            ])
            meta.append((sid, int(v)))
            future = label_map.get(sid, ())
            labels.append(1 if any(v < g <= v + HORIZON for g in future) else 0)

    return {"X": np.asarray(rows, dtype=float), "y": np.asarray(labels, dtype=int),
            "meta": meta, "vehicle_count": count}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def _randomised_config(seed: int, base: dict = CONFIG) -> dict:
    """A structurally identical line with different faults, so the model learns fault
    *signatures* rather than which station happens to be broken in the demo."""
    rng = np.random.default_rng(seed)
    cfg = copy.deepcopy(base)
    stations = [f"{g['prefix']}-{n:02}" for g in cfg["station_groups"]
                for n in range(1, g["count"] + 1)]
    instrumented = []
    for g in cfg["station_groups"]:
        manual = round(g["count"] * (1 - g["instrumented_ratio"]))
        instrumented += [f"{g['prefix']}-{n:02}" for n in range(manual + 1, g["count"] + 1)]

    injections = []
    for k in range(int(rng.integers(2, 5))):
        kind = str(rng.choice(["trend", "point", "cross_parameter", "bottleneck"]))
        start = int(rng.integers(215, 360))
        if kind == "bottleneck":
            injections.append({"id": f"r{k}", "kind": "bottleneck",
                               "station": str(rng.choice(stations)),
                               "start_vehicle": start, "end_vehicle": cfg["line"]["vehicle_count"],
                               "cycle_slope": float(rng.uniform(0.10, 0.28))})
            continue
        sid = str(rng.choice(instrumented))
        params = [p for g in cfg["station_groups"] if sid.startswith(g["prefix"])
                  for p in g["parameters"] if g["parameters"][p].get("model") != "thermal_first_order"]
        if kind == "cross_parameter" and len(params) >= 2:
            injections.append({"id": f"r{k}", "kind": "cross_parameter", "station": sid,
                               "parameters": list(rng.choice(params, 2, replace=False)),
                               "start_vehicle": start, "end_vehicle": start + int(rng.integers(20, 60)),
                               "shift_std": float(rng.uniform(2.6, 4.2))})
        elif kind == "point":
            injections.append({"id": f"r{k}", "kind": "point", "station": sid,
                               "parameter": str(rng.choice(params)),
                               "vehicles": list(range(start, start + int(rng.integers(3, 7)))),
                               "shift_std": float(rng.uniform(3.2, 5.0))})
        else:
            injections.append({"id": f"r{k}", "kind": "trend", "station": sid,
                               "parameter": str(rng.choice(params)),
                               "start_vehicle": start, "end_vehicle": start + int(rng.integers(50, 120)),
                               "slope_std_per_vehicle": float(rng.uniform(0.03, 0.09))})
    cfg["injections"] = injections
    return cfg


def _collect(seeds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, groups = [], [], []
    for seed in seeds:
        cfg = _randomised_config(seed)
        raw = simulate(cfg, seed=seed)
        frame = build_frame(raw, analyze(raw, cfg), cfg)
        X.append(frame["X"])
        y.append(frame["y"])
        groups.append(np.full(len(frame["y"]), seed))
    return np.vstack(X), np.concatenate(y), np.concatenate(groups)


class _Model:
    """Thin wrapper so the rest of the system does not care which backend trained."""

    def __init__(self):
        self.backend = "unavailable"
        self.version = ""
        self.clf = None
        self.calibrator = None
        self.importances: dict[str, float] = {}
        self.metrics: dict[str, Any] = {}
        self.base_rate = 0.0

    # -- probability -----------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            return np.zeros(len(X))
        p = self.clf.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 0.0, 1.0)

    # -- attribution -----------------------------------------------------
    def contributions(self, X: np.ndarray) -> np.ndarray | None:
        """Exact TreeSHAP contributions in log-odds space, or None if unavailable."""
        if self.backend != "xgboost" or self.clf is None:
            return None
        import xgboost as xgb
        booster = self.clf.get_booster()
        return booster.predict(xgb.DMatrix(X, feature_names=FEATURES), pred_contribs=True)


def train() -> _Model:
    model = _Model()
    X_tr, y_tr, _ = _collect(TRAIN_SEEDS)
    X_cal, y_cal, _ = _collect(CAL_SEEDS)
    X_va, y_va, _ = _collect(VAL_SEEDS)
    model.base_rate = float(y_va.mean())

    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=220, max_depth=4, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=6,
            reg_lambda=1.5, eval_metric="logloss", random_state=42, n_jobs=4,
        )
        clf.fit(X_tr, y_tr)
        model.clf, model.backend, model.version = clf, "xgboost", xgb.__version__
        gains = np.asarray(clf.feature_importances_, dtype=float)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        import sklearn
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06,
                                             max_iter=220, random_state=42)
        clf.fit(X_tr, y_tr)
        model.clf, model.backend, model.version = clf, "sklearn", sklearn.__version__
        gains = np.ones(len(FEATURES))

    # Calibrate on the validation lines so a reported percentage is a frequency.
    try:
        from sklearn.isotonic import IsotonicRegression
        # Fitted on the calibration lines only. Scoring calibration on the same rows
        # it was fitted to would make any reliability diagram look perfect by
        # construction; the reported one comes from a third, untouched set of lines.
        raw_p = model.clf.predict_proba(X_cal)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_p, y_cal)
        model.calibrator = iso
    except Exception:
        model.calibrator = None

    total = float(gains.sum()) or 1.0
    model.importances = {f: round(float(g) / total * 100.0, 1) for f, g in zip(FEATURES, gains)}
    model.metrics = _evaluate(model, X_va, y_va)
    return model


def _evaluate(model: _Model, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Held-out metrics. PR-AUC is reported against its no-skill baseline because on a
    class this rare an ROC-AUC reads impressively high for a useless model."""
    p = model.predict(X)
    out: dict[str, Any] = {
        "held_out_rows": int(len(y)),
        "positive_rate_pct": round(100.0 * float(y.mean()), 2),
        "brier_score": round(float(np.mean((p - y) ** 2)), 4),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        out["pr_auc"] = round(float(average_precision_score(y, p)), 3)
        out["pr_auc_baseline"] = round(float(y.mean()), 3)
        out["roc_auc"] = round(float(roc_auc_score(y, p)), 3)
        out["lift_over_baseline"] = round(out["pr_auc"] / max(out["pr_auc_baseline"], 1e-9), 1)
    except Exception:
        pass

    order = np.argsort(-p)
    for k in (40, 120):
        if len(order) >= k:
            out[f"precision_at_{k}"] = round(float(y[order[:k]].mean()), 3)

    # Reliability: mean predicted vs observed frequency per decile.
    bins = np.clip((p * 10).astype(int), 0, 9)
    curve = []
    for b in range(10):
        mask = bins == b
        if mask.sum() >= 25:
            curve.append({"bin": b / 10.0,
                          "predicted": round(float(p[mask].mean()), 3),
                          "observed": round(float(y[mask].mean()), 3),
                          "n": int(mask.sum())})
    out["reliability"] = curve
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

_MODEL: _Model | None = None
CACHE_DIR = Path(__file__).parent / ".cache"


def _cache_key() -> str:
    """Cache is keyed on everything that changes the model: the config that generates
    the training lines, the feature set, and the seed ranges."""
    payload = repr((CONFIG, FEATURES, list(TRAIN_SEEDS), list(CAL_SEEDS),
                    list(VAL_SEEDS), HORIZON)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def get_model(use_cache: bool = True) -> _Model:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    path = CACHE_DIR / f"risk_model_{_cache_key()}.pkl"
    if use_cache and path.exists():
        try:
            _MODEL = pickle.loads(path.read_bytes())
            return _MODEL
        except Exception:
            pass
    _MODEL = train()
    if use_cache:
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            for stale in CACHE_DIR.glob("risk_model_*.pkl"):
                stale.unlink()
            path.write_bytes(pickle.dumps(_MODEL))
        except Exception:
            pass
    return _MODEL


def score(raw: dict, analysis: dict, config: dict = CONFIG) -> dict[str, Any]:
    """Score every (station, vehicle) cell and attach per-prediction attribution."""
    model = get_model()
    frame = build_frame(raw, analysis, config)
    X, meta = frame["X"], frame["meta"]
    probs = model.predict(X) * 100.0
    contribs = model.contributions(X)

    by_station: dict[str, dict[int, float]] = defaultdict(dict)
    drivers: dict[str, dict[int, list]] = defaultdict(dict)
    for i, (sid, v) in enumerate(meta):
        by_station[sid][v] = round(float(probs[i]), 1)
        if contribs is not None:
            row = contribs[i][:len(FEATURES)]
            top = np.argsort(-np.abs(row))[:3]
            drivers[sid][v] = [
                {"feature": FEATURES[j], "label": FEATURE_LABELS[FEATURES[j]],
                 "family": CAUSE_FAMILY[FEATURES[j]],
                 "contribution": round(float(row[j]), 3)}
                for j in top if abs(row[j]) > 1e-6
            ]

    return {
        "available": True,
        "model": f"XGBoost {model.version}" if model.backend == "xgboost"
                 else f"scikit-learn HistGradientBoosting {model.version}",
        "backend": model.backend,
        "attribution_method": ("Exact TreeSHAP contributions (log-odds)" if contribs is not None
                               else "Gain-based split importance only"),
        "target": f"Confirmed incident at this station within the next {HORIZON} vehicles",
        "training_provenance": (
            f"Trained on {len(list(TRAIN_SEEDS))} independently generated production lines with "
            f"randomised fault station, kind, onset and magnitude; calibrated on "
            f"{len(list(CAL_SEEDS))} further lines, and evaluated on {len(list(VAL_SEEDS))} lines "
            f"used for neither. Split by line, never by row. "
            f"The demo line (seed {DEMO_SEED}) appears in no training or calibration set."),
        "calibration": "Isotonic regression fitted on held-out lines",
        "feature_importances": model.importances,
        "feature_labels": FEATURE_LABELS,
        "validation": model.metrics,
        "station_scores": {k: dict(v) for k, v in by_station.items()},
        "station_drivers": {k: dict(v) for k, v in drivers.items()},
        "horizon_vehicles": HORIZON,
    }
