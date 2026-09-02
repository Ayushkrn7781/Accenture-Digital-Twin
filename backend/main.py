"""DigitalTwin.ai API.

Endpoints are split by what each panel actually needs. The previous single endpoint
shipped every reading and operation on every poll - about 7 MB at the end of a run,
refetched several times a second - so that a chart could draw forty points. Here the
summary state is a few tens of KB and series are fetched only for the station in view.

Heavy work happens once during startup rather than on the import path, so the process
can answer a health check while it warms instead of blocking every request behind it.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import economics as econ
from ingestion import DatasetError, load_ground_truth_csv, load_historical_csv
from metrics import backtest, false_alarm_profile, sensor_gap_study
from ml_risk import DEMO_SEED, score
from state import build_state
from twin import CONFIG, ArtifactIndex, analyze, simulate

REPLAY_LIMIT = 4
REPLAY_TTL_SECONDS = 30 * 60

WARM: dict = {"ready": False, "stage": "starting"}
REPLAYS: "OrderedDict[str, dict]" = OrderedDict()


def _bundle(raw: dict, config=CONFIG) -> dict:
    analysis = analyze(raw, config)
    metrics = backtest(raw, analysis, config)
    try:
        ml = score(raw, analysis, config)
    except Exception:
        ml = None
    return {
        "raw": raw, "analysis": analysis, "metrics": metrics, "ml": ml, "config": config,
        "indices": {"flags": ArtifactIndex(analysis["flags"]),
                    "bottlenecks": ArtifactIndex(analysis["bottlenecks"])},
    }


def _warm_up() -> None:
    WARM["stage"] = "simulating line"
    raw = simulate(CONFIG, seed=DEMO_SEED)
    WARM["stage"] = "running detectors"
    bundle = _bundle(raw)
    WARM["stage"] = "measuring false-alarm floor"
    fa = false_alarm_profile(CONFIG, seeds=8)
    WARM["stage"] = "sensor coverage study"
    gap = sensor_gap_study(CONFIG)
    WARM["stage"] = "business case"
    state = build_state(raw, bundle["analysis"], bundle["metrics"], bundle["ml"],
                        indices=bundle["indices"])
    ec = econ.compute(bundle["metrics"], fa, state["line"])
    WARM.update({
        "ready": True, "stage": "ready", "bundle": bundle,
        "false_alarms": fa, "gap_study": gap, "economics": ec,
        "rollout": econ.rollout_case(ec, CONFIG.get("scalability", {})),
        "retrofit": econ.retrofit_case(CONFIG.get("sensor_retrofit_plan", []), gap, ec),
    })


@asynccontextmanager
async def lifespan(_: FastAPI):
    _warm_up()
    yield
    REPLAYS.clear()


app = FastAPI(title="DigitalTwin.ai API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _require_ready() -> dict:
    if not WARM.get("ready"):
        raise HTTPException(503, f"Twin is warming up: {WARM.get('stage')}.")
    return WARM


def _evict() -> None:
    now = time.time()
    for key in [k for k, v in REPLAYS.items() if now - v["created"] > REPLAY_TTL_SECONDS]:
        REPLAYS.pop(key, None)
    while len(REPLAYS) > REPLAY_LIMIT:
        REPLAYS.popitem(last=False)


def _resolve(replay_id: str | None) -> dict:
    if not replay_id:
        return _require_ready()["bundle"]
    replay = REPLAYS.get(replay_id)
    if not replay:
        raise HTTPException(404, "Replay session not found. Upload the dataset again.")
    return replay["bundle"]


# --------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"ready": WARM.get("ready", False), "stage": WARM.get("stage")}


@app.get("/api/state")
def state(at: int | None = None, replay_id: str | None = None):
    """Line summary and station cards for one point in the replay."""
    bundle = _resolve(replay_id)
    return build_state(bundle["raw"], bundle["analysis"], bundle["metrics"], bundle["ml"],
                       upto_vehicle=at, config=bundle["config"], indices=bundle["indices"])


@app.get("/api/stations/{station_id}/series")
def station_series(station_id: str, at: int | None = None,
                   n: int = Query(120, ge=10, le=600), replay_id: str | None = None):
    """Control-chart series for one station only."""
    bundle = _resolve(replay_id)
    raw = bundle["raw"]
    if not any(s["id"] == station_id for s in raw["stations"]):
        raise HTTPException(404, f"Unknown station '{station_id}'.")
    latest = at or raw.get("vehicle_count")
    rows = [r for r in raw["readings"] if r["station"] == station_id and r["vehicle"] <= latest]
    by_param: dict[str, list] = {}
    for r in rows:
        by_param.setdefault(r["parameter"], []).append(
            {"vehicle": r["vehicle"], "z": r["z"], "value": r["value"],
             "variant": r["variant"], "baseline_source": r.get("baseline_source")})
    for param in by_param:
        by_param[param] = by_param[param][-n:]
    flags = bundle["indices"]["flags"].window(station_id, 0, latest)
    return {"station": station_id, "series": by_param, "flags": flags,
            "thresholds": bundle["config"]["thresholds"]}


@app.get("/api/flags")
def flags(at: int | None = None, since: int | None = None, replay_id: str | None = None):
    bundle = _resolve(replay_id)
    latest = at or bundle["raw"].get("vehicle_count")
    lo = since if since is not None else 0
    return {"flags": [f for f in bundle["indices"]["flags"].upto(latest) if f["detected_at"] >= lo]}


@app.get("/api/bottlenecks")
def bottlenecks(at: int | None = None, replay_id: str | None = None):
    bundle = _resolve(replay_id)
    latest = at or bundle["raw"].get("vehicle_count")
    return {"bottlenecks": bundle["indices"]["bottlenecks"].upto(latest)}


@app.get("/api/validation")
def validation():
    """Everything behind the leadership view's claims, in one place."""
    warm = _require_ready()
    return {
        "backtest": warm["bundle"]["metrics"],
        "false_alarms": warm["false_alarms"],
        "sensor_gap": warm["gap_study"],
        "ml": (warm["bundle"]["ml"] or {}).get("validation"),
        "ml_provenance": {k: (warm["bundle"]["ml"] or {}).get(k) for k in
                          ("model", "backend", "attribution_method", "training_provenance",
                           "calibration", "target")},
        "calibration": CONFIG["bottleneck"].get("calibrated"),
        "propagation": warm["bundle"]["analysis"].get("propagation"),
    }


@app.get("/api/economics")
def economics():
    warm = _require_ready()
    return {"economics": warm["economics"], "rollout": warm["rollout"],
            "retrofit": warm["retrofit"], "scalability": CONFIG.get("scalability", {})}


@app.post("/api/datasets/upload")
async def upload_dataset(data_file: UploadFile = File(...),
                         ground_truth_file: UploadFile | None = File(None)):
    if not (data_file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Data file must be a CSV.")
    try:
        raw = load_historical_csv(await data_file.read(), CONFIG)
        if ground_truth_file:
            if not (ground_truth_file.filename or "").lower().endswith(".csv"):
                raise HTTPException(400, "Ground truth file must be a CSV.")
            raw["validation_scenarios"] = load_ground_truth_csv(await ground_truth_file.read(), raw)
    except DatasetError as exc:
        raise HTTPException(422, str(exc)) from exc

    replay_id = str(uuid4())
    REPLAYS[replay_id] = {"bundle": _bundle(raw), "created": time.time()}
    _evict()
    return {"replay_id": replay_id, "vehicle_count": raw["vehicle_count"],
            "source": raw["source"], "ground_truth_loaded": bool(raw.get("validation_scenarios"))}
