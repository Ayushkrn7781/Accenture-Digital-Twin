from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from twin import CONFIG, analyze, backtest, dashboard_state_from_raw, simulate
from ingestion import DatasetError, load_ground_truth_csv, load_historical_csv
from ml_risk import train_and_score
from uuid import uuid4

app=FastAPI(title="DigitalTwin.ai API")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
SYNTHETIC_RAW=simulate()
SYNTHETIC_ANALYSIS=analyze(SYNTHETIC_RAW)
SYNTHETIC_METRICS=backtest(SYNTHETIC_RAW, SYNTHETIC_ANALYSIS)
SYNTHETIC_ML=train_and_score(SYNTHETIC_RAW, SYNTHETIC_ANALYSIS)
STATE=dashboard_state_from_raw(SYNTHETIC_RAW, analysis=SYNTHETIC_ANALYSIS, metrics=SYNTHETIC_METRICS, ml_results=SYNTHETIC_ML)
REPLAYS={}

@app.get("/api/dashboard")
def dashboard(upto_vehicle: int | None = None):
  return dashboard_state_from_raw(SYNTHETIC_RAW, upto_vehicle, SYNTHETIC_ANALYSIS, SYNTHETIC_METRICS, SYNTHETIC_ML)

@app.get("/api/stations/{station_id}")
def station(station_id:str):
  return {
    "station":next(x for x in STATE["stations"] if x["id"]==station_id),
    "readings":[x for x in STATE["readings"] if x["station"]==station_id],
    "flags":[x for x in STATE["flags"] if x["station"]==station_id]
  }

@app.get("/api/ml/insights")
def ml_insights():
  return SYNTHETIC_ML

@app.post("/api/datasets/upload")
async def upload_dataset(data_file: UploadFile = File(...), ground_truth_file: UploadFile | None = File(None)):
  if not data_file.filename.lower().endswith(".csv"): raise HTTPException(400, "Data file must be a CSV.")
  try:
    raw=load_historical_csv(await data_file.read(), CONFIG)
    if ground_truth_file:
      if not ground_truth_file.filename.lower().endswith(".csv"): raise HTTPException(400, "Ground truth file must be a CSV.")
      raw["validation_scenarios"]=load_ground_truth_csv(await ground_truth_file.read(), raw)
  except DatasetError as exc: raise HTTPException(422, str(exc)) from exc
  replay_id=str(uuid4()); analysis=analyze(raw); ml_res=train_and_score(raw, analysis)
  REPLAYS[replay_id]={"raw":raw,"analysis":analysis,"metrics":backtest(raw, analysis),"ml":ml_res}
  return {"replay_id":replay_id,"vehicle_count":raw["vehicle_count"],"source":raw["source"],"ground_truth_loaded":bool(raw.get("validation_scenarios"))}

@app.get("/api/replays/{replay_id}/state")
def replay_state(replay_id: str, upto_vehicle: int = 1):
  replay=REPLAYS.get(replay_id)
  if not replay: raise HTTPException(404, "Replay session not found. Upload the dataset again.")
  raw=replay["raw"]
  return dashboard_state_from_raw(raw, max(1, min(upto_vehicle, raw["vehicle_count"])), replay["analysis"], replay["metrics"], replay.get("ml"))
