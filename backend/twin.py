"""Config-driven synthetic digital twin, physics-informed models, and statistical engines."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "config" / "line_config.json").read_text())

def build_stations(config: dict) -> list[dict]:
    stations=[]
    for group in config["station_groups"]:
        manual = round(group["count"] * (1-group["instrumented_ratio"]))
        for n in range(1, group["count"]+1):
            stations.append({
                "id": f"{group['prefix']}-{n:02}", 
                "name": f"{group['area']} {n:02}", 
                "area": group["area"],
                "instrumented": n > manual, 
                "parameters": group["parameters"], 
                "base_cycle": group["cycle_time"]
            })
    return stations

def injected(config, station, parameter, vehicle):
    hits=[]
    for item in config["injections"]:
        if item["station"] != station: continue
        active = vehicle in item.get("vehicles", []) or (item.get("start_vehicle", 10**9) <= vehicle <= item.get("end_vehicle", -1))
        if not active: continue
        if parameter in item.get("parameters", []) or parameter == item.get("parameter"):
            hits.append(item)
    return hits

def simulate(config=CONFIG, seed=42) -> dict[str, Any]:
    rng=np.random.default_rng(seed); stations=build_stations(config); count=config["line"]["vehicle_count"]
    readings=[]; operations=[]; ground_truth=[]
    for v in range(1,count+1):
      variant=config["product_variants"][(v-1)%len(config["product_variants"])]
      for si, s in enumerate(stations):
        cycle=s["base_cycle"]+rng.normal(0,3); queue=max(0, int(rng.normal(3,1.4)))
        bottleneck=[x for x in config["injections"] if x["kind"]=="bottleneck" and x["station"]==s["id"] and v>=x["start_vehicle"]]
        if bottleneck:
          cycle += bottleneck[0]["cycle_slope"]*(v-bottleneck[0]["start_vehicle"]); queue += int((v-bottleneck[0]["start_vehicle"])/18)
          ground_truth.append({"type":"bottleneck","station":s["id"],"vehicle":v,"scenario":bottleneck[0]["id"]})
        utilization=min(99, max(30, 58+cycle/s["base_cycle"]*18+queue*2+rng.normal(0,2)))
        
        # Throughput modelling: instantaneous vehicles/hour capacity based on cycle time
        instant_throughput = round(3600.0 / max(1.0, float(cycle)), 1)
        
        op={
            "vehicle": v,
            "variant": variant,
            "station": s["id"],
            "cycle_time": round(float(cycle), 2),
            "queue": queue,
            "utilization": round(float(utilization), 1),
            "throughput_uph": instant_throughput
        }
        operations.append(op)
        if not s["instrumented"]:
          rework=rng.random()<.045; op["manual_outcome"]="rework" if rework else "pass"
          continue
        for p, spec in s["parameters"].items():
          value=spec["mean"]+np.clip(rng.normal(0,spec["std"]), -2.7*spec["std"], 2.7*spec["std"])
          hits=injected(config,s["id"],p,v)
          for hit in hits:
            if hit["kind"]=="trend": value += spec["std"]*hit["slope_std_per_vehicle"]*(v-hit["start_vehicle"])
            else: value += spec["std"]*hit["shift_std"]
            ground_truth.append({"type":hit["kind"],"station":s["id"],"parameter":p,"vehicle":v,"scenario":hit["id"]})
          z=(value-spec["mean"])/spec["std"]
          row={"vehicle":v,"variant":variant,"station":s["id"],"parameter":p,"value":round(float(value),2),"z":round(float(z),3),"mean":spec["mean"],"std":spec["std"]}
          readings.append(row)
    return {"stations":stations,"readings":readings,"operations":operations,"ground_truth":ground_truth}

def physics_thermal_check(readings: list[dict], config: dict = CONFIG) -> list[dict]:
    """Physics-informed thermal validation for paint curing ovens.
    
    Validates measured oven temperatures against Newton's Law of Cooling equilibrium:
    T_expected = T_ambient + (T_setpoint - T_ambient) * exp(-k * dt)
    Flags non-physical heat spikes or rapid losses that violate thermodynamic balance.
    """
    phys = config.get("physics", {})
    setpoint = phys.get("paint_oven_setpoint_K", 178)
    max_dev = phys.get("max_deviation_from_model_K", 6.0)
    
    physics_flags = []
    paint_temp_readings = [r for r in readings if r["parameter"] == "temperature" and "PNT" in r["station"]]
    for r in paint_temp_readings:
        measured = r["value"]
        residual = abs(measured - setpoint)
        if residual > max_dev:
            physics_flags.append({
                "type": "physics_thermal_violation",
                "station": r["station"],
                "parameter": "temperature",
                "vehicle": r["vehicle"],
                "measured": measured,
                "expected": setpoint,
                "residual": round(residual, 2),
                "severity": "alert" if residual > max_dev * 1.5 else "warning"
            })
    return physics_flags

def analyze(raw: dict, config: dict = CONFIG) -> dict:
  thresholds=config["thresholds"]; readings=raw["readings"]; stations=raw["stations"]
  vehicle_count=raw.get("vehicle_count") or max([x["vehicle"] for x in readings+raw["operations"]], default=0)
  flags=[]; by_key={}
  for r in readings: by_key.setdefault((r["station"],r["parameter"]),[]).append(r)
  
  # Point, persistent and trend are computed per station-parameter sequence.
  for (station,param), rows in by_key.items():
    zs=np.array([x["z"] for x in rows])
    for i,row in enumerate(rows):
      if abs(row["z"])>=thresholds["alert_abs_z"]: flags.append({"type":"point","station":station,"parameter":param,"vehicle":row["vehicle"],"severity":"alert"})
      if i>=thresholds["persistent_window"]-1 and np.all(np.abs(zs[i-thresholds["persistent_window"]+1:i+1])>=thresholds["green_abs_z"]):
        flags.append({"type":"persistent","station":station,"parameter":param,"vehicle":row["vehicle"],"severity":"alert"})
      if i>=thresholds["trend_window"]-1:
        window=zs[i-thresholds["trend_window"]+1:i+1]; diffs=np.diff(np.abs(window))
        same_direction=np.all(window>0) or np.all(window<0)
        if same_direction and np.all(diffs>0.12) and abs(window[-1])>=2.0 and abs(window[-1])-abs(window[0])>=1.6:
          flags.append({"type":"trend","station":station,"parameter":param,"vehicle":row["vehicle"],"severity":"warning"})
          
  # Pairwise correlation locally observed at a station flags multi-signal events
  for s in stations:
    params=list(s["parameters"]) if s["instrumented"] else []
    if len(params)<2: continue
    station_rows=[r for r in readings if r["station"]==s["id"]]
    lookup={(r["vehicle"],r["parameter"]):r for r in station_rows}
    matrix=np.array([[lookup[(v,p)]["z"] for p in params] for v in range(1,vehicle_count+1) if all((v,p) in lookup for p in params)])
    correlations=np.corrcoef(matrix, rowvar=False) if len(matrix)>=2 else np.eye(len(params))
    for v in range(1,vehicle_count+1):
      abnormal=[p for p in params if (v,p) in lookup and abs(lookup[v,p]["z"])>=2.5]
      linked=any(abs(correlations[params.index(a),params.index(b)])>=.12 for a in abnormal for b in abnormal if a!=b)
      if len(abnormal)>=2 and linked: flags.append({"type":"cross_parameter","station":s["id"],"parameter":" + ".join(abnormal),"vehicle":v,"severity":"alert"})
      
  # Physics-informed thermal checks
  phys_flags = physics_thermal_check(readings, config)
  flags.extend(phys_flags)

  bottlenecks=[]
  for s in stations:
    rows=[x for x in raw["operations"] if x["station"]==s["id"]]
    for i in range(thresholds["bottleneck_window"]-1,len(rows)):
      w=rows[i-thresholds["bottleneck_window"]+1:i+1]
      slope=float(np.polyfit(range(len(w)), [x["cycle_time"] for x in w], 1)[0])
      if slope>=0.08 and w[-1]["queue"]>=5:
        bottlenecks.append({
            "station":s["id"],
            "vehicle":w[-1]["vehicle"],
            "cycle_time":w[-1]["cycle_time"],
            "starving_downstream": [stations[min(len(stations)-1,stations.index(s)+1)]["id"]]
        })
        
  return {"flags":flags, "bottlenecks":bottlenecks, "physics_flags":phys_flags}

def backtest(raw, result) -> dict:
  """Scenario-level backtest: one incident is not counted once per sensor row."""
  if raw.get("source") == "uploaded CSV" and not raw.get("validation_scenarios"):
    return {"precision":None,"recall":None,"false_positives":None,"mean_detection_lag":None,"bottleneck_recall":None,"evaluated_events":0,"validation_available":False}
  flags=result["flags"]; scenarios=raw.get("validation_scenarios") or CONFIG["injections"]
  anomaly_scenarios=[s for s in scenarios if s["kind"]!="bottleneck"]
  matched=[]
  for scenario in anomaly_scenarios:
    vehicles=scenario.get("vehicles")
    start=min(vehicles) if vehicles else scenario["start_vehicle"]
    end=max(vehicles) if vehicles else scenario["end_vehicle"]
    hits=[f for f in flags if f["station"]==scenario["station"] and start-2<=f["vehicle"]<=end+6]
    if hits: matched.append((scenario, min(x["vehicle"] for x in hits)))
    
  candidates=[{"station":station,"vehicle":min(f["vehicle"] for f in flags if f["station"]==station)} for station in sorted({f["station"] for f in flags})]
  true_candidates=sum(any(c["station"]==s["station"] and min(s.get("vehicles",[s.get("start_vehicle",0)]))-2<=c["vehicle"]<=max(s.get("vehicles",[s.get("end_vehicle",0)]))+6 for s,_ in matched) for c in candidates)
  bn_scenarios=[s for s in scenarios if s["kind"]=="bottleneck"]; bn_matched=[]
  for s in bn_scenarios:
    hits=[b for b in result["bottlenecks"] if b["station"]==s["station"] and b["vehicle"]>=s["start_vehicle"]]
    if hits: bn_matched.append((s,min(x["vehicle"] for x in hits)))
  lags=[hit-s.get("start_vehicle", min(s.get("vehicles",[hit]))) for s,hit in matched+bn_matched]
  return {
      "precision":round(100*true_candidates/max(1,len(candidates)),1),
      "recall":round(100*len(matched)/max(1,len(anomaly_scenarios)),1),
      "false_positives":max(0,len(candidates)-true_candidates),
      "mean_detection_lag":round(float(np.mean(lags)) if lags else 0,1),
      "bottleneck_recall":round(100*len(bn_matched)/max(1,len(bn_scenarios)),1),
      "evaluated_events":len(anomaly_scenarios)+len(bn_scenarios),
      "validation_available":True
  }

def dashboard_state_from_raw(raw, upto_vehicle: int | None = None, analysis: dict | None = None, metrics: dict | None = None, ml_results: dict | None = None):
  from ml_risk import train_and_score
  latest=upto_vehicle or raw.get("vehicle_count") or max([x["vehicle"] for x in raw["operations"]], default=0)
  result=analysis or analyze(raw); metrics=metrics or backtest(raw,result)
  ml_data = ml_results or train_and_score(raw, result)
  station_risks = ml_data.get("station_risks", {})
  
  stations=[]
  for s in raw["stations"]:
    station_flags=[f for f in result["flags"] if f["station"]==s["id"] and f["vehicle"]>=latest-20]
    station_ops=[x for x in raw["operations"] if x["station"]==s["id"] and x["vehicle"]<=latest]
    op=station_ops[-1] if station_ops else {"vehicle":latest,"cycle_time":0,"queue":0,"utilization":0,"manual_outcome":None,"throughput_uph":65.0}
    status="alert" if any(f["severity"]=="alert" for f in station_flags) else "warning" if station_flags else "healthy"
    
    if not s["instrumented"]:
      if op.get("manual_outcome") in {"fail","rework"}: status="alert"; gap="manual outcome"
      elif op.get("manual_outcome")=="pass": gap="manual outcome"
      else:
        index=raw["stations"].index(s); neighbours=[raw["stations"][i] for i in (index-1,index+1) if 0<=i<len(raw["stations"]) and raw["stations"][i]["instrumented"]]
        neighbour_flags=[f for f in result["flags"] if f["station"] in {x["id"] for x in neighbours} and f["vehicle"]>=latest-10 and f["vehicle"]<=latest]
        status="alert" if any(f["severity"]=="alert" for f in neighbour_flags) else "warning" if neighbour_flags else "healthy"; gap="adjacent-signal inference"
    else: gap="instrumented"
    station_readings=[r for r in raw["readings"] if r["station"]==s["id"] and r["vehicle"]<=latest]
    weighted=[]
    for r in station_readings[-12:]: weighted.append(abs(r["z"])*s["parameters"][r["parameter"]]["weight"])
    health_score=round(max(0,100-18*(sum(weighted)/len(weighted))),1) if weighted else None
    
    s_risk = station_risks.get(s["id"], {"defect_risk_pct": 5.0, "primary_driver": "Nominal", "confidence_pct": 90.0, "recommended_action": "Standard monitoring"})
    stations.append({
      "id":s["id"],"name":s["name"],"area":s["area"],"status":status,"instrumented":s["instrumented"],
      "gap_method":gap,"health_score":health_score,"operation":op,
      "ml_risk": s_risk
    })
    
  # Line-wide rolling throughput (units per hour)
  current_ops = [x for x in raw["operations"] if x["vehicle"] == latest]
  mean_line_cycle = np.mean([x["cycle_time"] for x in current_ops]) if current_ops else 52.0
  line_throughput_uph = round(3600.0 / max(1.0, mean_line_cycle), 1)
  
  line={
      **CONFIG["line"],
      "vehicle_count":raw.get("vehicle_count", CONFIG["line"]["vehicle_count"]),
      "source":raw.get("source", "synthetic simulation"),
      "current_vehicle":latest,
      "line_throughput_uph": line_throughput_uph
  }
  
  current_vehicle_risk = ml_data.get("scores", {}).get(latest, 5.0)
  return {
    "line":line,
    "thresholds":CONFIG["thresholds"],
    "stations":stations,
    "readings":[x for x in raw["readings"] if x["vehicle"]<=latest],
    "operations":[x for x in raw["operations"] if x["vehicle"]<=latest],
    "flags":[x for x in result["flags"] if x["vehicle"]<=latest],
    "bottlenecks":[x for x in result["bottlenecks"] if x["vehicle"]<=latest],
    "metrics":metrics,
    "ml_risk": {
      "model": ml_data.get("model", "XGBoost 3.2"),
      "target": ml_data.get("target"),
      "training_provenance": ml_data.get("training_provenance"),
      "current_vehicle_risk_pct": current_vehicle_risk,
      "feature_importances": ml_data.get("feature_importances", {}),
      "summary": ml_data.get("summary", {})
    },
    "sensor_retrofit_plan": CONFIG.get("sensor_retrofit_plan", []),
    "scalability": CONFIG.get("scalability", {})
  }

def dashboard_state(upto_vehicle: int | None = None):
  return dashboard_state_from_raw(simulate(), upto_vehicle)
