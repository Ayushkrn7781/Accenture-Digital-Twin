"""Read-only CSV ingestion for historical line-data replays.

The loader deliberately produces the exact same canonical structures as the
synthetic generator: readings, operations, stations, and optional validation
scenarios. No PLC/OT connection or write operation is present here.
"""
from __future__ import annotations
import csv
import io
from collections import defaultdict
from typing import Any
from twin import build_stations

DATA_COLUMNS = {"timestamp", "vehicle_id", "product_variant", "station_id", "parameter", "value", "cycle_time_seconds", "queue_length", "utilization_pct", "manual_outcome"}
GROUND_TRUTH_COLUMNS = {"vehicle_id", "station_id", "event_type", "event_start", "event_end", "confirmed_by"}

class DatasetError(ValueError):
    pass

def _rows(blob: bytes) -> list[dict[str, str]]:
    try:
        rows=list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))
    except UnicodeDecodeError as exc:
        raise DatasetError("CSV must be UTF-8 encoded.") from exc
    if not rows or not rows[0]: raise DatasetError("The CSV has no data rows.")
    return rows

def load_historical_csv(blob: bytes, config: dict) -> dict[str, Any]:
    rows=_rows(blob); columns=set(rows[0])
    required={"vehicle_id", "product_variant", "station_id"}
    missing=required-columns
    if missing: raise DatasetError(f"Missing required columns: {', '.join(sorted(missing))}")
    stations=build_stations(config); station_by_id={s["id"]:s for s in stations}
    # Preserve production ordering using timestamp where supplied, then file order.
    ordered=sorted(enumerate(rows), key=lambda x: (x[1].get("timestamp") or "", x[0]))
    vehicle_sequence={}
    for _, row in ordered:
        external=row["vehicle_id"].strip()
        if external and external not in vehicle_sequence: vehicle_sequence[external]=len(vehicle_sequence)+1
    readings=[]; operations_by_key={}
    for _, row in ordered:
        station_id=row["station_id"].strip(); external=row["vehicle_id"].strip()
        if station_id not in station_by_id: raise DatasetError(f"Unknown station_id '{station_id}'. Add it to the line configuration before replaying.")
        if not external: raise DatasetError("vehicle_id cannot be blank.")
        station=station_by_id[station_id]; vehicle=vehicle_sequence[external]; variant=row["product_variant"].strip() or "Unknown"
        key=(vehicle, station_id)
        if key not in operations_by_key:
            operations_by_key[key]={"vehicle":vehicle,"external_vehicle_id":external,"variant":variant,"station":station_id,
              "cycle_time":None,"queue":None,"utilization":None,"manual_outcome":None}
        operation=operations_by_key[key]
        try:
            if row.get("cycle_time_seconds"): operation["cycle_time"]=float(row["cycle_time_seconds"])
            if row.get("queue_length"): operation["queue"]=int(float(row["queue_length"]))
            if row.get("utilization_pct"): operation["utilization"]=float(row["utilization_pct"])
        except ValueError as exc: raise DatasetError(f"Invalid operation metric for vehicle '{external}', station '{station_id}'.") from exc
        outcome=(row.get("manual_outcome") or "").strip().lower()
        if outcome:
            if outcome not in {"pass", "fail", "rework"}: raise DatasetError("manual_outcome must be pass, fail, or rework.")
            operation["manual_outcome"]=outcome
        parameter=(row.get("parameter") or "").strip()
        value=(row.get("value") or "").strip()
        if not parameter and not value: continue
        if not station["instrumented"]: raise DatasetError(f"{station_id} is configured manual-only and cannot receive continuous readings.")
        if parameter not in station["parameters"]: raise DatasetError(f"Parameter '{parameter}' is not configured for {station_id}.")
        try: numeric_value=float(value)
        except ValueError as exc: raise DatasetError(f"Invalid value for {station_id}/{parameter}.") from exc
        spec=station["parameters"][parameter]; z=(numeric_value-spec["mean"])/spec["std"]
        readings.append({"vehicle":vehicle,"external_vehicle_id":external,"variant":variant,"station":station_id,"parameter":parameter,"value":numeric_value,"z":round(z,3),"mean":spec["mean"],"std":spec["std"]})
    if not readings and not operations_by_key: raise DatasetError("No usable readings or operations found.")
    # Missing operational metrics remain None instead of being silently converted to healthy values.
    operations=[]
    for op in operations_by_key.values():
        op["cycle_time"]=op["cycle_time"] if op["cycle_time"] is not None else 0
        op["queue"]=op["queue"] if op["queue"] is not None else 0
        op["utilization"]=op["utilization"] if op["utilization"] is not None else 0
        operations.append(op)
    return {"stations":stations,"readings":readings,"operations":operations,"ground_truth":[],"vehicle_count":len(vehicle_sequence),"source":"uploaded CSV"}

def load_ground_truth_csv(blob: bytes, raw: dict) -> list[dict]:
    rows=_rows(blob); missing={"vehicle_id", "station_id", "event_type"}-set(rows[0])
    if missing: raise DatasetError(f"Ground-truth CSV missing: {', '.join(sorted(missing))}")
    lookup={op["external_vehicle_id"]:op["vehicle"] for op in raw["operations"]}
    scenarios=[]
    for index,row in enumerate(rows,1):
        external=row["vehicle_id"].strip(); station=row["station_id"].strip(); kind=(row["event_type"] or "anomaly").strip()
        if external not in lookup: raise DatasetError(f"Ground truth vehicle_id '{external}' does not occur in the data CSV.")
        if station not in {s["id"] for s in raw["stations"]}: raise DatasetError(f"Ground truth station_id '{station}' is unknown.")
        start=lookup[external]
        # Event start/end can be vehicle IDs when supplied; otherwise the row vehicle is the event point.
        if row.get("event_start"): start=lookup.get(row["event_start"].strip(), start)
        end=lookup.get((row.get("event_end") or "").strip(), start)
        scenarios.append({"id":f"uploaded-{index}","kind":"bottleneck" if kind=="bottleneck" else kind,"station":station,"start_vehicle":start,"end_vehicle":end,"confirmed_by":row.get("confirmed_by", "")})
    return scenarios
