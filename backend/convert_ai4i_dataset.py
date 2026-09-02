"""Dataset Generator & AI4I Converter for DigitalTwin.ai.

Generates:
1. Full 420-vehicle benchmark line data CSV and ground truth CSV matching the 40-station topology.
2. Converter script to map AI4I 2020 Predictive Maintenance CSV into the DigitalTwin.ai line contract.
"""
from __future__ import annotations
import csv
from pathlib import Path
from twin import simulate, CONFIG

ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

def export_benchmark_dataset():
    raw = simulate(seed=42)
    
    # 1. Export Line Data CSV
    data_csv_path = OUTPUTS_DIR / "benchmark_line_data.csv"
    with open(data_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "vehicle_id", "product_variant", "station_id", 
            "parameter", "value", "cycle_time_seconds", "queue_length", 
            "utilization_pct", "manual_outcome"
        ])
        
        # Build lookup of operations by (vehicle, station)
        op_lookup = {(op["vehicle"], op["station"]): op for op in raw["operations"]}
        
        # Group readings by (vehicle, station)
        readings_by_key = {}
        for r in raw["readings"]:
            readings_by_key.setdefault((r["vehicle"], r["station"]), []).append(r)
            
        stations = raw["stations"]
        vehicle_count = raw["line"]["vehicle_count"] if "line" in raw else 420
        
        for v in range(1, vehicle_count + 1):
            v_id = f"V-{v:04d}"
            variant = CONFIG["product_variants"][(v - 1) % len(CONFIG["product_variants"])]
            ts = f"2026-01-05T{6 + (v // 60):02d}:{(v % 60):02d}:00Z"
            
            for s in stations:
                sid = s["id"]
                op = op_lookup.get((v, sid), {"cycle_time": 50.0, "queue": 2, "utilization": 65.0, "manual_outcome": None})
                r_list = readings_by_key.get((v, sid), [])
                
                if s["instrumented"] and r_list:
                    for r in r_list:
                        writer.writerow([
                            ts, v_id, variant, sid, r["parameter"], r["value"],
                            op["cycle_time"], op["queue"], op["utilization"], ""
                        ])
                else:
                    # Manual station
                    outcome = op.get("manual_outcome") or "pass"
                    writer.writerow([
                        ts, v_id, variant, sid, "", "",
                        op["cycle_time"], op["queue"], op["utilization"], outcome
                    ])
                    
    # 2. Export Ground Truth CSV
    gt_csv_path = OUTPUTS_DIR / "benchmark_ground_truth.csv"
    with open(gt_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vehicle_id", "station_id", "event_type", "event_start", "event_end", "confirmed_by"])
        for item in CONFIG["injections"]:
            start_v = item.get("start_vehicle", min(item.get("vehicles", [1])))
            end_v = item.get("end_vehicle", max(item.get("vehicles", [start_v])))
            writer.writerow([
                f"V-{start_v:04d}",
                item["station"],
                item["kind"],
                f"V-{start_v:04d}",
                f"V-{end_v:04d}",
                "Quality Audit Lead"
            ])
            
    print(f"Exported Benchmark Dataset:")
    print(f" - Line Data    : {data_csv_path}")
    print(f" - Ground Truth : {gt_csv_path}")

if __name__ == "__main__":
    export_benchmark_dataset()
