"""Export the simulated line as CSVs matching the upload contract.

Produces a line-data extract and its ground-truth companion, so the replay path can
be exercised end to end with data the twin did not generate in-process.

(This file was previously named for an AI4I converter it did not contain. The risk
model is trained on independently generated lines from this repository - see
ml_risk.py - and no external benchmark is used or claimed.)
"""
from __future__ import annotations

import csv
from pathlib import Path

from twin import CONFIG, simulate

OUTPUTS = Path(__file__).parent.parent / "outputs"

DATA_HEADER = ["timestamp", "vehicle_id", "product_variant", "station_id", "parameter",
               "value", "cycle_time_seconds", "queue_length", "utilization_pct", "manual_outcome"]
GT_HEADER = ["vehicle_id", "station_id", "event_type", "event_start", "event_end", "confirmed_by"]


def export(config: dict = CONFIG, seed: int = 42) -> tuple[Path, Path]:
    OUTPUTS.mkdir(exist_ok=True)
    raw = simulate(config, seed=seed)
    count = raw["vehicle_count"]

    ops = {(o["vehicle"], o["station"]): o for o in raw["operations"]}
    reads: dict[tuple, list] = {}
    for r in raw["readings"]:
        reads.setdefault((r["vehicle"], r["station"]), []).append(r)

    data_path = OUTPUTS / "benchmark_line_data.csv"
    with data_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(DATA_HEADER)
        for v in range(1, count + 1):
            vid = f"V-{v:04d}"
            ts = f"2026-01-05T{6 + v // 60:02d}:{v % 60:02d}:00Z"
            for s in raw["stations"]:
                op = ops.get((v, s["id"]))
                if not op:
                    continue
                rows = reads.get((v, s["id"]), [])
                if rows:
                    for r in rows:
                        w.writerow([ts, vid, op["variant"], s["id"], r["parameter"], r["value"],
                                    op["cycle_time"], op["queue"], op["utilization"], ""])
                else:
                    w.writerow([ts, vid, op["variant"], s["id"], "", "",
                                op["cycle_time"], op["queue"], op["utilization"],
                                op.get("manual_outcome") or "pass"])

    gt_path = OUTPUTS / "benchmark_ground_truth.csv"
    with gt_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(GT_HEADER)
        for item in config["injections"]:
            start = item.get("start_vehicle") or min(item.get("vehicles", [1]))
            end = item.get("end_vehicle") or max(item.get("vehicles", [start]))
            w.writerow([f"V-{start:04d}", item["station"], item["kind"],
                        f"V-{start:04d}", f"V-{end:04d}", "Quality Audit Lead"])
    return data_path, gt_path


if __name__ == "__main__":
    data, gt = export()
    print(f"Line data    : {data}")
    print(f"Ground truth : {gt}")
