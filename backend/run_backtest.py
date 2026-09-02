"""Full validation report. Every figure the dashboard and README quote is printed here.

    python run_backtest.py            # demo line
    python run_backtest.py --json     # machine-readable, used to regenerate the README
    python run_backtest.py --config config/line_config_plant_b.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from economics import compute, rollout_case
from metrics import backtest, false_alarm_profile, sensor_gap_study
from ml_risk import DEMO_SEED, get_model, score
from state import build_state
from twin import analyze, simulate
from twin import CONFIG as DEFAULT_CONFIG


def collect(config: dict, seeds: int = 10, with_ml: bool = True) -> dict:
    raw = simulate(config, seed=DEMO_SEED)
    analysis = analyze(raw, config)
    bt = backtest(raw, analysis, config)
    fa = false_alarm_profile(config, seeds=seeds)
    gap = sensor_gap_study(config)
    ml = score(raw, analysis, config) if with_ml else None
    st = build_state(raw, analysis, bt, ml, config=config)
    ec = compute(bt, fa, st["line"])
    return {"line": config["line"]["name"], "backtest": bt, "false_alarms": fa,
            "sensor_gap": gap, "throughput": {k: st["line"][k] for k in
                                              ("line_uph", "nominal_uph", "takt_s", "constraint_station")},
            "ml": (ml or {}).get("validation"), "ml_model": (ml or {}).get("model"),
            "propagation": analysis["propagation"], "economics": ec,
            "rollout": rollout_case(ec, config.get("scalability", {})),
            "calibration": config["bottleneck"].get("calibrated")}


def render(r: dict) -> str:
    bt, fa, ml, ec = r["backtest"], r["false_alarms"], r["ml"] or {}, r["economics"]
    cal, tp = r["calibration"] or {}, r["throughput"]
    w, out = 66, []
    line = lambda c="-": out.append(c * w)
    head = lambda t: (line("="), out.append(t), line("="))

    head(f"DIGITALTWIN.AI VALIDATION REPORT — {r['line']}")
    out.append("\nDETECTION (vs injected ground truth)")
    for k, label in [("precision", "Anomaly precision"), ("recall", "Anomaly recall"),
                     ("bottleneck_precision", "Bottleneck precision"),
                     ("bottleneck_recall", "Bottleneck recall"),
                     ("median_detection_lag", "Median lead time (veh)"),
                     ("p90_detection_lag", "p90 lead time (veh)"),
                     ("false_positives", "False-positive stations"),
                     ("detected_events", "Events detected"), ("evaluated_events", "Events evaluated")]:
        out.append(f"  {label:<32} {bt.get(k)}")
    out.append(f"  {'Latent origin identified':<32} {bt.get('latent_origin_identified')}"
               f"  ({bt.get('latent_origin_station')})")

    out.append("\nFALSE-ALARM FLOOR (injection-free runs — every alarm is false by construction)")
    for k, label in [("total_false_alarms_per_run", "Total per run"),
                     ("anomaly_false_alarms_per_run", "  anomaly"),
                     ("bottleneck_false_alarms_per_run", "  bottleneck"),
                     ("physics_false_alarms_per_run", "  physics"),
                     ("false_alarms_per_1000_vehicles", "Per 1,000 vehicles"),
                     ("arl0_vehicles", "ARL0 (vehicles)"), ("seeds", "Runs measured")]:
        out.append(f"  {label:<32} {fa.get(k)}")

    if cal:
        out.append("\nCALIBRATED LIMITS (swept to a stated budget, confirmed on held-out runs)")
        out.append(f"  {'EWMA L':<32} {cal.get('ewma_L')}")
        out.append(f"  {'CUSUM h (sigma)':<32} {cal.get('cusum_h_sigma')}")
        out.append(f"  {'Oven tau z-critical':<32} {cal.get('tau_fit_z_crit')}")
        out.append(f"  {'Budget / measured per run':<32} {cal.get('budget_false_alarms_per_run')}"
                   f" / {(cal.get('measured_false_alarms_per_run_heldout') or {}).get('total')}")

    out.append("\nTHROUGHPUT")
    out.append(f"  {'Line rate (bottleneck-governed)':<32} {tp['line_uph']} UPH")
    out.append(f"  {'Nominal rate':<32} {tp['nominal_uph']} UPH")
    out.append(f"  {'Constraint station':<32} {tp['constraint_station']} (takt {tp['takt_s']}s)")

    if ml:
        out.append(f"\nRISK MODEL — {r['ml_model']} (held-out lines only)")
        for k, label in [("pr_auc", "PR-AUC"), ("pr_auc_baseline", "No-skill baseline"),
                         ("lift_over_baseline", "Lift"), ("roc_auc", "ROC-AUC"),
                         ("brier_score", "Brier score"), ("precision_at_40", "Precision@40"),
                         ("positive_rate_pct", "Positive rate (%)"), ("held_out_rows", "Held-out rows")]:
            out.append(f"  {label:<32} {ml.get(k)}")

    gap = r["sensor_gap"]
    out.append("\nSENSOR COVERAGE STUDY")
    out.append(f"  {'coverage':<12}{'recall':>10}{'precision':>12}{'bn recall':>12}{'lead':>8}{'trace':>8}")
    for lv in gap["levels"]:
        out.append(f"  {int(lv['instrumented_ratio']*100):>3}%{'':<8}{str(lv['recall']):>10}"
                   f"{str(lv['precision']):>12}{str(lv['bottleneck_recall']):>12}"
                   f"{str(lv['median_detection_lag']):>8}"
                   f"{('yes' if lv['latent_origin_identified'] else 'no'):>8}")

    out.append("\nBUSINESS CASE (computed from the metrics above)")
    a = ec["annual"]
    for k, label in [("defects_prevented", "Defects prevented / yr"),
                     ("rework_avoided_usd", "Rework avoided"), ("scrap_avoided_usd", "Scrap avoided"),
                     ("throughput_value_usd", "Throughput recovered"),
                     ("investigation_cost_usd", "Less false-alarm cost"),
                     ("net_benefit_usd", "Net annual benefit")]:
        out.append(f"  {label:<32} {a[k]:,}")
    out.append(f"  {'Payback (months)':<32} {r['rollout']['payback_months']}")
    line("=")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-ml", action="store_true")
    args = p.parse_args()

    config = json.loads(args.config.read_text()) if args.config else DEFAULT_CONFIG
    if not args.no_ml:
        get_model()
    result = collect(config, seeds=args.seeds, with_ml=not args.no_ml)
    print(json.dumps(result, indent=2, default=str) if args.json else render(result))


if __name__ == "__main__":
    main()
