"""Business case computed from measured detector performance.

Nothing here is a quoted figure. Benefits scale with the recall and lead time the
backtest actually achieved, and costs include the investigation burden of the
false-alarm rate actually measured on fault-free runs - so improving the detector
moves the business case, and a noisy detector is charged for the noise it creates.

The unit economics themselves are illustrative assumptions, declared in
``config/economics.json`` and reported alongside every result.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from twin import ROOT

ECONOMICS = json.loads((ROOT / "config" / "economics.json").read_text())


def _actionable_fraction(lead_vehicles: float, cfg: dict = ECONOMICS) -> float:
    """How much of a correctly-detected incident the floor can actually prevent.

    Detecting a fault one vehicle before it lands is worth far less than detecting it
    twenty vehicles out. Treating every detection as fully preventable is the single
    most common way this kind of model overstates its value.
    """
    curve = cfg["actionability"]
    return float(np.interp(max(0.0, lead_vehicles),
                           curve["lead_time_vehicles"], curve["fraction_actionable"]))


def compute(metrics: dict, false_alarms: dict, throughput: dict | None = None,
            cfg: dict = ECONOMICS) -> dict[str, Any]:
    """Annualised business case for one line."""
    a = cfg["assumptions"]
    vehicles_year = a["vehicles_per_year"]

    recall = (metrics.get("recall") or 0.0) / 100.0
    precision = (metrics.get("precision") or 0.0) / 100.0
    lead = metrics.get("median_detection_lag")
    lead = float(lead) if lead is not None else 0.0
    actionable = _actionable_fraction(lead, cfg)

    # --- benefit: defects prevented -------------------------------------
    baseline_defects = vehicles_year * a["baseline_defect_rate"]
    defects_prevented = baseline_defects * recall * actionable
    rework_saved = defects_prevented * a["rework_cost_per_defect_usd"]
    scrap_saved = defects_prevented * a["scrap_rate_of_defects"] * a["scrap_cost_per_vehicle_usd"]

    # --- benefit: throughput recovered ----------------------------------
    # What earlier detection actually buys is the *duration* of degraded running that
    # is avoided, not a year of it. A constraint detected N vehicles sooner saves N
    # vehicles' worth of degraded operation per occurrence - so the benefit is the
    # loss rate multiplied by the lead time and the number of occurrences, never by
    # the whole year. Annualising the peak gap is how this figure reaches absurdity.
    nominal_uph = float((throughput or {}).get("nominal_uph") or 0.0)
    recoverable_uph = float((throughput or {}).get("recoverable_uph") or 0.0)
    loss_fraction = (recoverable_uph / nominal_uph) if nominal_uph else 0.0
    # Degradation ramps rather than stepping, so the average loss over the avoided
    # window is about half the peak.
    mean_loss_fraction = loss_fraction * 0.5

    events_per_run = float((throughput or {}).get("bottleneck_events_per_run") or 0.0)
    runs_per_year = vehicles_year / max(float((false_alarms or {}).get("vehicles_per_run") or 1), 1.0)
    events_per_year = events_per_run * runs_per_year

    vehicles_recovered = mean_loss_fraction * lead * events_per_year * actionable
    cap = vehicles_year * a.get("max_recoverable_share", 0.05)
    capped = vehicles_recovered > cap
    vehicles_recovered = min(vehicles_recovered, cap)
    throughput_value = vehicles_recovered * a["contribution_margin_per_vehicle_usd"]

    # --- cost: false alarms ---------------------------------------------
    fa_per_run = float(false_alarms.get("total_false_alarms_per_run") or 0.0)
    runs_per_year = vehicles_year / max(false_alarms.get("vehicles_per_run") or 1, 1)
    false_alarms_year = fa_per_run * runs_per_year
    investigation_cost = false_alarms_year * a["investigation_cost_per_false_alarm_usd"]

    gross = rework_saved + scrap_saved + throughput_value
    net = gross - investigation_cost

    band = cfg["sensitivity"]["band"]
    return {
        "currency": cfg["currency"],
        "assumptions": a,
        "derived_from_measurement": {
            "anomaly_recall_pct": metrics.get("recall"),
            "anomaly_precision_pct": metrics.get("precision"),
            "median_detection_lead_vehicles": lead,
            "actionable_fraction": round(actionable, 3),
            "false_alarms_per_run": round(fa_per_run, 2),
            "arl0_vehicles": false_alarms.get("arl0_vehicles"),
        },
        "annual": {
            "defects_prevented": round(defects_prevented),
            "rework_avoided_usd": round(rework_saved),
            "scrap_avoided_usd": round(scrap_saved),
            "throughput_value_usd": round(throughput_value),
            "vehicles_recovered": round(vehicles_recovered),
            "vehicles_recovered_capped": capped,
            "constraint_events_per_year": round(events_per_year),
            "mean_loss_fraction_while_degraded": round(mean_loss_fraction, 4),
            "gross_benefit_usd": round(gross),
            "false_alarm_investigations": round(false_alarms_year),
            "investigation_cost_usd": round(investigation_cost),
            "net_benefit_usd": round(net),
        },
        "sensitivity": {
            "band_pct": int(band * 100),
            "net_benefit_low_usd": round(net * (1 - band)),
            "net_benefit_high_usd": round(net * (1 + band)),
            "dominant_assumptions": cfg["sensitivity"]["dominant_assumptions"],
        },
        "note": ("Throughput benefit is the degraded-running time avoided by detecting a "
                 "constraint earlier - loss rate x lead time x occurrences - not an annualised "
                 "peak gap. Benefit scales with measured recall and lead time; false-alarm "
                 "investigation cost is subtracted using the rate measured on "
                 "injection-free runs. Unit economics are stated assumptions, not "
                 "observations from a specific plant."),
    }


def rollout_case(economics: dict, scalability: dict) -> dict[str, Any]:
    """Payback against the configured rollout phases."""
    phases = scalability.get("rollout_phases", [])
    total_cost = sum(p.get("cost_estimate_usd", 0) for p in phases)
    total_weeks = sum(p.get("duration_weeks", 0) for p in phases)
    net = economics["annual"]["net_benefit_usd"]
    payback_months = round(total_cost / (net / 12.0), 1) if net > 0 else None
    return {
        "phases": phases,
        "total_implementation_cost_usd": total_cost,
        "total_duration_weeks": total_weeks,
        "annual_net_benefit_usd": net,
        "payback_months": payback_months,
        "first_year_roi_pct": round(100.0 * (net - total_cost) / total_cost, 1) if total_cost else None,
    }


def retrofit_case(sensor_plan: list[dict], gap_study: dict, economics: dict) -> dict[str, Any]:
    """Price the sensor retrofit plan against measured coverage sensitivity.

    The benefit of adding instrumentation is not asserted - it is the recall the
    coverage study shows is lost when instrumentation is absent, valued at the same
    unit economics as everything else.
    """
    per_point = gap_study.get("recall_lost_per_coverage_point")
    annual_net = economics["annual"]["net_benefit_usd"]
    rows = []
    for item in sensor_plan:
        n = len(item.get("target_stations", []))
        cost = n * item.get("unit_cost_usd", 0)
        coverage_gain = n / 40.0
        value = (annual_net * per_point * coverage_gain) if per_point else None
        rows.append({
            **item,
            "station_count": n,
            "total_cost_usd": cost,
            "coverage_gain_fraction": round(coverage_gain, 3),
            "estimated_annual_value_usd": round(value) if value else None,
            "payback_months": (round(cost / (value / 12.0), 1)
                               if value and value > 0 else None),
        })
    return {
        "items": rows,
        "total_capex_usd": sum(r["total_cost_usd"] for r in rows),
        "basis": ("Value per point of regained instrumentation coverage is taken from the "
                  "measured sensor-coverage study, not assumed."),
    }
