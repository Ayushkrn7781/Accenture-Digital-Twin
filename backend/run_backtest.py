"""Run deterministic backtest & ML risk engine validation."""
from twin import simulate, analyze, backtest
from ml_risk import train_and_score

if __name__ == "__main__":
    raw = simulate()
    analysis = analyze(raw)
    bt = backtest(raw, analysis)
    ml = train_and_score(raw, analysis)
    
    print("=" * 60)
    print("DIGITALTWIN.AI - STATISTICAL & ML RISK BACKTEST REPORT")
    print("=" * 60)
    print(f"Validation Available : {bt['validation_available']}")
    print(f"Precision            : {bt['precision']}%")
    print(f"Anomaly Recall       : {bt['recall']}%")
    print(f"False Positives      : {bt['false_positives']}")
    print(f"Mean Detection Lag   : {bt['mean_detection_lag']} vehicles")
    print(f"Bottleneck Recall    : {bt['bottleneck_recall']}%")
    print(f"Evaluated Events     : {bt['evaluated_events']}")
    print("-" * 60)
    print("PREDICTIVE ML MODEL (GRADIENT-BOOSTED TREES)")
    print(f"Model Architecture   : {ml['model']}")
    print(f"Prediction Target    : {ml['target']}")
    print(f"High Risk Count      : {ml['summary']['high_risk_vehicles_count']} vehicles")
    print(f"Top Predictive Feature: {ml['summary']['top_predictive_feature']}")
    print("=" * 60)
