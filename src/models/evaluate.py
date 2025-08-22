from pathlib import Path
import pandas as pd, joblib
from src.utils.io import load_yaml, save_json
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def main(data_cfg="configs/data.yaml", feat_cfg="configs/features.yaml"):
    data_cfg = load_yaml(data_cfg)
    feat_cfg = load_yaml(feat_cfg)
    feats = feat_cfg["base_features"] + feat_cfg["engineered"]

    test = pd.read_csv(Path(data_cfg["processed_dir"]) / "test.csv")
    X_test, y_test = test[feats], test[data_cfg["target"]]

    pipe = joblib.load("models/artifacts/best_pipeline.joblib")
    p = pipe.predict_proba(X_test)[:,1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, p)),
        "pr_auc": float(average_precision_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p))
    }
    save_json(metrics, "models/metrics/test_metrics.json")
    print("Test:", {k: round(v,4) for k,v in metrics.items()})

if __name__ == "__main__":
    main()
