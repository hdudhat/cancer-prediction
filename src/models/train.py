from pathlib import Path
import json
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from src.utils.io import load_yaml, ensure_dir, save_json
from src.features.build_features import make_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def _make_model(spec):
    t = spec["type"]
    params = spec.get("params", {})
    if t == "logreg":
        return LogisticRegression(**params)
    if t == "rf":
        return RandomForestClassifier(**params)
    if t == "gb":
        return GradientBoostingClassifier(**params)
    raise ValueError(f"Unknown model type: {t}")

def _metric_dict(y_true, proba):
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba))
    }

def _choose_threshold(y_true, proba, specificity_target=0.90):
    # grid-search threshold to meet specificity; maximize sensitivity
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_sens = 0.5, -1
    y_true = np.asarray(y_true)
    for t in thresholds:
        yhat = (proba >= t).astype(int)
        tn = ((yhat==0)&(y_true==0)).sum()
        fp = ((yhat==1)&(y_true==0)).sum()
        fn = ((yhat==0)&(y_true==1)).sum()
        tp = ((yhat==1)&(y_true==1)).sum()
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        if spec >= specificity_target and sens > best_sens:
            best_sens, best_t = sens, float(t)
    return float(best_t)

def main(
    data_cfg_path="configs/data.yaml",
    feat_cfg_path="configs/features.yaml",
    model_cfg_path="configs/model.yaml",
    artifacts_dir="models/artifacts",
    metrics_dir="models/metrics"
):
    data_cfg = load_yaml(data_cfg_path)
    feat_cfg = load_yaml(feat_cfg_path)
    model_cfg = load_yaml(model_cfg_path)

    processed = Path(data_cfg["processed_dir"])
    train = pd.read_csv(processed / "train.csv")
    val = pd.read_csv(processed / "val.csv")

    target = data_cfg["target"]
    base = feat_cfg["base_features"]
    eng = feat_cfg["engineered"]
    feats = base + eng
    scale_cols = feat_cfg["scale_columns"]

    X_train, y_train = train[feats], train[target]
    X_val, y_val = val[feats], val[target]

    # build candidates
    preprocess = make_preprocess(scale_cols)
    results = {}
    pipes = {}
    for name, spec in load_yaml(model_cfg_path)["candidates"].items():
        clf = _make_model(spec)
        pipe = Pipeline([("pre", preprocess), ("clf", clf)])
        pipe.fit(X_train, y_train)
        p_val = pipe.predict_proba(X_val)[:,1]
        metrics = _metric_dict(y_val, p_val)
        results[name] = metrics
        pipes[name] = pipe
        print(name, metrics)

    # select best
    key = model_cfg.get("selection_metric", "pr_auc")
    best_name = max(results, key=lambda k: results[k][key])
    best_pipe = pipes[best_name]
    print("Selected:", best_name)

    # choose threshold on validation
    p_val = best_pipe.predict_proba(X_val)[:,1]
    threshold = _choose_threshold(y_val, p_val, model_cfg.get("specificity_target", 0.90))

    # save artifacts & metrics
    ensure_dir(artifacts_dir); ensure_dir(metrics_dir)
    pipe_path = Path(artifacts_dir) / "best_pipeline.joblib"
    joblib.dump(best_pipe, pipe_path)

    save_json({
        "model": best_name,
        "val": results[best_name],
        "threshold": threshold
    }, Path(metrics_dir) / "val_metrics.json")

    print("Saved model ->", pipe_path)
    print("Saved metrics -> models/metrics/val_metrics.json")

if __name__ == "__main__":
    main()
