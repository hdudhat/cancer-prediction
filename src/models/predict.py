import sys
import pandas as pd, joblib
from src.utils.io import load_yaml

def main(in_csv, out_csv, data_cfg="configs/data.yaml", feat_cfg="configs/features.yaml"):
    data_cfg = load_yaml(data_cfg); feat_cfg = load_yaml(feat_cfg)
    feats = feat_cfg["base_features"] + feat_cfg["engineered"]

    df = pd.read_csv(in_csv)
    pipe = joblib.load("models/artifacts/best_pipeline.joblib")
    df["probability"] = pipe.predict_proba(df[feats])[:,1]
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
