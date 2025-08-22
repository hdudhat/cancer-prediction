from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.io import load_yaml, ensure_dir
from .validate_schema import coerce_and_validate

def main(cfg_path="configs/data.yml", feats_cfg="configs/features.yml"):
    data_cfg = load_yaml(cfg_path)
    feat_cfg = load_yaml(feats_cfg)

    raw_path = Path(data_cfg["raw_path"])
    assert raw_path.exists(), f"CSV not found: {raw_path}"

    df = pd.read_csv(raw_path)
    features = feat_cfg["base_features"]
    target = data_cfg["target"]
    df = coerce_and_validate(df, features, target)

    # add engineered columns (compute here so splits persist the same FE)
    df["Age2"] = df["Age"] ** 2
    df["BMI2"] = df["BMI"] ** 2
    df["BMI_x_PhysicalActivity"] = df["BMI"] * df["PhysicalActivity"]
    df["Smoking_x_GeneticRisk"] = df["Smoking"] * df["GeneticRisk"]
    df["History_x_GeneticRisk"] = df["CancerHistory"] * df["GeneticRisk"]

    seed = data_cfg["seed"]
    test_size = data_cfg["test_size"]
    val_size = data_cfg["val_size"]

    # 1) test split
    X, y = df.drop(columns=[target]), df[target]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    # 2) val split (as fraction of full data)
    val_ratio_of_temp = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_temp, stratify=y_temp, random_state=seed
    )

    out_dir = Path(data_cfg["processed_dir"])
    ensure_dir(out_dir)
    X_train.assign(**{target: y_train}).to_csv(out_dir / "train.csv", index=False)
    X_val.assign(**{target: y_val}).to_csv(out_dir / "val.csv", index=False)
    X_test.assign(**{target: y_test}).to_csv(out_dir / "test.csv", index=False)
    print("Saved splits to", out_dir)

if __name__ == "__main__":
    main()
