from pathlib import Path
from src.models.train import main

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    main(
        data_cfg_path=str(ROOT / "configs/data.yaml"),
        feat_cfg_path=str(ROOT / "configs/features.yaml"),
        model_cfg_path=str(ROOT / "configs/model.yaml"),
        artifacts_dir=str(ROOT / "models/artifacts"),
        metrics_dir=str(ROOT / "models/metrics"),
    )
