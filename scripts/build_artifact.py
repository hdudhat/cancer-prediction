from pathlib import Path
from src.models.evaluate import main

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    main(
        data_cfg=str(ROOT / "configs/data.yaml"),
        feat_cfg=str(ROOT / "configs/features.yaml"),
    )
