from pathlib import Path
from src.data.make_dataset import main

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]  # project root
    main(
        cfg_path=str(ROOT / "configs/data.yaml"),
        feats_cfg=str(ROOT / "configs/features.yaml"),
    )
