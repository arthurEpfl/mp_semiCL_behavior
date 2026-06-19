"""Shared data-loading and preprocessing logic used by train.py and preprocess.py."""

import pickle
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import functions
from src.preprocessing import preprocessing_pipeline


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def prepare_data(cfg: dict):
    """Load abs_dict, filter fish, stack with dish geometry, and preprocess.

    Returns
    -------
    df_for_training : pd.DataFrame  — lagged feature matrix with targets
    df_base         : pd.DataFrame  — pre-lag frame used to seed sampling
    """
    data_cfg = cfg["data"]
    pkl_path = Path(data_cfg["pkl_path"])

    print(f"Loading abs_dict from {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        abs_dict = pickle.load(f)

    fish_ids = data_cfg.get("fish_ids", None)
    if fish_ids is not None:
        abs_dict = {f"f{i}": abs_dict[f"f{i}"] for i in fish_ids if f"f{i}" in abs_dict}
        print(f"Using fish: {list(abs_dict.keys())}")
    else:
        print(f"Using all fish: {list(abs_dict.keys())}")

    dish_x = data_cfg.get("dish_x", None)
    dish_y = data_cfg.get("dish_y", None)
    dish_radius = data_cfg.get("dish_radius", None)
    if dish_radius is not None:
        print(f"Dish geometry: center=({dish_x}, {dish_y}), radius={dish_radius} px")
    df_raw = functions.stack_dict_pipeline(
        abs_dict, dish_x=dish_x, dish_y=dish_y, dish_radius=dish_radius
    )

    df_for_training, df_base = preprocessing_pipeline(
        df_raw,
        cols_to_lag=data_cfg["cols_to_lag"],
        horizon_frames=data_cfg.get("horizon_frames", 1),
        subsample=data_cfg.get("subsample", 1),
        max_lag_seconds=data_cfg.get("max_lag_seconds", 3),
    )

    return df_for_training, df_base
