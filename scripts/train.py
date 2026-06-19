"""
Train an autoregressive zebrafish behavior model.

Usage
-----
    uv run scripts/train.py --config configs/default.toml --output runs/run_001/

The script will write to the output directory:
    checkpoint.pt   - model weights, scalers, config, feature columns
    df_base.pkl     - pre-lag dataframe used to seed sampling
    loss_curve.png  - training and validation loss over epochs
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from common import load_config, prepare_data
from src.model_NN import save_checkpoint, train_nn_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Train a zebrafish behavior model.")
    p.add_argument("--config", required=True, type=Path, help="Path to TOML config file.")
    p.add_argument("--output", required=True, type=Path, help="Directory to write outputs.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1–4. Load, filter, stack, preprocess
    # ------------------------------------------------------------------ #
    df_for_training, df_base = prepare_data(cfg)

    base_path = output_dir / "df_base.pkl"
    with open(base_path, "wb") as f:
        pickle.dump(df_base, f)
    print(f"Saved df_base → {base_path}")

    # ------------------------------------------------------------------ #
    # 5. Train
    # ------------------------------------------------------------------ #
    train_cfg = cfg["training"]
    model, scalers, splits, train_losses, val_losses = train_nn_pipeline(
        df_for_training,
        early_stopping=train_cfg.get("early_stopping", False),
        train_frac=train_cfg.get("train_frac", 0.8),
        lr=train_cfg.get("lr", 1e-3),
        batch_size=train_cfg.get("batch_size", 64),
        num_epochs=train_cfg.get("num_epochs", 5000),
        N1=cfg["model"].get("hidden_sizes", [128])[0],
        patience=train_cfg.get("patience", 10),
        seed=train_cfg.get("seed", 0),
    )

    # ------------------------------------------------------------------ #
    # 6. Save loss curve
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL loss")
    ax.legend()
    fig.savefig(output_dir / "loss_curve.png", dpi=120)
    plt.close(fig)
    print(f"Saved loss curve → {output_dir / 'loss_curve.png'}")

    # ------------------------------------------------------------------ #
    # 7. Save checkpoint
    # ------------------------------------------------------------------ #
    # feature_cols = all columns except targets
    target_cols = ["f_speed_next", "d_ori_next"]
    feature_cols = [c for c in df_for_training.columns if c not in target_cols]

    ckpt_path = output_dir / "checkpoint.pt"
    save_checkpoint(ckpt_path, model, scalers, cfg, feature_cols)
    print(f"Saved checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
