from . import functions
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path




def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------  Model registry  ------------------------------------

MODEL_REGISTRY: dict[str, type] = {"mlp": None}  # filled after Network is defined


def build_model(n_features: int, model_config: dict) -> "Network":
    """Instantiate a model from the model section of a config dict."""
    kind = model_config.get("type", "mlp")
    if kind not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{kind}'. Available: {list(MODEL_REGISTRY)}")
    hidden_sizes = model_config.get("hidden_sizes", [128])
    return MODEL_REGISTRY[kind](n_features, hidden_sizes[0])


def save_checkpoint(path, model, scalers, config, feature_cols):
    """Save model weights, scalers, config and feature column names to a single file."""
    path = Path(path)
    model_config = config.get("model", {})
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "n_features": len(feature_cols),
                "type": model_config.get("type", "mlp"),
                "hidden_sizes": model_config.get("hidden_sizes", [128]),
            },
            "scalers": scalers,
            "feature_cols": feature_cols,
            "config": config,
        },
        path,
    )


def load_checkpoint(path):
    """Load checkpoint and return (model, scalers, config, feature_cols)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["model_config"]["n_features"], ck["model_config"])
    model.load_state_dict(ck["model_state"])
    return model, ck["scalers"], ck["config"], ck["feature_cols"]


# -----------------------------------------------------------------------------------------


def train_nn_pipeline(
    df: pd.DataFrame,
    early_stopping: bool = True,
    train_frac: float = 0.8,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_epochs: int = 100,
    N1: int = 128,
    patience: int = 10,
    seed: int = 0,
    device: torch.device = None,
):
    """
    Train probabilistic neural network on preprocessed dataframe.

    Input
    -----
    df : DataFrame returned by preprocessing_pipeline.
         Must already contain:
             - lagged features
             - engineered features
             - targets: "f_speed_next", "d_ori_next"

    What it does
    ------------
    1) Split df into train (train_frac) / val (1-train_frac) temporally.
    2) Normalize inputs and targets (fit scalers on train only).
    3) Train neural network with early stopping on val loss.
    4) Return trained model and fitted scalers.

    Returns
    -------
    model : torch.nn.Module
    scalers : dict  {"X": MinMaxScaler, "speed": MinMaxScaler, "ori": MinMaxScaler}
    splits : dict   {"train": {"X", "speed", "d_ori"}, "val": {...}}
    train_losses, val_losses : list[float]
    """
    
    target_cols = ["f_speed_next", "d_ori_next"]

    # Temporal train / val split
    n = len(df)
    split = int(n * train_frac)
    X = df.drop(columns=target_cols)
    y = df[target_cols]

    X_train_raw, X_val_raw = X.iloc[:split], X.iloc[split:]
    y_train,     y_val     = y.iloc[:split], y.iloc[split:]

    old_columns = X_train_raw.columns.tolist()

    # ------------------------ Standardize features and targets ------------------------
    # ---- X ----
    scaler_X = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train = scaler_X.fit_transform(X_train_raw)
    X_val   = scaler_X.transform(X_val_raw)

    # ---- speed ----
    scaler_speed = MinMaxScaler(feature_range=(0.0, 1.0))
    target_speed_train = scaler_speed.fit_transform(y_train["f_speed_next"].to_numpy().reshape(-1,1)).ravel()
    target_speed_val   = scaler_speed.transform(y_val["f_speed_next"].to_numpy().reshape(-1,1)).ravel()

    # ---- orientation ----
    scaler_ori = MinMaxScaler(feature_range=(-1.0, 1.0))
    target_ori_train = scaler_ori.fit_transform(y_train["d_ori_next"].to_numpy().reshape(-1,1)).ravel()
    target_ori_val   = scaler_ori.transform(y_val["d_ori_next"].to_numpy().reshape(-1,1)).ravel()

    scalers = {"X": scaler_X, "speed": scaler_speed, "ori": scaler_ori}

    X_train = pd.DataFrame(X_train, columns=old_columns)
    X_val   = pd.DataFrame(X_val,   columns=old_columns)

    target_speed_train = pd.Series(target_speed_train, name="speed")
    target_speed_val   = pd.Series(target_speed_val,   name="speed")
    target_ori_train   = pd.Series(target_ori_train,   name="ori")
    target_ori_val     = pd.Series(target_ori_val,     name="ori")

    splits = {
        "train": {"X": X_train, "speed": target_speed_train, "d_ori": target_ori_train},
        "val":   {"X": X_val,   "speed": target_speed_val,   "d_ori": target_ori_val},
    }
    
    
    X_train_np = X_train.to_numpy(dtype="float32")
    y_train_speed_np = target_speed_train.to_numpy(dtype="float32").reshape(-1, 1)
    y_train_dori_np = target_ori_train.to_numpy(dtype="float32").reshape(-1, 1)


    dataset = CustomDataset(X_train_np, y_train_speed_np, y_train_dori_np)

    if device is None:
        device = _get_device()

    pin = device.type == "cuda"
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Network(X_train.shape[1], N1).to(device)

    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-5)

    
    # Validation tensors
    X_val_np = X_val.to_numpy(dtype="float32")
    y_val_speed_np = target_speed_val.to_numpy(dtype="float32").reshape(-1, 1)
    y_val_dori_np = target_ori_val.to_numpy(dtype="float32").reshape(-1, 1)

    val_dataset = CustomDataset(X_val_np, y_val_speed_np, y_val_dori_np)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    
    
    # ---- Before training loop ----
    train_losses = []
    val_losses = []
    

    #  ----       
    min_delta = 1e-4       
    best_val = float("inf")
    epochs_no_improve = 0
    best_state_dict = None

    for epoch in range(num_epochs):

        # =========================
        # TRAIN
        # =========================
        model.train()
        train_loss_sum = 0.0
        num_batches = 0

        for x, y_true_speed, y_true_dori in data_loader:
            x, y_true_speed, y_true_dori = x.to(device), y_true_speed.to(device), y_true_dori.to(device)
            out = model(x)

            mu_st    = out[:, 0]
            sigma_st = out[:, 1]
            p_pos    = out[:, 2]
            mu_ln    = out[:, 3]
            sigma_ln = out[:, 4]

            loss_value = loss(y_true_speed, y_true_dori, mu_st, sigma_st, p_pos, mu_ln, sigma_ln)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            train_loss_sum += float(loss_value.item())
            num_batches += 1

        train_loss_avg = train_loss_sum / max(1, num_batches)
        train_losses.append(train_loss_avg)

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        val_loss_sum = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for x_val, y_val_speed, y_val_dori in val_loader:
                x_val, y_val_speed, y_val_dori = x_val.to(device), y_val_speed.to(device), y_val_dori.to(device)
                out = model(x_val)

                mu_st    = out[:, 0]
                sigma_st = out[:, 1]
                p_pos    = out[:, 2]
                mu_ln    = out[:, 3]
                sigma_ln = out[:, 4]

                loss_value = loss(y_val_speed, y_val_dori, mu_st, sigma_st, p_pos, mu_ln, sigma_ln)
                val_loss_sum += float(loss_value.item())
                num_val_batches += 1

        val_loss_avg = val_loss_sum / max(1, num_val_batches)
        val_losses.append(val_loss_avg)

        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train loss: {train_loss_avg:.6f} | "
            f"Val loss: {val_loss_avg:.6f}")

        # =========================
        # EARLY STOPPING (on val loss)
        # =========================
        if early_stopping:
            if val_loss_avg < (best_val - min_delta):
                best_val = val_loss_avg
                epochs_no_improve = 0
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered (no improvement for {patience} epochs). "
                        f"Best val loss: {best_val:.6f}")
                    break

    # ---- Restore best model weights  ----
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model = model.to("cpu")

    return model, scalers, splits, train_losses, val_losses











# -------------------------------------     Dataloader     --------------------------------------
        
class CustomDataset(Dataset):
    def __init__(self, X_np, y_speed_np, y_dori_np):
        self.X = torch.from_numpy(X_np)          # (N, 260)
        self.y_speed = torch.from_numpy(y_speed_np)          # (N, 1)
        self.y_dori = torch.from_numpy(y_dori_np)          # (N, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_speed[idx], self.y_dori[idx]




    
    
    
    
    
# -------------------------------------     Network     --------------------------------------

class Network(nn.Module):
    def __init__(self, N_init, N1):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(N_init, N1)
        self.linear3 = nn.Linear(N1, 5)

    def forward(self, x):
        z = F.relu(self.linear1(x))
        z = self.linear3(z)
        return z


# Register after class definition
MODEL_REGISTRY["mlp"] = Network





# -------------------------------------     Loss     --------------------------------------

def loss(
        y_speed: torch.Tensor,
        y_dori: torch.Tensor,             
        mu_st: torch.Tensor,          
        sigma_st: torch.Tensor,       
        p_pos: torch.Tensor, # Probability of positive speed     
        mu_ln: torch.Tensor,          
        sigma_ln: torch.Tensor,       
        nu: float = 5.0,              
        eps: float = 1e-6,
    ) -> torch.Tensor:
        
        
        # flatten
        y_speed = y_speed.squeeze(-1)
        y_dori = y_dori.squeeze(-1)
        mu_st = mu_st.squeeze(-1)
        sigma_st = sigma_st.squeeze(-1)
        p_pos = p_pos.squeeze(-1)
        mu_ln = mu_ln.squeeze(-1)
        sigma_ln = sigma_ln.squeeze(-1)

        # reparametrization for valid domains (no clamping)
        sigma_st = torch.nn.functional.softplus(sigma_st) + eps
        sigma_ln = torch.nn.functional.softplus(sigma_ln) + eps
        sigma_min, sigma_max = 1e-3, 1.0  # à ajuster (souvent 0.3–1.5 max)
        sigma_ln = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(sigma_ln)
        p_pos    = torch.sigmoid(p_pos)

        # ---------- Student-t log pdf (location-scale) ----------
        # log f(y) = log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5*log(nu*pi) - log(sigma)
        #            - ((nu+1)/2)*log(1 + ((y-mu)/sigma)^2 / nu)
        z = (y_dori - mu_st) / sigma_st
        logc = (
            torch.lgamma(torch.tensor((nu + 1.0) / 2.0, device=y_dori.device, dtype=y_dori.dtype))
            - torch.lgamma(torch.tensor(nu / 2.0, device=y_dori.device, dtype=y_dori.dtype))
            - 0.5 * (math.log(nu) + math.log(math.pi))
            - torch.log(sigma_st)
        )
        log_phi_st = logc - ((nu + 1.0) / 2.0) * torch.log1p((z * z) / nu)

        # ---------- Zero-inflated lognormal log-likelihood ----------
        # if y==0: log(1 - p_pos)
        # if y>0 : log(p_pos) + log f_ln(y)
        y_pos = y_speed > 0
        # lognormal pdf: f(y)= 1/(y*sigma*sqrt(2pi)) * exp(-(log y - mu)^2/(2 sigma^2))
        y_safe = torch.where(y_pos, y_speed, torch.ones_like(y_speed))  # avoid log(0); won't be used when y==0
        logy = torch.log(y_safe)

        log_phi_ln = (
            -torch.log(y_safe)
            - torch.log(sigma_ln)
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * ((logy - mu_ln) / sigma_ln) ** 2
        )

        log_like_lnB = torch.where(
            y_pos,
            torch.log(p_pos + eps) + log_phi_ln,
            torch.log1p(-p_pos + eps)  # log(1-p)
        )

        # ---------- Total negative log-likelihood ----------
        # matches your "Loss = L_st + L_ln,B" up to averaging conventions
        loss = -(log_phi_st + log_like_lnB).mean()
        return loss
    
    
    
    
    