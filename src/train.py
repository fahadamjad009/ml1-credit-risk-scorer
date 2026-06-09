"""
src/train.py
Home Credit Default Risk — model training.

Two models:
  1. XGBoost baseline  — fast, interpretable, strong benchmark
  2. PyTorch NN        — TabNet-style MLP with a feature-attention gate

Both are saved to models/ with their val-AUC logged.

Usage:
    python -m src.train               # trains both
    python -m src.train --model xgb   # baseline only
    python -m src.train --model nn    # neural net only
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from src.data import get_splits
from src.features import build_features

logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PROC_DIR   = ROOT / "data" / "processed"
MODELS_DIR.mkdir(exist_ok=True)

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PyTorch model definition
# ══════════════════════════════════════════════════════════════════════════════

class FeatureAttention(nn.Module):
    """
    Soft feature-attention gate.
    Learns a per-feature importance weight via a small 2-layer MLP,
    then multiplies the input element-wise. Analogous to the first
    step of TabNet's sequential attention but simpler and faster.
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_features, n_features * 2),
            nn.ReLU(),
            nn.Linear(n_features * 2, n_features),
            nn.Sigmoid(),          # outputs in (0, 1) — soft mask
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class CreditRiskNN(nn.Module):
    """
    Tabular MLP for binary credit-risk classification.

    Architecture:
        FeatureAttention(143) →
        BN → Linear(143→256) → GELU → Dropout(0.3) →
        BN → Linear(256→128) → GELU → Dropout(0.3) →
        BN → Linear(128→64)  → GELU → Dropout(0.2) →
        Linear(64→1)          → sigmoid (applied at inference)

    GELU preferred over ReLU for tabular data — smoother gradient
    for the skewed financial features.
    """
    def __init__(self, n_features: int, dropout1: float = 0.3, dropout2: float = 0.2):
        super().__init__()
        self.attention = FeatureAttention(n_features)
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 256),
            nn.GELU(),
            nn.Dropout(dropout1),

            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout1),

            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout2),

            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        return self.net(x)          # raw logits; BCEWithLogitsLoss expects these


# ══════════════════════════════════════════════════════════════════════════════
# 2. Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss from class counts."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    w = n_neg / n_pos
    logger.info("pos_weight = %.2f  (neg=%d / pos=%d)", w, n_neg, n_pos)
    return torch.tensor(w, dtype=torch.float32)


def _make_loaders(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    y_train: np.ndarray,
    y_val:   np.ndarray,
    batch_size: int = 2048,
) -> tuple[DataLoader, DataLoader]:
    t = lambda a: torch.from_numpy(a)
    train_ds = TensorDataset(t(X_train), t(y_train.astype(np.float32)))
    val_ds   = TensorDataset(t(X_val),   t(y_val.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader


def _val_auc(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb).squeeze(1)
            preds.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(yb.numpy())
    return roc_auc_score(np.concatenate(targets), np.concatenate(preds))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Train functions
# ══════════════════════════════════════════════════════════════════════════════

def train_xgb(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    y_train: np.ndarray,
    y_val:   np.ndarray,
) -> float:
    """Train XGBoost baseline. Returns val AUC."""
    logger.info("── XGBoost baseline ──────────────────────────────────────")
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = n_neg / n_pos,   # handle imbalance
        eval_metric       = "auc",
        early_stopping_rounds = 30,
        random_state      = SEED,
        n_jobs            = -1,
        verbosity         = 0,
        device            = "cpu",
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    elapsed = time.time() - t0

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    logger.info("XGBoost val AUC: %.4f  (%.0fs, best iter=%d)", val_auc, elapsed, model.best_iteration)

    path = MODELS_DIR / "xgb_baseline.joblib"
    joblib.dump(model, path)
    logger.info("Saved → %s", path)
    return val_auc


def train_nn(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    y_train: np.ndarray,
    y_val:   np.ndarray,
    n_epochs:   int   = 50,
    lr:         float = 3e-4,
    batch_size: int   = 2048,
    patience:   int   = 8,
) -> float:
    """Train PyTorch NN with early stopping. Returns best val AUC."""
    logger.info("── PyTorch NN ────────────────────────────────────────────")
    logger.info("Device: %s", DEVICE)

    # scale — NN sensitive to feature magnitude; XGB is not
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    joblib.dump(scaler, MODELS_DIR / "nn_scaler.joblib")

    n_features = X_train_s.shape[1]
    model = CreditRiskNN(n_features).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y_train).to(DEVICE))

    train_loader, val_loader = _make_loaders(X_train_s, X_val_s, y_train, y_val, batch_size)

    best_auc   = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)

        scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)
        val_auc  = _val_auc(model, val_loader)
        elapsed  = time.time() - t0

        logger.info(
            "Epoch %2d/%d | loss=%.4f | val_auc=%.4f | lr=%.2e | %.0fs",
            epoch, n_epochs, avg_loss, val_auc,
            scheduler.get_last_lr()[0], elapsed,
        )

        if val_auc > best_auc + 1e-4:
            best_auc   = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # restore best weights and save
    model.load_state_dict(best_state)
    torch.save(
        {"model_state": model.state_dict(), "n_features": n_features},
        MODELS_DIR / "nn_credit_risk.pt",
    )
    logger.info("Best val AUC: %.4f — saved → %s", best_auc, MODELS_DIR / "nn_credit_risk.pt")
    return best_auc


# ══════════════════════════════════════════════════════════════════════════════
# 4. Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(model: str = "both") -> None:
    # load base splits
    X_train_raw, X_val_raw, y_train, y_val, feat = get_splits()

    # engineer features
    X_train, X_val, feat_eng = build_features(X_train_raw, X_val_raw, feat)

    # persist feature names for API/UI
    (PROC_DIR / "feature_names_eng.txt").write_text("\n".join(feat_eng))

    results: dict[str, float] = {}

    if model in ("xgb", "both"):
        results["xgb"] = train_xgb(X_train, X_val, y_train, y_val)

    if model in ("nn", "both"):
        results["nn"] = train_nn(X_train, X_val, y_train, y_val)

    print("\n── Results ───────────────────────────────────────────────")
    for name, auc in results.items():
        print(f"  {name:4s}  val AUC = {auc:.4f}")
    print("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["xgb", "nn", "both"],
        default="both",
        help="Which model to train (default: both)",
    )
    args = parser.parse_args()
    main(args.model)
