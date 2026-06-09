"""
src/evaluate.py
Home Credit Default Risk — model evaluation.

Produces:
  reports/roc_curve.png           — ROC curves for both models
  reports/calibration_curve.png   — reliability diagrams
  reports/feature_importance.png  — XGBoost top-30 gain importance
  reports/shap_summary.png        — SHAP beeswarm for XGBoost top-20
  reports/metrics_summary.txt     — AUC, Gini, KS statistic

Usage:
    python -m src.evaluate
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import shap
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler

from src.data import get_splits
from src.features import build_features
from src.train import CreditRiskNN, MODELS_DIR, DEVICE

logger = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
PROC_DIR    = ROOT / "data" / "processed"

# ── style ─────────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
ORANGE = "#F59E0B"
RED    = "#EF4444"
GREY   = "#6B7280"
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        11,
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load models + data
# ══════════════════════════════════════════════════════════════════════════════

def _load_data():
    X_train_raw, X_val_raw, y_train, y_val, feat = get_splits()
    X_train, X_val, feat_eng = build_features(X_train_raw, X_val_raw, feat)
    return X_train, X_val, y_train, y_val, feat_eng


def _load_xgb():
    path = MODELS_DIR / "xgb_baseline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"XGBoost model not found at {path}. Run src.train first.")
    return joblib.load(path)


def _load_nn(n_features: int) -> tuple[nn.Module, StandardScaler]:
    pt_path  = MODELS_DIR / "nn_credit_risk.pt"
    sc_path  = MODELS_DIR / "nn_scaler.joblib"
    if not pt_path.exists():
        raise FileNotFoundError(f"NN model not found at {pt_path}. Run src.train first.")
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)
    model = CreditRiskNN(n_features).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler = joblib.load(sc_path)
    return model, scaler


def _nn_predict(model: nn.Module, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    X_s = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        # avoid torch.from_numpy — copy via tensor() to bypass numpy C API bridge
        tensor = torch.tensor(X_s, dtype=torch.float32).to(DEVICE)
        logits = model(tensor).squeeze(1)
        return np.array(logits.sigmoid().cpu().detach().tolist(), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Metrics
# ══════════════════════════════════════════════════════════════════════════════

def _ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between default and non-default score dists."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def compute_metrics(
    y_val: np.ndarray,
    xgb_prob: np.ndarray,
    nn_prob: np.ndarray,
) -> dict:
    metrics = {}
    for name, prob in [("xgb", xgb_prob), ("nn", nn_prob)]:
        auc    = roc_auc_score(y_val, prob)
        gini   = 2 * auc - 1
        ks     = _ks_statistic(y_val, prob)
        brier  = brier_score_loss(y_val, prob)
        metrics[name] = dict(auc=auc, gini=gini, ks=ks, brier=brier)
        logger.info(
            "%s | AUC=%.4f | Gini=%.4f | KS=%.4f | Brier=%.4f",
            name.upper(), auc, gini, ks, brier,
        )
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 3. Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc(y_val, xgb_prob, nn_prob) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, prob, color in [
        ("XGBoost", xgb_prob, BLUE),
        ("PyTorch NN", nn_prob, ORANGE),
    ]:
        fpr, tpr, _ = roc_curve(y_val, prob)
        auc = roc_auc_score(y_val, prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Home Credit Default Risk")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    path = REPORTS_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_calibration(y_val, xgb_prob, nn_prob) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, prob, color in [
        ("XGBoost", xgb_prob, BLUE),
        ("PyTorch NN", nn_prob, ORANGE),
    ]:
        fraction_pos, mean_pred = calibration_curve(y_val, prob, n_bins=15, strategy="quantile")
        brier = brier_score_loss(y_val, prob)
        ax.plot(mean_pred, fraction_pos, "o-", color=color, lw=2,
                label=f"{name}  (Brier = {brier:.4f})")

    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve — Reliability Diagram")
    ax.legend()

    path = REPORTS_DIR / "calibration_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_feature_importance(xgb_model, feat_eng: list[str]) -> None:
    importance = xgb_model.feature_importances_
    indices    = np.argsort(importance)[-30:][::-1]
    top_names  = [feat_eng[i] for i in indices]
    top_vals   = importance[indices]

    fig, ax = plt.subplots(figsize=(9, 8))
    bars = ax.barh(range(30), top_vals[::-1], color=BLUE, alpha=0.85)
    ax.set_yticks(range(30))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost — Top 30 Features by Gain")
    ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)

    path = REPORTS_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_shap(xgb_model, X_val: np.ndarray, feat_eng: list[str]) -> None:
    logger.info("Computing SHAP values (sample of 2000 rows) …")
    # subsample for speed — SHAP TreeExplainer is O(n*depth)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_val), size=min(2000, len(X_val)), replace=False)
    X_sample = X_val[idx]

    explainer  = shap.TreeExplainer(xgb_model)
    shap_vals  = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_vals, X_sample,
        feature_names=feat_eng,
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary — XGBoost (sample n=2000)", pad=12)

    path = REPORTS_DIR / "shap_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def save_metrics(metrics: dict) -> None:
    lines = ["Home Credit Default Risk — Evaluation Summary", "=" * 50, ""]
    for model_name, m in metrics.items():
        lines += [
            f"Model : {model_name.upper()}",
            f"  AUC   : {m['auc']:.4f}",
            f"  Gini  : {m['gini']:.4f}",
            f"  KS    : {m['ks']:.4f}",
            f"  Brier : {m['brier']:.4f}",
            "",
        ]
    path = REPORTS_DIR / "metrics_summary.txt"
    path.write_text("\n".join(lines))
    logger.info("Saved → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("Loading data …")
    X_train, X_val, y_train, y_val, feat_eng = _load_data()

    logger.info("Loading models …")
    xgb_model      = _load_xgb()
    nn_model, scaler = _load_nn(X_val.shape[1])

    logger.info("Generating predictions …")
    xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
    nn_prob  = _nn_predict(nn_model, scaler, X_val)

    metrics = compute_metrics(y_val, xgb_prob, nn_prob)
    save_metrics(metrics)

    logger.info("Plotting …")
    plot_roc(y_val, xgb_prob, nn_prob)
    plot_calibration(y_val, xgb_prob, nn_prob)
    plot_feature_importance(xgb_model, feat_eng)
    plot_shap(xgb_model, X_val, feat_eng)

    print("\n── Evaluation complete ───────────────────────────────────")
    for model_name, m in metrics.items():
        print(f"  {model_name.upper():10s} AUC={m['auc']:.4f}  Gini={m['gini']:.4f}  KS={m['ks']:.4f}")
    print(f"\nReports saved to: {REPORTS_DIR}")
    print("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
