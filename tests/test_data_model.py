"""
tests/test_data_model.py
Smoke tests for ML1 — Home Credit Default Risk Scorer.

Tests cover:
  - data loading and schema
  - feature engineering correctness
  - model artefact existence and prediction shape
  - API health endpoint

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Processed data artefacts
# ══════════════════════════════════════════════════════════════════════════════

class TestProcessedData:
    def test_arrays_exist(self):
        for fname in ["X_train.npy","X_val.npy","y_train.npy","y_val.npy"]:
            assert (PROC_DIR / fname).exists(), f"Missing {fname}"

    def test_feature_names_exist(self):
        assert (PROC_DIR / "feature_names.txt").exists()
        assert (PROC_DIR / "feature_names_eng.txt").exists()

    def test_shapes_consistent(self):
        X_train = np.load(PROC_DIR / "X_train.npy")
        X_val   = np.load(PROC_DIR / "X_val.npy")
        y_train = np.load(PROC_DIR / "y_train.npy")
        y_val   = np.load(PROC_DIR / "y_val.npy")
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0]   == y_val.shape[0]
        assert X_train.shape[1] == X_val.shape[1]

    def test_feature_count(self):
        X_train = np.load(PROC_DIR / "X_train.npy")
        assert X_train.shape[1] == 120, f"Expected 120 base features, got {X_train.shape[1]}"

    def test_engineered_feature_count(self):
        feat_eng = (PROC_DIR / "feature_names_eng.txt").read_text().splitlines()
        assert len(feat_eng) == 143, f"Expected 143 engineered features, got {len(feat_eng)}"

    def test_no_nan_in_arrays(self):
        for fname in ["X_train.npy", "X_val.npy"]:
            arr = np.load(PROC_DIR / fname)
            assert not np.isnan(arr).any(), f"NaN found in {fname}"

    def test_no_inf_in_arrays(self):
        for fname in ["X_train.npy", "X_val.npy"]:
            arr = np.load(PROC_DIR / fname)
            assert not np.isinf(arr).any(), f"Inf found in {fname}"

    def test_target_binary(self):
        for fname in ["y_train.npy", "y_val.npy"]:
            y = np.load(PROC_DIR / fname)
            assert set(np.unique(y)).issubset({0, 1}), f"Non-binary target in {fname}"

    def test_class_imbalance_roughly_8pct(self):
        y_train = np.load(PROC_DIR / "y_train.npy")
        pos_rate = y_train.mean()
        assert 0.06 < pos_rate < 0.12, f"Unexpected positive rate: {pos_rate:.3f}"

    def test_dtype_float32(self):
        X_train = np.load(PROC_DIR / "X_train.npy")
        assert X_train.dtype == np.float32, f"Expected float32, got {X_train.dtype}"

    def test_split_sizes(self):
        X_train = np.load(PROC_DIR / "X_train.npy")
        X_val   = np.load(PROC_DIR / "X_val.npy")
        total   = X_train.shape[0] + X_val.shape[0]
        val_pct = X_val.shape[0] / total
        assert 0.18 < val_pct < 0.22, f"Val split {val_pct:.2f} outside expected 20%"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    @pytest.fixture(scope="class")
    def arrays(self):
        X_train = np.load(PROC_DIR / "X_train.npy")
        feat    = (PROC_DIR / "feature_names.txt").read_text().splitlines()
        return X_train, feat

    def test_build_features_shape(self, arrays):
        from src.features import build_features
        X, feat = arrays
        X_small = X[:100]
        X_eng_tr, X_eng_va, feat_eng = build_features(X_small, X_small, feat)
        assert X_eng_tr.shape == (100, 143)
        assert X_eng_va.shape == (100, 143)
        assert len(feat_eng) == 143

    def test_engineered_feature_names_present(self, arrays):
        from src.features import build_features
        X, feat = arrays
        _, _, feat_eng = build_features(X[:10], X[:10], feat)
        expected = [
            "ANNUITY_TO_INCOME", "CREDIT_TO_INCOME", "EXT_SOURCE_MEAN",
            "AGE_BIN", "EMP_BIN", "EXT_SOURCE_1_x_EXT_SOURCE_2",
        ]
        for name in expected:
            assert name in feat_eng, f"Missing engineered feature: {name}"

    def test_no_nan_after_engineering(self, arrays):
        from src.features import build_features
        X, feat = arrays
        X_eng, _, _ = build_features(X[:500], X[:500], feat)
        assert not np.isnan(X_eng).any()

    def test_no_inf_after_engineering(self, arrays):
        from src.features import build_features
        X, feat = arrays
        X_eng, _, _ = build_features(X[:500], X[:500], feat)
        assert not np.isinf(X_eng).any()

    def test_dtype_preserved(self, arrays):
        from src.features import build_features
        X, feat = arrays
        X_eng, _, _ = build_features(X[:10], X[:10], feat)
        assert X_eng.dtype == np.float32


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model artefacts
# ══════════════════════════════════════════════════════════════════════════════

class TestModelArtefacts:
    def test_xgb_model_exists(self):
        assert (MODELS_DIR / "xgb_baseline.joblib").exists()

    def test_nn_model_exists(self):
        assert (MODELS_DIR / "nn_credit_risk.pt").exists()

    def test_nn_scaler_exists(self):
        assert (MODELS_DIR / "nn_scaler.joblib").exists()


# ══════════════════════════════════════════════════════════════════════════════
# 4. XGBoost predictions
# ══════════════════════════════════════════════════════════════════════════════

class TestXGBoostModel:
    @pytest.fixture(scope="class")
    def xgb_and_data(self):
        import joblib
        from src.features import build_features
        xgb  = joblib.load(MODELS_DIR / "xgb_baseline.joblib")
        X    = np.load(PROC_DIR / "X_val.npy")
        y    = np.load(PROC_DIR / "y_val.npy")
        feat = (PROC_DIR / "feature_names.txt").read_text().splitlines()
        _, Xe, _ = build_features(X[:200], X[:200], feat)
        return xgb, Xe, y[:200]

    def test_predict_proba_shape(self, xgb_and_data):
        xgb, Xe, _ = xgb_and_data
        proba = xgb.predict_proba(Xe)
        assert proba.shape == (200, 2)

    def test_predict_proba_sums_to_one(self, xgb_and_data):
        xgb, Xe, _ = xgb_and_data
        proba = xgb.predict_proba(Xe)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_probabilities_in_range(self, xgb_and_data):
        xgb, Xe, _ = xgb_and_data
        proba = xgb.predict_proba(Xe)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_val_auc_above_threshold(self, xgb_and_data):
        from sklearn.metrics import roc_auc_score
        xgb, Xe, y = xgb_and_data
        auc = roc_auc_score(y, xgb.predict_proba(Xe)[:, 1])
        assert auc > 0.70, f"XGBoost AUC {auc:.4f} below threshold 0.70"


# ══════════════════════════════════════════════════════════════════════════════
# 5. PyTorch NN predictions
# ══════════════════════════════════════════════════════════════════════════════

class TestNNModel:
    @pytest.fixture(scope="class")
    def nn_and_data(self):
        import joblib
        import torch
        from src.train import CreditRiskNN, DEVICE
        from src.features import build_features

        ckpt   = torch.load(MODELS_DIR / "nn_credit_risk.pt", map_location="cpu", weights_only=True)
        nn     = CreditRiskNN(ckpt["n_features"]).to(DEVICE)
        nn.load_state_dict(ckpt["model_state"])
        nn.eval()
        scaler = joblib.load(MODELS_DIR / "nn_scaler.joblib")

        X    = np.load(PROC_DIR / "X_val.npy")
        y    = np.load(PROC_DIR / "y_val.npy")
        feat = (PROC_DIR / "feature_names.txt").read_text().splitlines()
        _, Xe, _ = build_features(X[:200], X[:200], feat)
        Xs   = scaler.transform(Xe).astype(np.float32)
        return nn, Xs, y[:200]

    def test_output_shape(self, nn_and_data):
        import torch
        nn, Xs, _ = nn_and_data
        with torch.no_grad():
            out = nn(torch.tensor(Xs, dtype=torch.float32)).squeeze(1)
        assert out.shape == (200,)

    def test_probabilities_in_range(self, nn_and_data):
        import torch
        nn, Xs, _ = nn_and_data
        with torch.no_grad():
            proba = torch.sigmoid(nn(torch.tensor(Xs, dtype=torch.float32)).squeeze(1)).tolist()
        assert all(0.0 <= p <= 1.0 for p in proba)

    def test_val_auc_above_threshold(self, nn_and_data):
        import torch
        from sklearn.metrics import roc_auc_score
        nn, Xs, y = nn_and_data
        with torch.no_grad():
            proba = torch.sigmoid(nn(torch.tensor(Xs, dtype=torch.float32)).squeeze(1)).tolist()
        auc = roc_auc_score(y, proba)
        assert auc > 0.60, f"NN AUC {auc:.4f} below threshold 0.60"

    def test_n_features_matches_scaler(self, nn_and_data):
        import torch
        nn, Xs, _ = nn_and_data
        ckpt = torch.load(MODELS_DIR / "nn_credit_risk.pt", map_location="cpu", weights_only=True)
        assert ckpt["n_features"] == Xs.shape[1]


# ══════════════════════════════════════════════════════════════════════════════
# 6. Reports
# ══════════════════════════════════════════════════════════════════════════════

class TestReports:
    REPORTS_DIR = ROOT / "reports"

    def test_reports_dir_exists(self):
        assert self.REPORTS_DIR.exists()

    @pytest.mark.parametrize("fname", [
        "roc_curve.png",
        "calibration_curve.png",
        "feature_importance.png",
        "shap_summary.png",
        "metrics_summary.txt",
    ])
    def test_report_file_exists(self, fname):
        assert (self.REPORTS_DIR / fname).exists(), f"Missing report: {fname}"

    def test_metrics_summary_contains_auc(self):
        txt = (self.REPORTS_DIR / "metrics_summary.txt").read_text()
        assert "AUC" in txt
        assert "XGB" in txt or "xgb" in txt.lower()
