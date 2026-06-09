"""
src/features.py
Home Credit Default Risk — feature engineering.

Adds domain-informed ratio features, polynomial interactions on top
numeric predictors, and age/employment binning.

Usage:
    from src.features import build_features

    X_train_eng, X_val_eng, feature_names_eng = build_features(
        X_train, X_val, feature_names
    )
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── column index helper ───────────────────────────────────────────────────────

def _idx(feature_names: List[str], col: str) -> int | None:
    """Return column index or None if not present."""
    try:
        return feature_names.index(col)
    except ValueError:
        return None


# ── domain ratio features ─────────────────────────────────────────────────────
# Each tuple: (new_name, numerator_col, denominator_col)
# These ratios are standard in credit-risk literature and consistently
# appear in top Kaggle kernels for this dataset.

RATIO_SPECS = [
    # Debt-service burden — what fraction of income goes to the loan annuity
    ("ANNUITY_TO_INCOME",       "AMT_ANNUITY",       "AMT_INCOME_TOTAL"),
    # Loan-to-income — size of credit relative to income
    ("CREDIT_TO_INCOME",        "AMT_CREDIT",        "AMT_INCOME_TOTAL"),
    # Annuity-to-credit — implicit loan term proxy (higher = shorter tenor)
    ("ANNUITY_TO_CREDIT",       "AMT_ANNUITY",       "AMT_CREDIT"),
    # Goods price as fraction of credit — LTV proxy
    ("GOODS_TO_CREDIT",         "AMT_GOODS_PRICE",   "AMT_CREDIT"),
    # Income per family member
    ("INCOME_PER_PERSON",       "AMT_INCOME_TOTAL",  "CNT_FAM_MEMBERS"),
    # Children-to-family ratio — dependency load
    ("CHILDREN_RATIO",          "CNT_CHILDREN",      "CNT_FAM_MEMBERS"),
    # Employment length relative to age — career stability signal
    ("EMPLOYED_TO_AGE",         "DAYS_EMPLOYED",     "DAYS_BIRTH"),
    # Credit bureau enquiries (year) per credit duration proxy
    ("EXT_SOURCE_MEAN",         None,                None),   # handled separately
]

EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def _add_ratios(
    X: np.ndarray,
    feature_names: List[str],
    new_cols: List[str],
    new_data: List[np.ndarray],
) -> None:
    """Append ratio features (in-place on new_cols / new_data lists)."""
    for spec in RATIO_SPECS:
        name, num_col, den_col = spec

        # EXT_SOURCE_MEAN is handled separately below
        if num_col is None:
            continue

        ni = _idx(feature_names, num_col)
        di = _idx(feature_names, den_col)
        if ni is None or di is None:
            logger.debug("Skipping ratio %s — missing column(s).", name)
            continue

        num = X[:, ni].astype(np.float64)
        den = X[:, di].astype(np.float64)
        # safe division — zero denominator → 0
        ratio = np.where(np.abs(den) > 1e-9, num / den, 0.0)
        new_cols.append(name)
        new_data.append(ratio.astype(np.float32))

    # EXT_SOURCE mean — average of up to 3 external credit scores
    ext_indices = [_idx(feature_names, c) for c in EXT_SOURCES]
    ext_indices = [i for i in ext_indices if i is not None]
    if ext_indices:
        ext_stack = X[:, ext_indices].astype(np.float64)
        # NaN was already imputed in data.py; mean is straightforward
        ext_mean = ext_stack.mean(axis=1)
        new_cols.append("EXT_SOURCE_MEAN")
        new_data.append(ext_mean.astype(np.float32))

        # EXT_SOURCE min and max — spread signals risk dispersion
        new_cols.append("EXT_SOURCE_MIN")
        new_data.append(ext_stack.min(axis=1).astype(np.float32))
        new_cols.append("EXT_SOURCE_MAX")
        new_data.append(ext_stack.max(axis=1).astype(np.float32))
        new_cols.append("EXT_SOURCE_STD")
        new_data.append(ext_stack.std(axis=1).astype(np.float32))


# ── polynomial interactions ───────────────────────────────────────────────────
# Pairwise products of the highest-signal numeric features.
# Kept small (10 pairs) to avoid dimensionality blowup.

POLY_PAIRS = [
    ("EXT_SOURCE_1", "EXT_SOURCE_2"),
    ("EXT_SOURCE_1", "EXT_SOURCE_3"),
    ("EXT_SOURCE_2", "EXT_SOURCE_3"),
    ("EXT_SOURCE_MEAN", "CREDIT_TO_INCOME"),
    ("EXT_SOURCE_MEAN", "ANNUITY_TO_INCOME"),
    ("DAYS_BIRTH",      "EXT_SOURCE_1"),
    ("DAYS_BIRTH",      "EXT_SOURCE_2"),
    ("DAYS_BIRTH",      "EXT_SOURCE_3"),
    ("CREDIT_TO_INCOME","ANNUITY_TO_INCOME"),
    ("DAYS_EMPLOYED",   "EXT_SOURCE_2"),
]


def _add_poly(
    X: np.ndarray,
    all_names: List[str],
    new_cols: List[str],
    new_data: List[np.ndarray],
) -> None:
    """Append pairwise product features."""
    for col_a, col_b in POLY_PAIRS:
        ia = _idx(all_names, col_a)
        ib = _idx(all_names, col_b)
        if ia is None or ib is None:
            logger.debug("Skipping poly %s×%s — missing column(s).", col_a, col_b)
            continue
        product = (X[:, ia] * X[:, ib]).astype(np.float32)
        new_cols.append(f"{col_a}_x_{col_b}")
        new_data.append(product)


# ── binning ───────────────────────────────────────────────────────────────────

def _add_bins(
    X: np.ndarray,
    feature_names: List[str],
    new_cols: List[str],
    new_data: List[np.ndarray],
) -> None:
    """
    Bin age and employment length into ordinal buckets.
    DAYS_BIRTH and DAYS_EMPLOYED are negative integers (days before application).
    """
    birth_i = _idx(feature_names, "DAYS_BIRTH")
    if birth_i is not None:
        age_years = -X[:, birth_i] / 365.25
        # 5 buckets: <25, 25-35, 35-45, 45-55, 55+
        bins  = [0, 25, 35, 45, 55, 200]
        labels = [0, 1, 2, 3, 4]
        age_bin = np.digitize(age_years, bins[1:]).astype(np.float32)
        new_cols.append("AGE_BIN")
        new_data.append(age_bin)

    emp_i = _idx(feature_names, "DAYS_EMPLOYED")
    if emp_i is not None:
        # DAYS_EMPLOYED = 365243 is a sentinel for pensioners/unemployed — cap it
        emp_days = -X[:, emp_i].copy()
        emp_days = np.clip(emp_days, 0, 40 * 365)
        emp_years = emp_days / 365.25
        # 4 buckets: <1yr, 1-3yr, 3-7yr, 7+yr
        emp_bin = np.digitize(emp_years, [1, 3, 7]).astype(np.float32)
        new_cols.append("EMP_BIN")
        new_data.append(emp_bin)


# ── public API ────────────────────────────────────────────────────────────────

def build_features(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Engineer features from raw (already cleaned) arrays.

    Fit statistics are computed on X_train only; X_val uses the same
    column indices — no leakage (all operations here are deterministic
    column transformations, not fit-dependent scalers).

    Returns
    -------
    X_train_eng, X_val_eng : float32 arrays with original + new columns
    feature_names_eng      : updated feature name list
    """
    results = []
    for label, X in [("train", X_train), ("val", X_val)]:
        new_cols: List[str] = []
        new_data: List[np.ndarray] = []

        _add_ratios(X, feature_names, new_cols, new_data)

        # poly and bin need the full (original + ratio) name list
        combined_names = feature_names + new_cols
        combined_X     = np.concatenate(
            [X] + [d.reshape(-1, 1) for d in new_data], axis=1
        )
        _add_poly(combined_X, combined_names, new_cols, new_data)
        _add_bins(X, feature_names, new_cols, new_data)

        X_eng = np.concatenate(
            [X] + [d.reshape(-1, 1) for d in new_data], axis=1
        ).astype(np.float32)
        results.append(X_eng)

        if label == "train":
            logger.info(
                "Feature engineering: %d base → %d engineered features (+%d)",
                len(feature_names), X_eng.shape[1], len(new_cols),
            )
            logger.info("New features: %s", new_cols)

    feature_names_eng = feature_names + new_cols
    return results[0], results[1], feature_names_eng


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    from pathlib import Path
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    proc = Path(__file__).resolve().parents[1] / "data" / "processed"
    X_train = np.load(proc / "X_train.npy")
    X_val   = np.load(proc / "X_val.npy")
    feat    = (proc / "feature_names.txt").read_text().splitlines()

    X_tr_eng, X_va_eng, feat_eng = build_features(X_train, X_val, feat)

    print(f"\nX_train_eng: {X_tr_eng.shape}  dtype={X_tr_eng.dtype}")
    print(f"X_val_eng:   {X_va_eng.shape}  dtype={X_va_eng.dtype}")
    print(f"Total features: {len(feat_eng)}")
    print(f"New features added: {feat_eng[120:]}")

    # sanity — no NaN or Inf
    assert not np.isnan(X_tr_eng).any(), "NaN in train!"
    assert not np.isinf(X_tr_eng).any(), "Inf in train!"
    print("\nSanity check passed — no NaN or Inf.")
