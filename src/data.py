"""
src/data.py
Home Credit Default Risk — data loading, cleaning, and train/test split.

Usage:
    from src.data import load_data, get_splits

    X_train, X_val, y_train, y_val, feature_names = get_splits()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

TRAIN_CSV = RAW_DIR / "application_train.csv"

# ── constants ─────────────────────────────────────────────────────────────────
TARGET_COL   = "TARGET"
ID_COL       = "SK_ID_CURR"
RANDOM_STATE = 42
VAL_SIZE     = 0.20

# Columns dropped before modelling:
#   • ID (leaks identity)
#   • high-cardinality free-text (ORGANIZATION_TYPE has 58 levels — kept; FLAG_DOCUMENT_* kept as-is)
DROP_COLS: List[str] = [ID_COL]


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_target(df: pd.DataFrame) -> pd.Series:
    """
    Return binary target series.
    Home Credit TARGET is already 0/1 integers; this validates and casts.
    Raises ValueError if TARGET contains unexpected values.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found in dataframe.")

    target = df[TARGET_COL]
    unique_vals = set(target.dropna().unique())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"Unexpected values in TARGET: {unique_vals - {0, 1}}"
        )

    n_missing = target.isna().sum()
    if n_missing:
        logger.warning("TARGET has %d missing values — rows dropped.", n_missing)

    return target.astype(np.int8)


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode object columns.
    Binary categoricals (2 unique values) get 0/1.
    Multi-class categoricals get integer codes.
    NaN in categoricals becomes -1 (handled downstream by imputer/embedding).
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        # fillna to a sentinel string so NaN gets a consistent code
        df[col] = le.fit_transform(df[col].fillna("__MISSING__"))
    return df


def _impute_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median-impute all remaining numeric NaNs.
    Median is robust to the heavy skew common in financial features.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    medians   = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)
    return df


# ── public API ────────────────────────────────────────────────────────────────

def load_data(
    path: Path = TRAIN_CSV,
    drop_cols: List[str] = DROP_COLS,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and minimally clean the Home Credit application CSV.

    Returns
    -------
    X : pd.DataFrame  — feature matrix (all numeric after encoding)
    y : pd.Series     — binary target (int8, 0/1)
    """
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d cols", *df.shape)

    y = _resolve_target(df)

    # drop rows where target is missing
    mask = y.notna()
    df   = df[mask].reset_index(drop=True)
    y    = y[mask].reset_index(drop=True)

    # drop unwanted columns
    df = df.drop(columns=[c for c in drop_cols if c in df.columns] + [TARGET_COL])

    # encode then impute
    df = _encode_categoricals(df)
    df = _impute_numerics(df)

    logger.info("Feature matrix shape after cleaning: %s", df.shape)
    logger.info(
        "Class balance — 0: %d (%.1f%%)  1: %d (%.1f%%)",
        (y == 0).sum(), (y == 0).mean() * 100,
        (y == 1).sum(), (y == 1).mean() * 100,
    )

    return df, y


def get_splits(
    path: Path = TRAIN_CSV,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load data and return stratified train / validation splits as numpy arrays.

    Returns
    -------
    X_train, X_val  : float32 numpy arrays
    y_train, y_val  : int8 numpy arrays
    feature_names   : list of column names (same order as X columns)
    """
    X, y = load_data(path)
    feature_names = list(X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X.values.astype(np.float32),
        y.values.astype(np.int8),
        test_size=val_size,
        random_state=random_state,
        stratify=y,          # preserve 8% default rate in both splits
    )

    logger.info(
        "Split → train: %d rows | val: %d rows | features: %d",
        len(X_train), len(X_val), len(feature_names),
    )

    # persist processed splits for downstream reuse
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROC_DIR / "X_train.npy", X_train)
    np.save(PROC_DIR / "X_val.npy",   X_val)
    np.save(PROC_DIR / "y_train.npy", y_train)
    np.save(PROC_DIR / "y_val.npy",   y_val)

    feat_path = PROC_DIR / "feature_names.txt"
    feat_path.write_text("\n".join(feature_names))
    logger.info("Processed arrays saved to %s", PROC_DIR)

    return X_train, X_val, y_train, y_val, feature_names


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    X_train, X_val, y_train, y_val, feat = get_splits()
    print(f"\nX_train: {X_train.shape}  dtype={X_train.dtype}")
    print(f"X_val:   {X_val.shape}  dtype={X_val.dtype}")
    print(f"y_train: {y_train.shape}  pos_rate={y_train.mean():.3f}")
    print(f"y_val:   {y_val.shape}  pos_rate={y_val.mean():.3f}")
    print(f"Features ({len(feat)}): {feat[:5]} ... {feat[-3:]}")
