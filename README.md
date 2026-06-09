# ML1 — Deep Learning Credit Risk Scorer

PyTorch attention neural network + XGBoost ensemble for credit default probability scoring, trained on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset (307K applicants).

---

## Results

| Model | Val AUC | Gini | KS Statistic | Brier Score |
|---|---|---|---|---|
| XGBoost (baseline) | **0.7689** | 0.5379 | 0.4028 | 0.1782 |
| PyTorch Attention NN | 0.7571 | 0.5142 | 0.3814 | 0.1989 |

> Single-table only (application_train.csv). Production models incorporating auxiliary tables (bureau, previous applications) typically reach 0.80+ AUC on this dataset.

---

## Architecture

```
Input (143 features)
    │
    ▼
FeatureAttention gate
  Linear(143→286) → ReLU → Linear(286→143) → Sigmoid
  x = x * gate(x)          ← soft per-feature mask
    │
    ▼
BatchNorm → Linear(143→256) → GELU → Dropout(0.3)
BatchNorm → Linear(256→128) → GELU → Dropout(0.3)
BatchNorm → Linear(128→64)  → GELU → Dropout(0.2)
Linear(64→1) → BCEWithLogitsLoss
```

**Training:** AdamW + cosine LR schedule, `pos_weight=11.4` for 91/9 class imbalance, early stopping on val AUC (patience=8).

**Feature engineering:** 120 base features → 143 engineered (domain ratios, EXT_SOURCE aggregates, polynomial pairs, age/employment bins).

---

## Project structure

```
ml1-credit-risk-scorer/
├── src/
│   ├── data.py          # Home Credit loader, cleaning, train/val split
│   ├── features.py      # Domain ratio + polynomial feature engineering
│   ├── train.py         # XGBoost + PyTorch NN training
│   └── evaluate.py      # ROC, calibration, SHAP, metrics
├── app/
│   ├── api.py           # FastAPI prediction service
│   └── ui.py            # Streamlit interactive dashboard
├── tests/
│   └── test_data_model.py
├── models/              # Saved model artefacts (gitignored)
├── reports/             # PNG plots + metrics_summary.txt
├── data/
│   ├── raw/             # application_train.csv (gitignored)
│   └── processed/       # .npy arrays + feature name lists
└── .github/workflows/ci.yml
```

---

## Setup

```bash
git clone https://github.com/fahadamjad009/ml1-credit-risk-scorer
cd ml1-credit-risk-scorer

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install "numpy>=2.0,<2.1"
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk) and place `application_train.csv` in `data/raw/`.

---

## Usage

**Train both models:**
```bash
python -m src.data          # process + split
python -m src.features      # engineer features (smoke test)
python -m src.train --model both
python -m src.evaluate      # ROC, SHAP, calibration plots → reports/
```

**Run the API:**
```bash
uvicorn app.api:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

**Run the Streamlit dashboard:**
```bash
streamlit run app/ui.py
# → http://localhost:8501
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/models` | Model metadata |
| POST | `/predict` | Single applicant score |
| POST | `/predict/batch` | Batch scoring (max 500) |

Example:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"EXT_SOURCE_1":0.6,"EXT_SOURCE_2":0.55,"EXT_SOURCE_3":0.45,"AMT_CREDIT":450000,"AMT_ANNUITY":22500,"AMT_INCOME_TOTAL":150000}'
```

Response:
```json
{
  "xgb_default_prob": 0.122,
  "nn_default_prob": 0.134,
  "xgb_risk_label": "MEDIUM",
  "nn_risk_label": "MEDIUM",
  "inference_ms": 4.2
}
```

---

## Key dependencies

| Package | Version | Role |
|---|---|---|
| torch | 2.4.0+cpu | Neural network |
| xgboost | 3.2.0 | Gradient boosting baseline |
| numpy | 2.0.x | Array operations |
| scikit-learn | — | Preprocessing, metrics |
| shap | 0.51.0 | Model explainability |
| fastapi | — | Prediction API |
| streamlit | — | Interactive dashboard |
| plotly | — | Interactive charts |

---

## Evaluation plots

| Plot | Description |
|---|---|
| `reports/roc_curve.png` | ROC curves for both models on 61K val set |
| `reports/calibration_curve.png` | Reliability diagrams + Brier scores |
| `reports/feature_importance.png` | XGBoost top-30 features by gain |
| `reports/shap_summary.png` | SHAP beeswarm (n=2,000 sample) |

---

*Fahad Amjad · [github.com/fahadamjad009](https://github.com/fahadamjad009) · Sydney, Australia*
