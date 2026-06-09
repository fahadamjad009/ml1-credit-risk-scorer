"""
app/ui.py  —  Credit Risk Scorer · dark theme
streamlit run app/ui.py
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

from src.train import CreditRiskNN, MODELS_DIR, DEVICE
from src.features import build_features
from src.data import PROC_DIR

st.set_page_config(page_title="Credit Risk Scorer", page_icon="💳",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .stApp,[data-testid="stAppViewContainer"]{background:#0d0d0d;color:#d4d4d4}
  [data-testid="stSidebar"]{background:#0a0a0a;border-right:1px solid #1a1a1a}
  [data-testid="stSidebar"] label,[data-testid="stSidebar"] p,
  [data-testid="stSidebar"] .stSlider,[data-testid="stSidebar"] div{color:#ccc !important;font-weight:600 !important}
  [data-testid="stSidebar"] .stMarkdown strong{color:#ffffff !important;font-size:.9rem;font-weight:800 !important}
  .tag{font-family:"Courier New",monospace;font-size:.7rem;font-weight:800;
       color:#00ff88;letter-spacing:4px;text-transform:uppercase;
       border-left:3px solid #00ff88;padding-left:10px;margin:28px 0 14px 0}
  .pcard{background:#111;border:1px solid #222;border-radius:8px;padding:24px 20px;text-align:center}
  .mlabel{font-family:"Courier New",monospace;font-size:.7rem;font-weight:800;
          color:#00ff88;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px}
  .pval{font-family:"Courier New",monospace;font-size:3rem;font-weight:800;line-height:1.1}
  .pdesc{font-size:.8rem;color:#999;font-weight:600;margin:6px 0 12px}
  .badge{font-family:"Courier New",monospace;font-size:.72rem;font-weight:800;
         letter-spacing:2px;padding:4px 14px;border-radius:12px;display:inline-block}
  .bl{background:rgba(0,255,136,.1);color:#00ff88;border:1px solid rgba(0,255,136,.3)}
  .bm{background:rgba(255,170,0,.1); color:#ffaa00;border:1px solid rgba(255,170,0,.3)}
  .bh{background:rgba(255,80,80,.1); color:#ff5050;border:1px solid rgba(255,80,80,.3)}
  .rcard{background:#111;border:1px solid #222;border-radius:8px;padding:16px;text-align:center}
  .rlabel{font-family:"Courier New",monospace;font-size:.65rem;font-weight:800;
          color:#00ff88;letter-spacing:3px;text-transform:uppercase;margin-bottom:4px}
  .rval{font-family:"Courier New",monospace;font-size:1.9rem;color:#ffffff;font-weight:800;margin:4px 0 2px}
  .rhint{font-size:.7rem;color:#888;font-weight:600}
  #MainMenu,footer,header{visibility:hidden}
  .block-container{padding-top:1.2rem}
</style>
""", unsafe_allow_html=True)

# ── plotly base — NO margin here ──────────────────────────────────────────────
def pl_layout(**extra):
    base = dict(
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#111",
        font=dict(color="#888", family="Courier New, monospace", size=11),
        xaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#1e1e1e",
                   tickfont=dict(color="#ccc", size=11)),
        yaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#1e1e1e",
                   tickfont=dict(color="#ccc", size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#222",
                    font=dict(color="#ccc", size=11)),
        title=dict(font=dict(color="#ccc", size=12, family="Courier New")),
    )
    base.update(extra)
    return base

def risk_label(p): return "LOW" if p<.10 else ("MEDIUM" if p<.25 else "HIGH")
def badge_cls(p):  return "bl"  if p<.10 else ("bm"     if p<.25 else "bh")
def pcolor(p):     return "#00ff88" if p<.10 else ("#ffaa00" if p<.25 else "#ff5050")

# ── model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models(v=3):  # bump v to bust cache
    xgb    = joblib.load(MODELS_DIR / "xgb_baseline.joblib")
    ckpt   = torch.load(MODELS_DIR / "nn_credit_risk.pt", map_location="cpu", weights_only=True)
    nn_net = CreditRiskNN(ckpt["n_features"]).to(DEVICE)
    nn_net.load_state_dict(ckpt["model_state"])
    nn_net.eval()
    scaler     = joblib.load(MODELS_DIR / "nn_scaler.joblib")
    feat_names = (PROC_DIR / "feature_names_eng.txt").read_text().splitlines()
    feat_base  = (PROC_DIR / "feature_names.txt").read_text().splitlines()

    X_val = np.load(PROC_DIR / "X_val.npy")
    y_val = np.load(PROC_DIR / "y_val.npy")
    _, Xve, _ = build_features(X_val, X_val, feat_base)
    xgb_vp = xgb.predict_proba(Xve)[:, 1]

    Xvs = scaler.transform(Xve).astype(np.float32)
    nn_vp_list = []
    with torch.no_grad():
        for i in range(0, len(Xvs), 4096):
            b = torch.tensor(Xvs[i:i+4096], dtype=torch.float32)
            nn_vp_list.extend(nn_net(b).squeeze(1).sigmoid().tolist())
    nn_vp = np.array(nn_vp_list, dtype=np.float32)

    # compute medians from training data for sane UI defaults
    X_train = np.load(PROC_DIR / "X_train.npy")
    _, Xte, _ = build_features(X_train, X_train, feat_base)
    feat_medians = np.median(Xte, axis=0).astype(np.float32)
    return xgb, nn_net, scaler, feat_names, y_val, xgb_vp, nn_vp, feat_medians

def score(xgb, nn_net, scaler, feat_names, inputs, feat_medians):
    # default to training medians so NN input is in-distribution
    row = {f: float(feat_medians[i]) for i, f in enumerate(feat_names[:120])}
    for k, v in inputs.items():
        if k in row: row[k] = float(v)
    X  = np.array([[row[f] for f in feat_names[:120]]], dtype=np.float32)
    _, Xe, _ = build_features(X, X, feat_names[:120])
    xp = float(xgb.predict_proba(Xe)[:, 1][0])
    Xs = scaler.transform(Xe).astype(np.float32)
    with torch.no_grad():
        np_ = float(nn_net(torch.tensor(Xs, dtype=torch.float32)).squeeze(1).sigmoid().item())
    return xp, np_, Xe

xgb_model, nn_model, scaler, feat_names, y_val, xgb_vp, nn_vp, feat_medians = load_models()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Courier New;font-size:.6rem;color:#00ff88;letter-spacing:4px;font-weight:700;margin-bottom:14px">▶ APPLICANT INPUT</div>', unsafe_allow_html=True)
    st.markdown("**Credit scores**")
    ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.60, 0.01)
    ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.55, 0.01)
    ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.45, 0.01)
    st.markdown("**Loan**")
    amt_credit  = st.number_input("Credit amount",  value=450_000, step=10_000)
    amt_annuity = st.number_input("Annual annuity", value=22_500,  step=1_000)
    amt_income  = st.number_input("Annual income",  value=150_000, step=5_000)
    amt_goods   = st.number_input("Goods price",    value=400_000, step=10_000)
    st.markdown("**Demographics**")
    age  = st.slider("Age (years)", 18, 70, 35)
    emp  = st.slider("Employment (years)", 0, 30, 5)
    kids = st.slider("Children", 0, 10, 1)
    fam  = st.slider("Family members", 1, 10, 3)
    st.divider()
    st.caption("ML1 · Home Credit · 307K rows")

inputs = dict(EXT_SOURCE_1=ext1, EXT_SOURCE_2=ext2, EXT_SOURCE_3=ext3,
              AMT_CREDIT=amt_credit, AMT_ANNUITY=amt_annuity,
              AMT_INCOME_TOTAL=amt_income, AMT_GOODS_PRICE=amt_goods,
              DAYS_BIRTH=-age*365, DAYS_EMPLOYED=-emp*365,
              CNT_CHILDREN=kids, CNT_FAM_MEMBERS=fam)
xp, np_, Xe = score(xgb_model, nn_model, scaler, feat_names, inputs, feat_medians)
ens = (xp + np_) / 2

# ── header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="font-family:Courier New;font-size:2rem;font-weight:700;color:#d4d4d4;letter-spacing:-1px;margin:0 0 4px 0">💳 Credit Risk Scorer</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-family:Courier New;font-size:.65rem;color:#888;font-weight:700;letter-spacing:3px;text-transform:uppercase;margin:0">PyTorch Attention NN · XGBoost · 307K Applicants · 143 Features · Home Credit Default Risk</p>', unsafe_allow_html=True)

# ── prediction cards ──────────────────────────────────────────────────────────
st.markdown('<div class="tag">Live Prediction</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
for col, label, prob in [(c1,"XGBoost",xp),(c2,"PyTorch NN",np_),(c3,"Ensemble",ens)]:
    with col:
        st.markdown(f"""<div class="pcard">
          <div class="mlabel">{label}</div>
          <div class="pval" style="color:{pcolor(prob)}">{prob:.1%}</div>
          <div class="pdesc">default probability</div>
          <span class="badge {badge_cls(prob)}">{risk_label(prob)}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── gauges ────────────────────────────────────────────────────────────────────
g1, g2 = st.columns(2)
def gauge(prob, title):
    c = pcolor(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob*100, 1),
        number=dict(suffix="%", font=dict(color=c, size=32, family="Courier New")),
        title=dict(text=title, font=dict(color="#555", size=11, family="Courier New")),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor="#222",
                      tickfont=dict(color="#444", size=9)),
            bar=dict(color=c, thickness=0.2),
            bgcolor="#111",
            bordercolor="#1e1e1e",
            steps=[
                dict(range=[0,10],  color="rgba(0,255,136,0.05)"),
                dict(range=[10,25], color="rgba(255,170,0,0.05)"),
                dict(range=[25,100],color="rgba(255,80,80,0.05)"),
            ],
            threshold=dict(line=dict(color=c, width=2),
                           thickness=0.75, value=prob*100),
        )
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=10),
                      paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
                      font=dict(color="#888", family="Courier New"))
    return fig

with g1: st.plotly_chart(gauge(xp,  "XGBoost Default Probability"),  use_container_width=True)
with g2: st.plotly_chart(gauge(np_, "PyTorch NN Default Probability"), use_container_width=True)

# ── risk ratios ───────────────────────────────────────────────────────────────
st.markdown('<div class="tag">Key Risk Ratios</div>', unsafe_allow_html=True)
debt = amt_annuity/amt_income if amt_income else 0
ci   = amt_credit/amt_income  if amt_income else 0
ltv  = amt_goods/amt_credit   if amt_credit else 0
extm = (ext1+ext2+ext3)/3

r1,r2,r3,r4 = st.columns(4)
for col, lbl, val, hint in [
    (r1,"Annuity / Income", f"{debt:.2f}",  ">0.5 = high debt burden"),
    (r2,"Credit / Income",  f"{ci:.1f}×",   ">5× = elevated risk"),
    (r3,"LTV (goods/credit)",f"{ltv:.1%}",  "<80% preferred"),
    (r4,"Avg credit score",  f"{extm:.3f}", "0–1, higher = better"),
]:
    with col:
        st.markdown(f"""<div class="rcard">
          <div class="rlabel">{lbl}</div>
          <div class="rval">{val}</div>
          <div class="rhint">{hint}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── analytics tabs ────────────────────────────────────────────────────────────
st.markdown('<div class="tag">Model Analytics</div>', unsafe_allow_html=True)
t1, t2, t3, t4, t5 = st.tabs(["ROC Curves","Score Distribution","Feature Importance","Model Comparison","SHAP"])

with t1:
    fxg,txg,_ = roc_curve(y_val, xgb_vp)
    fnn,tnn,_ = roc_curve(y_val, nn_vp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fxg, y=txg, name="XGBoost  AUC=0.7689",
                             line=dict(color="#00ff88", width=2)))
    fig.add_trace(go.Scatter(x=fnn, y=tnn, name="PyTorch NN  AUC=0.7571",
                             line=dict(color="#ffaa00", width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random baseline",
                             line=dict(color="#2a2a2a", width=1, dash="dash")))
    fig.update_layout(
        height=380,
        margin=dict(l=50, r=20, t=40, b=50),
        title=dict(text="ROC Curve — 61,503 validation applicants",
                   font=dict(color="#ccc", size=12, family="Courier New")),
        xaxis=dict(title="False Positive Rate", gridcolor="#1e1e1e",
                   tickfont=dict(color="#bbb"), title_font=dict(color="#ccc", size=12)),
        yaxis=dict(title="True Positive Rate",  gridcolor="#1e1e1e",
                   tickfont=dict(color="#bbb"), title_font=dict(color="#ccc", size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888")),
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
        font=dict(family="Courier New"),
    )
    st.plotly_chart(fig, use_container_width=True)

with t2:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_val), 4000, replace=False)
    sy, sxp = y_val[idx], xgb_vp[idx]
    fig = go.Figure()
    for mask, name, color in [
        (sy==0, "Non-default", "rgba(0,255,136,0.5)"),
        (sy==1, "Default",     "rgba(255,80,80,0.6)"),
    ]:
        fig.add_trace(go.Histogram(
            x=sxp[mask], name=name, opacity=0.75, nbinsx=50,
            marker_color=color, xbins=dict(start=0, end=1, size=0.02)))
    fig.add_vline(x=xp, line_dash="dot", line_color=pcolor(xp), line_width=2,
                  annotation_text=f"  This applicant {xp:.1%}",
                  annotation_font_color=pcolor(xp), annotation_font_size=10)
    fig.update_layout(
        barmode="overlay", height=360,
        margin=dict(l=50, r=20, t=40, b=50),
        title=dict(text="XGBoost Score Distribution (n=4,000 sample)",
                   font=dict(color="#ccc", size=12, family="Courier New")),
        xaxis=dict(title="Predicted Default Probability", gridcolor="#1e1e1e",
                   tickfont=dict(color="#bbb"), title_font=dict(color="#ccc", size=12)),
        yaxis=dict(title="Count", gridcolor="#1e1e1e",
                   tickfont=dict(color="#bbb"), title_font=dict(color="#ccc", size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888")),
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
        font=dict(family="Courier New"),
    )
    st.plotly_chart(fig, use_container_width=True)

with t3:
    imp   = xgb_model.feature_importances_
    idx2  = np.argsort(imp)[-20:]
    names = [feat_names[i] for i in idx2]
    vals  = imp[idx2]
    colors = ["#00ff88" if any(x in n for x in ["_TO_","_x_","MEAN","MIN","MAX","STD","BIN"])
              else "#2a4060" for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#bbb", size=10),
    ))
    fig.update_layout(
        height=540,
        margin=dict(l=200, r=80, t=40, b=20),
        title=dict(text="Top-20 Features by Gain  ·  green = engineered",
                   font=dict(color="#ccc", size=12, family="Courier New")),
        xaxis=dict(title="Importance (gain)", gridcolor="#1e1e1e",
                   tickfont=dict(color="#bbb"), title_font=dict(color="#ccc", size=12)),
        yaxis=dict(tickfont=dict(color="#ddd", size=10)),
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
        font=dict(family="Courier New"),
    )
    st.plotly_chart(fig, use_container_width=True)

with t4:
    df = pd.DataFrame({
        "Model":           ["XGBoost",    "PyTorch NN",  "Ensemble"],
        "Val AUC":         [0.7689,        0.7571,        round((0.7689+0.7571)/2,4)],
        "Gini":            [0.5379,        0.5142,        "—"],
        "KS Statistic":    [0.4028,        0.3814,        "—"],
        "Brier Score":     [0.1782,        0.1989,        "—"],
        "This applicant":  [f"{xp:.2%}",  f"{np_:.2%}",  f"{ens:.2%}"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig = go.Figure()
    for name, vals, color in [
        ("XGBoost",    [0.7689, 0.5379, 0.4028], "#00ff88"),
        ("PyTorch NN", [0.7571, 0.5142, 0.3814], "#ffaa00"),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=["AUC", "Gini", "KS Statistic"], y=vals,
            marker_color=color, opacity=0.85,
            text=[f"{v:.4f}" for v in vals], textposition="outside",
            textfont=dict(color="#ccc", size=11),
        ))
    fig.update_layout(
        barmode="group", height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text="Validation Metrics Comparison",
                   font=dict(color="#ccc", size=12, family="Courier New")),
        xaxis=dict(gridcolor="#1e1e1e", tickfont=dict(color="#ccc", size=11)),
        yaxis=dict(gridcolor="#1e1e1e", range=[0, 0.9],
                   tickfont=dict(color="#bbb"), title_font=dict(color="#ccc", size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#888")),
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
        font=dict(family="Courier New"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── SHAP tab ──────────────────────────────────────────────────────────────────
with t5:
    shap_path = ROOT / "reports" / "shap_summary.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP beeswarm — XGBoost top-20 features (n=2,000 sample)")
    else:
        st.info("Run `python -m src.evaluate` to generate the SHAP plot.")

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""<div style="font-family:'Courier New',monospace;font-size:.7rem;
font-weight:700;color:#aaa;text-align:center;border-top:1px solid #2a2a2a;
padding:16px 0 6px;margin-top:24px;letter-spacing:1px">
ML1 · Home Credit Default Risk · 307K applicants · 143 features ·
PyTorch 2.4 + XGBoost 3.2 · Fahad Amjad · github.com/fahadamjad009
</div>""", unsafe_allow_html=True)
