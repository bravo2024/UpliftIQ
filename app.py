from __future__ import annotations
"""UpliftIQ — causal uplift estimation dashboard (Bain & Co).

Compares X-learner, T-learner and a two-model causal-forest approximation on a
marketing-campaign A/B test and ranks audience cells by incremental response.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data import make_synthetic
from src.model import fit_causal_models
from src.evaluate import causal_report, qini_curves_for_plot
from src.core import treatment_effect_distribution_stats

st.set_page_config(page_title="UpliftIQ | Campaign Causal Uplift", layout="wide", page_icon="🧪")

BG, PANEL, FG, GRID = "#0b1220", "#16223a", "#e2e8f0", "#334155"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID,
    "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "text.color": FG,
    "grid.color": GRID, "legend.facecolor": PANEL, "legend.edgecolor": GRID,
})
COLORS = {"X-learner": "#10b981", "T-learner": "#f59e0b",
          "Causal-forest (2-model)": "#8b5cf6", "random": "#64748b"}


@st.cache_data(show_spinner="Generating campaign A/B data…")
def get_fit(n: int, seed: int, test_size: float):
    data = make_synthetic(n=n, seed=seed)
    fit = fit_causal_models(data, seed=seed, test_size=test_size)
    report = causal_report(fit)
    curves = qini_curves_for_plot(fit)
    return fit, report, curves, data


with st.sidebar:
    st.header("⚙️ A/B Experiment")
    n = st.slider("Campaign cells (sample size)", 2000, 40000, 12000, 1000)
    seed = st.number_input("Random seed", 0, 999, 42)
    test_size = st.slider("Holdout fraction", 0.15, 0.40, 0.25, 0.05)
    st.caption("Bain & Co · Causal ML / Uplift")

st.title("🧪 UpliftIQ — Incremental Response Estimation")
st.markdown("Estimate **CATE** with an **X-learner**, T-learner and two-model causal forest, "
            "then target the audience cells most responsive to the campaign variant.")

fit, report, curves, data = get_fit(int(n), int(seed), float(test_size))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Campaign cells", f"{data['n_samples']:,}")
c2.metric("Conversion (control+treat)", f"{data['positive_rate']:.2%}")
c3.metric("Treatment share", f"{data['treatment_rate']:.1%}")
c4.metric("Observed ATE", f"{data['ate']:.4f}")

tab_q, tab_t, tab_d = st.tabs(["📊 Qini curves", "📋 Estimator metrics", "🪜 Decile lift"])

with tab_q:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    frac = np.linspace(0, 1, len(curves["random"]))
    for name in ("random", "X-learner", "T-learner", "Causal-forest (2-model)"):
        ax.plot(frac, curves[name], label=name, color=COLORS[name],
                lw=2 if name != "random" else 1.5, ls="--" if name == "random" else "-")
    ax.set_xlabel("Fraction targeted (sorted by predicted CATE)")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.set_title("Qini curves — incremental response vs. targeting depth")
    ax.legend(); ax.grid(alpha=0.25)
    st.pyplot(fig)

with tab_t:
    rows = []
    for name, m in report["per_estimator"].items():
        td = m["te_distribution"]
        rows.append({
            "Estimator": name,
            "normalized_qini": round(m["normalized_qini"], 6),
            "spearman_vs_tau": round(m["spearman_vs_true_tau"], 4),
            "CATE_mean": round(td["mean"], 4),
            "CATE_std": round(td["std"], 4),
            "share_positive": f"{td['share_positive']:.1%}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Estimator"), use_container_width=True)
    st.caption("normalized_qini > 0 ⇒ beats random targeting. spearman_vs_tau measures "
               "CATE-ranking fidelity against the oracle effect.")

with tab_d:
    est_name = st.selectbox("Estimator", list(report["per_estimator"].keys()), index=0)
    dec = report["per_estimator"][est_name]["decile_lift"]
    ddf = pd.DataFrame(dec)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(ddf["decile"], ddf["uplift"],
                  color=[COLORS[est_name] if v > 0 else "#f43f5e" for v in ddf["uplift"]])
    ax.axhline(0, color=COLORS["random"], lw=1)
    ax.set_xlabel("Decile (1 = highest predicted CATE)")
    ax.set_ylabel("Uplift (treated − control)")
    ax.set_title(f"Decile uplift — {est_name}")
    ax.set_xticks(ddf["decile"]); ax.grid(alpha=0.25, axis="y")
    st.pyplot(fig)
    st.dataframe(ddf.set_index("decile"), use_container_width=True)

st.markdown("---")
st.markdown("**Method:** Potential outcomes · X-learner (Künzel 2019) · T-learner · "
            "two-model causal forest (Wager & Athey 2018 spirit) · Qini & normalized Qini.")
