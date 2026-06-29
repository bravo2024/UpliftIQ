"""app.py - UpliftIQ: Causal Uplift Modeling Dashboard.

A causal inference platform for treatment effect estimation with:
- T-Learner, S-Learner, X-Learner implementations from scratch
- Qini curves for model evaluation
- Persuadable/Sure Thing/Lost Cause identification
- Campaign ROI simulation
- Hillstrom Email Marketing Dataset (randomized experiment)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import streamlit as st
from src.data import load_hillstrom, prepare_binary_uplift
from src.model import (
    SLearner, TLearner, XLearner, compute_ate,
    compute_qini, uplift_curve_data, segment_analysis, persuasion_categories
)
from src.visualizations import (
    plot_qini_curves, plot_uplift_curve, plot_ate_comparison,
    plot_segment_analysis, plot_uplift_distribution, plot_persuasion_pie,
    plot_treatment_balance
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="UpliftIQ | Causal Uplift Modeling", layout="wide", page_icon="🧮")

# ---------------------------------------------------------------------------
# CSS + Hero
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 1rem;
    background: linear-gradient(135deg, #14532d 0%, #166534 55%, #22c55e 100%);
    color: white;
    margin-bottom: 1rem;
}
.hero h1 { margin-bottom: 0.2rem; }
.hero p  { margin-bottom: 0; opacity: 0.92; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🧮 UpliftIQ</h1>
    <p>Causal uplift modeling · T/S/X-Learners · Qini evaluation · Treatment effect estimation</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙ Configuration")
    data_source = st.radio("Dataset", ["Hillstrom (real)", "Synthetic (demo)"], index=0)
    treatment_arm = st.selectbox("Treatment Arm", ["Men's Email (vs Control)", "Women's Email (vs Control)"])
    treatment_idx = 1 if "Men" in treatment_arm else 2

    st.divider()
    st.subheader("Algorithms")
    selected_learners = st.multiselect(
        "Select learners to train",
        ["S-Learner", "T-Learner", "X-Learner"],
        default=["S-Learner", "T-Learner", "X-Learner"]
    )

    st.divider()
    st.subheader("Business Parameters")
    intervention_cost = st.number_input("Cost per customer ($)", 1, 100, 5, 1)
    revenue_per_conversion = st.number_input("Revenue per conversion ($)", 10, 500, 50, 10)
    campaign_budget = st.slider("Campaign budget (% of customers)", 10, 100, 30, 5)

    st.divider()
    st.caption("Built with NumPy · Streamlit")
    st.code("streamlit run app.py", language="bash")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data(source):
    if source == "Hillstrom (real)":
        try:
            return load_hillstrom()
        except Exception:
            return make_synthetic_uplift()
    from src.data import make_synthetic_uplift
    return make_synthetic_uplift()


data = load_data(data_source)
binary_data = prepare_binary_uplift(data, treatment_col=treatment_idx)


# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
ate_result = compute_ate(binary_data["y"], binary_data["treatment"])
cols = st.columns(6)
cols[0].metric("Samples", f"{binary_data['n_samples']:,}")
cols[1].metric("Features", len(binary_data["features"]))
cols[2].metric("ATE", f"{ate_result['ate']:.4f}")
cols[3].metric("95% CI", f"[{ate_result['ci_lower']:.4f}, {ate_result['ci_upper']:.4f}]")
cols[4].metric("Treatment Group", f"{ate_result['n_treat']:,}")
cols[5].metric("Control Group", f"{ate_result['n_control']:,}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_data, tab_model, tab_segments, tab_persuade, tab_campaign = st.tabs([
    "🔍 Data Explorer", "🧪 Model Lab", "📊 Segments",
    "🎯 Persuasion Analysis", "💰 Campaign Optimizer"
])


# ===== TAB 1: Data Explorer =====
with tab_data:
    st.subheader("Hillstrom Email Marketing Dataset")

    st.markdown("""
    **Randomized Controlled Experiment** — The gold standard for causal inference.

    - **64,000 customers** randomly assigned to 3 groups
    - **Treatment**: Email campaign (Men's or Women's merchandise)
    - **Outcomes**: Website visit, conversion, dollar spend
    - **Key insight**: Randomization ensures P(T|X) = P(T), making ATE identifiable

    Mathematical framework:
    - **Potential outcomes**: Y_i(1), Y_i(0) — what would happen under treatment/control
    - **Fundamental problem**: We only observe ONE outcome per unit
    - **ATE**: E[Y(1) - Y(0)] = E[Y|T=1] - E[Y|T=0] (identifiable under randomization)
    """)

    if data["df"] is not None:
        st.dataframe(data["df"].head(100), use_container_width=True)

    st.divider()
    st.markdown("**Treatment Balance Check**")
    st.markdown("In a randomized experiment, feature distributions should overlap across groups:")
    st.pyplot(plot_treatment_balance(
        binary_data["treatment"], binary_data["X"],
        binary_data["features"], top_n=6
    ))


# ===== TAB 2: Model Lab =====
with tab_model:
    st.subheader("Uplift Model Training")

    if not selected_learners:
        st.warning("Select at least one learner in the sidebar.")
    else:
        learner_descriptions = {
            "S-Learner": "**S-Learner**: Single model with treatment as feature\nτ̂(x) = μ̂(x, T=1) - μ̂(x, T=0)",
            "T-Learner": "**T-Learner**: Two separate models (treatment vs control)\nτ̂(x) = μ̂₁(x) - μ̂₀(x)",
            "X-Learner": "**X-Learner**: Cross-learner with propensity weighting\nτ̂(x) = g(x)·τ̂₁(x) + (1-g(x))·τ̂₀(x)",
        }
        desc_lines = "\n\n".join(learner_descriptions[l] for l in selected_learners if l in learner_descriptions)
        st.markdown(f"**{len(selected_learners)} learner(s)** for CATE estimation:\n\n{desc_lines}")

        if st.button("🚀 Train Selected Models", key="train"):
            trained_results = {}

            if "S-Learner" in selected_learners:
                with st.spinner("Training S-Learner..."):
                    s_learner = SLearner()
                    s_learner.fit(binary_data["X"], binary_data["treatment"], binary_data["y"])
                    cate_s = s_learner.predict_cate(binary_data["X"])
                    qini_s = compute_qini(binary_data["y"], binary_data["treatment"], cate_s)
                    uc_s = uplift_curve_data(binary_data["y"], binary_data["treatment"], cate_s)
                    uc_s["qini"] = qini_s[2]
                    trained_results["S-Learner"] = {"cate": cate_s, "qini": qini_s, "uplift_curve": uc_s, "model": s_learner}

            if "T-Learner" in selected_learners:
                with st.spinner("Training T-Learner..."):
                    t_learner = TLearner()
                    t_learner.fit(binary_data["X"], binary_data["treatment"], binary_data["y"])
                    cate_t = t_learner.predict_cate(binary_data["X"])
                    qini_t = compute_qini(binary_data["y"], binary_data["treatment"], cate_t)
                    uc_t = uplift_curve_data(binary_data["y"], binary_data["treatment"], cate_t)
                    uc_t["qini"] = qini_t[2]
                    trained_results["T-Learner"] = {"cate": cate_t, "qini": qini_t, "uplift_curve": uc_t, "model": t_learner}

            if "X-Learner" in selected_learners:
                with st.spinner("Training X-Learner..."):
                    x_learner = XLearner()
                    x_learner.fit(binary_data["X"], binary_data["treatment"], binary_data["y"])
                    cate_x = x_learner.predict_cate(binary_data["X"])
                    qini_x = compute_qini(binary_data["y"], binary_data["treatment"], cate_x)
                    uc_x = uplift_curve_data(binary_data["y"], binary_data["treatment"], cate_x)
                    uc_x["qini"] = qini_x[2]
                    trained_results["X-Learner"] = {"cate": cate_x, "qini": qini_x, "uplift_curve": uc_x, "model": x_learner}

            st.session_state.results = trained_results
            st.success(f"Trained {len(trained_results)} model(s)!")

    if st.session_state.results:
        results = st.session_state.results

        # Qini curves
        st.markdown("**Qini Curves** (higher = better)")
        st.pyplot(plot_qini_curves({k: v["uplift_curve"] for k, v in results.items()}))

        # Uplift distribution
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**S-Learner Uplift Distribution**")
            st.pyplot(plot_uplift_distribution(results["S-Learner"]["cate"]))
        with c2:
            st.markdown("**T-Learner Uplift Distribution**")
            st.pyplot(plot_uplift_distribution(results["T-Learner"]["cate"]))
        with c3:
            st.markdown("**X-Learner Uplift Distribution**")
            st.pyplot(plot_uplift_distribution(results["X-Learner"]["cate"]))


# ===== TAB 3: Segments =====
with tab_segments:
    st.subheader("Uplift Segments")

    if st.session_state.results:
        # Use best model (highest Qini)
        best_name = max(st.session_state.results,
                        key=lambda k: st.session_state.results[k]["uplift_curve"]["qini"])
        best_cate = st.session_state.results[best_name]["cate"]

        segments = segment_analysis(binary_data["y"], binary_data["treatment"], best_cate)

        st.pyplot(plot_segment_analysis(segments))

        st.markdown("**Segment Interpretation**")
        for seg_name, seg_data in segments.items():
            st.markdown(f"""
            **{seg_name}**: {seg_data['size']:,} customers
            - Actual uplift: {seg_data['actual_uplift']:.4f}
            - Predicted uplift: {seg_data['avg_predicted_uplift']:.4f}
            - Treatment response: {seg_data['treat_response']:.4f}
            - Control response: {seg_data['ctrl_response']:.4f}
            """)

        st.divider()
        st.markdown("""
        **Mathematical Interpretation:**

        Segments are ranked by predicted uplift (Q1 = highest predicted uplift).

        - **Q1** should have the highest actual uplift (these are the "persuadables")
        - **Q5** should have the lowest or negative uplift (these are "sure things" or "lost causes")

        The gap between predicted and actual uplift measures model calibration.
        """)
    else:
        st.info("Train models in the Model Lab tab first.")


# ===== TAB 4: Persuasion Analysis =====
with tab_persuade:
    st.subheader("Persuasion Categories")

    st.markdown("""
    The **JLH framework** categorizes customers based on their response to treatment:

    - **Persuadables**: Would respond ONLY if treated (τ > 0). These are the target!
    - **Sure Things**: Would respond regardless (high baseline, no uplift). Don't waste budget.
    - **Lost Causes**: Would not respond regardless (low baseline, no uplift). Don't waste budget.
    - **Sleeping Dogs**: Treatment suppresses response (τ < 0). Do NOT treat!
    """)

    if st.session_state.results:
        best_name = max(st.session_state.results,
                        key=lambda k: st.session_state.results[k]["uplift_curve"]["qini"])
        best_cate = st.session_state.results[best_name]["cate"]

        categories = persuasion_categories(binary_data["y"], binary_data["treatment"], best_cate)
        st.pyplot(plot_persuasion_pie(categories))

        st.markdown("**Category Details**")
        for cat_name, cat_data in categories.items():
            st.markdown(f"**{cat_name}**: {cat_data['description']} — {cat_data['count']:,} ({cat_data['pct']:.1%})")

        st.divider()
        st.markdown("""
        **Business Value:**

        Targeting only **persuadables** maximizes campaign ROI:
        - Don't waste budget on Sure Things (they'll convert anyway)
        - Don't waste budget on Lost Causes (they won't convert anyway)
        - NEVER treat Sleeping Dogs (treatment hurts!)

        The uplift model's job is to identify who is persuadable.
        """)
    else:
        st.info("Train models first.")


# ===== TAB 5: Campaign Optimizer =====
with tab_campaign:
    st.subheader("Campaign Budget Optimization")

    st.markdown(f"""
    **Business Parameters:**
    - Cost per email: ${intervention_cost}
    - Revenue per conversion: ${revenue_per_conversion}
    - Budget: {campaign_budget}% of customers ({int(binary_data['n_samples'] * campaign_budget / 100):,} customers)
    """)

    if st.session_state.results:
        best_name = max(st.session_state.results,
                        key=lambda k: st.session_state.results[k]["uplift_curve"]["qini"])
        best_cate = st.session_state.results[best_name]["cate"]

        n_target = int(binary_data["n_samples"] * campaign_budget / 100)
        sorted_idx = np.argsort(-best_cate)[:n_target]

        treat_mask = binary_data["treatment"][sorted_idx] == 1
        ctrl_mask = binary_data["treatment"][sorted_idx] == 0

        # Estimate incremental conversions
        treat_response = binary_data["y"][sorted_idx][treat_mask].mean() if treat_mask.sum() > 0 else 0
        ctrl_response = binary_data["y"][sorted_idx][ctrl_mask].mean() if ctrl_mask.sum() > 0 else 0
        incremental_rate = treat_response - ctrl_response

        estimated_incremental = int(incremental_rate * n_target)
        campaign_cost = n_target * intervention_cost
        campaign_revenue = estimated_incremental * revenue_per_conversion
        roi = (campaign_revenue - campaign_cost) / campaign_cost * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Targeted", f"{n_target:,}")
        c2.metric("Est. Incremental", f"{estimated_incremental:,}")
        c3.metric("Campaign Cost", f"${campaign_cost:,.0f}")
        c4.metric("ROI", f"{roi:.0f}%")

        st.divider()
        st.markdown(f"""
        **ROI Analysis:**

        By targeting the top {campaign_budget}% by predicted uplift:
        - **{estimated_incremental:,} incremental conversions** (vs random targeting)
        - **Campaign cost**: ${campaign_cost:,.0f} ({n_target:,} × ${intervention_cost})
        - **Incremental revenue**: ${campaign_revenue:,.0f}
        - **Net profit**: ${campaign_revenue - campaign_cost:,.0f}
        - **ROI**: {roi:.0f}%

        **Mathematical Formulation:**
        ```
        max  Σᵢ  τ̂(xᵢ) · zᵢ          (maximize incremental conversions)
        s.t.  Σᵢ  c · zᵢ  ≤  B        (budget constraint)
              Σᵢ  zᵢ  ≤  K            (capacity constraint)
              zᵢ ∈ {0, 1}              (binary targeting)
        ```

        The greedy solution (sort by τ̂(xᵢ) descending, take top-K) is optimal
        when the budget constraint is the binding constraint.
        """)
    else:
        st.info("Train models first to see campaign optimization.")


# ---------------------------------------------------------------------------
# Deploy notes
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Deployment & production notes"):
    st.markdown("""
    **UpliftIQ** — Production deployment:

    1. **A/B testing**: Deploy uplift model alongside existing targeting
    2. **Holdout**: Maintain a random holdout group for ongoing ATE estimation
    3. **Monitoring**: Track Qini coefficient over time (model decay)
    4. **Refresh**: Retrain monthly with new experiment data
    5. **Compliance**: Document causal assumptions (unconfoundedness)
    """)
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
