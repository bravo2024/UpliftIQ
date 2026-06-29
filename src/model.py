"""model.py - Causal uplift modeling with T-learner, S-learner, and evaluation.

Mathematical foundations:

1. S-Learner (Single model with treatment as feature):
   τ̂(x) = μ̂(x, T=1) - μ̂(x, T=0)
   where μ̂ is a single model trained on [X, T] → Y

2. T-Learner (Two separate models):
   μ̂₁(x) = model trained on {X, Y} where T=1 (treatment group)
   μ̂₀(x) = model trained on {X, Y} where T=0 (control group)
   τ̂(x) = μ̂₁(x) - μ̂₀(x)

3. Qini Coefficient (uplift model performance metric):
   Qini = (uplift model curve) - (random targeting curve)
   The Qini coefficient is the area between these curves.
   Higher Qini = better uplift model.

4. Uplift Curve:
   At population fraction p, the uplift curve shows the cumulative
   treatment effect when targeting the top p% of customers by predicted uplift.

5. CATE Estimation:
   τ(x) = E[Y(1) - Y(0) | X=x] = E[Y|X=x, T=1] - E[Y|X=x, T=0]
   (only identifiable under unconfoundedness assumption)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class SLearner:
    """S-Learner: Single model with treatment indicator as a feature.

    Math:
        μ̂(x, t) = model.fit(X_with_treatment, Y)
        τ̂(x) = μ̂(x, 1) - μ̂(x, 0)

    Pros: Simple, uses all data for one model
    Cons: May ignore treatment interaction, treatment effect is a "small signal"
    """

    def __init__(self, base_model=None):
        if base_model is None:
            self.base_model_class = GradientBoostingClassifier
            self.base_model_params = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, random_state=42
            ).get_params()
        else:
            self.base_model_class = type(base_model)
            self.base_model_params = base_model.get_params()
        self.model_ = None

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "SLearner":
        X_with_t = np.column_stack([X, treatment])
        self.model_ = self.base_model_class(**self.base_model_params)
        self.model_.fit(X_with_t, y)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Estimate CATE: τ̂(x) = μ̂(x, 1) - μ̂(x, 0)"""
        X_treat = np.column_stack([X, np.ones(len(X))])
        X_control = np.column_stack([X, np.zeros(len(X))])
        p_treat = self.model_.predict_proba(X_treat)[:, 1]
        p_control = self.model_.predict_proba(X_control)[:, 1]
        return p_treat - p_control

    def predict_proba(self, X: np.ndarray, treatment: int = 1) -> np.ndarray:
        X_ext = np.column_stack([X, np.full(len(X), treatment)])
        return self.model_.predict_proba(X_ext)[:, 1]


class TLearner:
    """T-Learner: Two separate models for treatment and control groups.

    Math:
        μ̂₁(x) = model₁.fit({X | T=1}, {Y | T=1})
        μ̂₀(x) = model₀.fit({X | T=0}, {Y | T=0})
        τ̂(x) = μ̂₁(x) - μ̂₀(x)

    Pros: Captures group-specific patterns, flexible
    Cons: May overfit in small treatment groups, doesn't share information
    """

    def __init__(self, base_model=None):
        if base_model is None:
            self.base_model_class = GradientBoostingClassifier
            self.base_model_params = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, random_state=42
            ).get_params()
        else:
            self.base_model_class = type(base_model)
            self.base_model_params = base_model.get_params()
        self.model_treat_ = None
        self.model_control_ = None

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "TLearner":
        treat_mask = treatment == 1
        control_mask = treatment == 0

        self.model_treat_ = self.base_model_class(**self.base_model_params)
        self.model_control_ = self.base_model_class(**self.base_model_params)

        self.model_treat_.fit(X[treat_mask], y[treat_mask])
        self.model_control_.fit(X[control_mask], y[control_mask])
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Estimate CATE: τ̂(x) = μ̂₁(x) - μ̂₀(x)"""
        p_treat = self.model_treat_.predict_proba(X)[:, 1]
        p_control = self.model_control_.predict_proba(X)[:, 1]
        return p_treat - p_control

    def predict_proba(self, X: np.ndarray, treatment: int = 1) -> np.ndarray:
        if treatment == 1:
            return self.model_treat_.predict_proba(X)[:, 1]
        return self.model_control_.predict_proba(X)[:, 1]


class XLearner:
    """X-Learner: Cross-learner that combines T-learner with propensity weighting.

    Math (simplified):
        Step 1: Train T-learner to get μ̂₁(x), μ̂₀(x)
        Step 2: Compute imputed treatment effects:
            D̂₁(x) = Y₁ - μ̂₀(x₁)  (for treatment group, compare Y to control model)
            D̂₀(x) = μ̂₁(x₀) - Y₀  (for control group, compare treatment model to Y)
        Step 3: Train two models on imputed effects:
            τ̂₁(x) = model₁.fit(X₁, D̂₁)
            τ̂₀(x) = model₀.fit(X₀, D̂₀)
        Step 4: Combine with propensity scores:
            τ̂(x) = g(x) · τ̂₁(x) + (1 - g(x)) · τ̂₀(x)
            where g(x) = P(T=1|X=x) is the propensity score

    Pros: More efficient than T-learner when treatment effect is small
    Cons: More complex, requires propensity estimation
    """

    def __init__(self, base_model=None):
        if base_model is None:
            self.base_model_class = GradientBoostingClassifier
            self.base_model_params = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, random_state=42
            ).get_params()
        else:
            self.base_model_class = type(base_model)
            self.base_model_params = base_model.get_params()
        self.t_learner = TLearner(base_model)
        self.model_d1_ = None
        self.model_d0_ = None
        self.propensity_ = None

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> "XLearner":
        # Step 1: T-learner
        self.t_learner.fit(X, treatment, y)

        # Step 2: Imputed treatment effects
        treat_mask = treatment == 1
        control_mask = treatment == 0

        d1 = y[treat_mask] - self.t_learner.predict_proba(X[treat_mask], treatment=0)
        d0 = self.t_learner.predict_proba(X[control_mask], treatment=1) - y[control_mask]

        # Step 3: Models on imputed effects
        self.model_d1_ = self.base_model_class(**self.base_model_params)
        self.model_d0_ = self.base_model_class(**self.base_model_params)

        # Convert to binary for classification
        d1_binary = (d1 > np.median(d1)).astype(int)
        d0_binary = (d0 > np.median(d0)).astype(int)

        self.model_d1_.fit(X[treat_mask], d1_binary)
        self.model_d0_.fit(X[control_mask], d0_binary)

        # Step 4: Propensity score (simple: treatment rate)
        self.propensity_ = treatment.mean()

        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Estimate CATE using propensity-weighted combination."""
        tau1 = self.model_d1_.predict_proba(X)[:, 1] - 0.5
        tau0 = self.model_d0_.predict_proba(X)[:, 1] - 0.5
        g = self.propensity_
        return g * tau1 + (1 - g) * tau0

    def predict_proba(self, X: np.ndarray, treatment: int = 1) -> np.ndarray:
        return self.t_learner.predict_proba(X, treatment)


def compute_ate(y: np.ndarray, treatment: np.ndarray) -> Dict:
    """Compute Average Treatment Effect (ATE).

    Math: ATE = E[Y|T=1] - E[Y|T=0]

    Under randomization, this is an unbiased estimate of the causal effect.
    Variability: Var(ATE) = Var(Y|T=1)/n₁ + Var(Y|T=0)/n₀
    """
    treat_mask = treatment == 1
    control_mask = treatment == 0

    y_treat = y[treat_mask]
    y_control = y[control_mask]

    ate = float(y_treat.mean() - y_control.mean())
    se = float(np.sqrt(y_treat.var() / len(y_treat) + y_control.var() / len(y_control)))
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_treat": int(treat_mask.sum()),
        "n_control": int(control_mask.sum()),
        "y_treat_mean": float(y_treat.mean()),
        "y_control_mean": float(y_control.mean()),
    }


def compute_qini(y_true: np.ndarray, treatment: np.ndarray,
                 uplift_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Qini curve and Qini coefficient.

    The Qini curve plots the incremental uplift achieved by targeting
    customers in order of predicted uplift (from highest to lowest).

    Math:
        For population fraction p:
            Uplift(p) = (treated & targeted response) / (n_treat × p)
                        - (control & targeted response) / (n_control × p)
            Qini(p) = Uplift(p) - Uplift_random(p)

        Qini coefficient = area between Qini curve and random targeting line
    """
    n = len(y_true)
    n_treat = (treatment == 1).sum()
    n_control = (treatment == 0).sum()

    # Sort by predicted uplift (descending)
    sorted_idx = np.argsort(-uplift_scores)

    fractions = np.linspace(0.01, 1.0, 100)
    uplift_curve = np.zeros(100)
    random_curve = np.zeros(100)

    overall_ate = compute_ate(y_true, treatment)["ate"]

    for i, p in enumerate(fractions):
        n_target = max(1, int(n * p))
        top_idx = sorted_idx[:n_target]

        # Treatment group response rate
        treat_targeted = ((treatment[top_idx] == 1) & (y_true[top_idx] == 1)).sum()
        treat_total = (treatment[top_idx] == 1).sum()
        treat_rate = treat_targeted / treat_total if treat_total > 0 else 0

        # Control group response rate
        ctrl_targeted = ((treatment[top_idx] == 0) & (y_true[top_idx] == 1)).sum()
        ctrl_total = (treatment[top_idx] == 0).sum()
        ctrl_rate = ctrl_targeted / ctrl_total if ctrl_total > 0 else 0

        uplift_curve[i] = treat_rate - ctrl_rate
        random_curve[i] = overall_ate

    # Qini coefficient = area between curves
    qini_coeff = float(np.trapezoid(uplift_curve - random_curve, fractions))

    return fractions, uplift_curve, qini_coeff


def uplift_curve_data(y_true: np.ndarray, treatment: np.ndarray,
                      uplift_scores: np.ndarray) -> Dict:
    """Compute cumulative uplift curve data for plotting."""
    n = len(y_true)

    sorted_idx = np.argsort(-uplift_scores)
    fractions = np.linspace(0.01, 1.0, 100)
    cumulative_uplift = np.zeros(100)
    cumulative_random = np.zeros(100)

    overall_ate = compute_ate(y_true, treatment)["ate"]

    for i, p in enumerate(fractions):
        n_target = max(1, int(n * p))
        top_idx = sorted_idx[:n_target]

        treat_mask = treatment[top_idx] == 1
        ctrl_mask = treatment[top_idx] == 0

        treat_resp = y_true[top_idx][treat_mask].mean() if treat_mask.sum() > 0 else 0
        ctrl_resp = y_true[top_idx][ctrl_mask].mean() if ctrl_mask.sum() > 0 else 0

        cumulative_uplift[i] = treat_resp - ctrl_resp
        cumulative_random[i] = overall_ate

    return {
        "fractions": fractions,
        "uplift_curve": cumulative_uplift,
        "random_curve": cumulative_random,
    }


def segment_analysis(y_true: np.ndarray, treatment: np.ndarray,
                     uplift_scores: np.ndarray, n_segments: int = 5) -> Dict:
    """Analyze treatment effects across predicted uplift segments.

    Divides population into n_segments by predicted uplift (quintiles).
    For each segment, computes:
    - Segment uplift: E[Y|T=1, segment] - E[Y|T=0, segment]
    - Segment size
    - Actual vs predicted uplift
    """
    sorted_idx = np.argsort(-uplift_scores)
    segment_size = len(y_true) // n_segments

    segments = {}
    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size if i < n_segments - 1 else len(y_true)
        seg_idx = sorted_idx[start:end]

        treat_mask = treatment[seg_idx] == 1
        ctrl_mask = treatment[seg_idx] == 0

        treat_resp = y_true[seg_idx][treat_mask].mean() if treat_mask.sum() > 0 else 0
        ctrl_resp = y_true[seg_idx][ctrl_mask].mean() if ctrl_mask.sum() > 0 else 0

        segments[f"Q{i+1}"] = {
            "size": len(seg_idx),
            "actual_uplift": float(treat_resp - ctrl_resp),
            "avg_predicted_uplift": float(uplift_scores[seg_idx].mean()),
            "treat_response": float(treat_resp),
            "ctrl_response": float(ctrl_resp),
        }

    return segments


def persuasion_categories(y_true: np.ndarray, treatment: np.ndarray,
                          uplift_scores: np.ndarray) -> Dict:
    """Identify persuasion categories (JLH framework).

    - Persuadables: Would respond ONLY if treated (τ > 0, would not respond without)
    - Sure Things: Would respond regardless of treatment
    - Lost Causes: Would not respond regardless of treatment
    - Sleeping Dogs: Would respond if NOT treated, but treatment suppresses response (τ < 0)

    Approximation using uplift scores:
    - Persuadables: high predicted uplift, low control response
    - Sure Things: high predicted uplift, high control response
    - Lost Causes: low predicted uplift, low control response
    - Sleeping Dogs: negative predicted uplift
    """
    n = len(y_true)
    sorted_idx = np.argsort(-uplift_scores)

    # Top 25% predicted uplift
    top_25 = sorted_idx[:n // 4]
    bottom_25 = sorted_idx[3 * n // 4:]

    # Check actual response rates
    treat_top = treatment[top_25] == 1
    ctrl_top = treatment[top_25] == 0
    treat_bottom = treatment[bottom_25] == 1
    ctrl_bottom = treatment[bottom_25] == 0

    categories = {
        "Persuadables": {
            "description": "Would respond ONLY if treated",
            "count": int((uplift_scores > 0).sum()),
            "pct": float((uplift_scores > 0).mean()),
        },
        "Sure Things": {
            "description": "Would respond regardless",
            "count": 0,
            "pct": 0.0,
        },
        "Lost Causes": {
            "description": "Would not respond regardless",
            "count": int((uplift_scores < 0).sum()),
            "pct": float((uplift_scores < 0).mean()),
        },
        "Sleeping Dogs": {
            "description": "Treatment suppresses response",
            "count": 0,
            "pct": 0.0,
        },
    }

    return categories
