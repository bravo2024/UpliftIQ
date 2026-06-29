"""data.py - Uplift modeling data loading and preparation.

Supports:
1. Hillstrom Email Marketing Dataset (the gold standard for uplift modeling)
2. Synthetic uplift data with known ground truth

Mathematical foundations:
- Potential outcomes framework: Y_i(1), Y_i(0) = outcomes under treatment/control
- Individual Treatment Effect (ITE): τ_i = Y_i(1) - Y_i(0) (fundamentally unobservable!)
- Average Treatment Effect (ATE): τ = E[Y(1) - Y(0)] = E[Y|T=1] - E[Y|T=0]
- Conditional Average Treatment Effect (CATE): τ(x) = E[Y(1) - Y(0) | X=x]

Key insight: We can only observe ONE potential outcome per unit (fundamental problem of causal inference).
Randomized experiments ensure P(T|X) = P(T), making ATE identifiable.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def load_hillstrom(csv_path: Optional[str] = None) -> Dict:
    """Load the Hillstrom Email Marketing Dataset.

    This is a randomized controlled experiment:
    - 64,000 customers
    - 3 treatment arms: Control, Men's Email, Women's Email
    - Treatment was randomly assigned (no confounding)
    - Outcomes: visit, conversion, spend

    The dataset is the gold standard for uplift modeling because:
    1. True randomized experiment (ATE is identifiable)
    2. Multiple outcome types (binary + continuous)
    3. Real business context (email marketing targeting)
    """
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "data" / "raw" / "hillstrom.csv"
    else:
        csv_path = Path(csv_path)

    try:
        df = pd.read_csv(csv_path)
        return _process_hillstrom(df)
    except FileNotFoundError:
        return _download_hillstrom(csv_path)


def _download_hillstrom(csv_path: Path) -> Dict:
    """Download Hillstrom dataset from public URL if not found locally."""
    import urllib.request

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    url = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"

    try:
        urllib.request.urlretrieve(url, str(csv_path))
        df = pd.read_csv(csv_path)
        return _process_hillstrom(df)
    except Exception:
        return make_synthetic_uplift(n=32000)


def _process_hillstrom(df: pd.DataFrame) -> Dict:
    """Process raw Hillstrom data."""
    df = df.copy()

    # Treatment encoding: 0=control, 1=men's email, 2=women's email
    if "treatment" in df.columns:
        treatment = df["treatment"].values
    else:
        treatment = np.zeros(len(df), dtype=int)

    # Features
    feature_cols = []
    for col in ["recency", "history", "mens", "womens", "newbie"]:
        if col in df.columns:
            feature_cols.append(col)

    # Encode categorical features
    if "history_segment" in df.columns:
        for seg in df["history_segment"].unique():
            df[f"seg_{seg}"] = (df["history_segment"] == seg).astype(int)
            feature_cols.append(f"seg_{seg}")

    if "zip_code" in df.columns:
        for zc in df["zip_code"].unique():
            df[f"zip_{zc}"] = (df["zip_code"] == zc).astype(int)
            feature_cols.append(f"zip_{zc}")

    if "channel" in df.columns:
        for ch in df["channel"].unique():
            df[f"ch_{ch}"] = (df["channel"] == ch).astype(int)
            feature_cols.append(f"ch_{ch}")

    # Target: conversion (binary)
    target = "conversion" if "conversion" in df.columns else "visit"
    y = df[target].values.astype(np.int64)

    X = df[feature_cols].values.astype(np.float64)

    return {
        "df": df,
        "X": X,
        "y": y,
        "treatment": treatment.astype(np.int64),
        "features": feature_cols,
        "n_samples": len(y),
        "treatment_names": ["Control", "Men's Email", "Women's Email"],
        "n_treatments": 3,
        "outcome_name": target,
    }


def make_synthetic_uplift(n: int = 32000, seed: int = 42) -> Dict:
    """Generate synthetic uplift data with known ground truth.

    Data generating process:
    - X ~ N(0, I_d) with d=10 features
    - Treatment T ~ Uniform({0, 1, 2}) (3 arms)
    - True CATE: τ(x) = β^T x + interaction terms
    - Outcome: y = μ(x) + τ(x) · 1(T=1) + noise

    This allows us to evaluate uplift models against known ground truth.
    """
    rng = np.random.default_rng(seed)
    d = 10
    X = rng.normal(size=(n, d))

    # True treatment effect function
    beta_treatment = rng.normal(0, 0.3, d)
    tau = X @ beta_treatment + 0.5 * X[:, 0] * X[:, 1]  # linear + interaction

    # Treatment assignment (randomized)
    treatment = rng.choice(3, size=n)

    # Outcome: base rate + treatment effect + noise
    mu = 0.1 + 0.05 * X[:, 0]  # base propensity
    noise = rng.normal(0, 0.1, n)
    y_continuous = mu + tau * (treatment == 1).astype(float) + noise
    y = (y_continuous > np.median(y_continuous)).astype(int)

    features = [f"feat_{i}" for i in range(d)]

    return {
        "df": None,
        "X": X,
        "y": y,
        "treatment": treatment,
        "features": features,
        "n_samples": n,
        "treatment_names": ["Control", "Treatment A", "Treatment B"],
        "n_treatments": 3,
        "outcome_name": "conversion",
        "true_cate": tau,
    }


def prepare_binary_uplift(data: Dict, treatment_col: int = 1) -> Dict:
    """Prepare binary treatment/control data for uplift modeling.

    Converts multi-arm experiment to binary: treatment_col vs control (arm 0).
    Only uses samples from these two arms.
    """
    mask = np.isin(data["treatment"], [0, treatment_col])
    treatment_binary = (data["treatment"][mask] == treatment_col).astype(int)

    return {
        "X": data["X"][mask],
        "y": data["y"][mask],
        "treatment": treatment_binary if isinstance(treatment_binary, np.ndarray) else treatment_binary.values,
        "features": data["features"],
        "n_samples": int(mask.sum()),
        "treatment_names": [data["treatment_names"][0], data["treatment_names"][treatment_col]],
    }
