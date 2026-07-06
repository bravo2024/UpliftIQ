from __future__ import annotations
"""Synthetic marketing-campaign A/B test data for UpliftIQ.

Each row is a CAMPAIGN × AUDIENCE-CELL exposure (not a customer-tenure record).
A randomized treatment flag W (~50%) toggles whether the cell received the new
campaign variant. Outcomes follow the potential-outcomes model with a
HETEROGENEOUS CATE that depends on campaign/creative/audience features — e.g.
highly personalized creatives in low-competition markets produce large uplift,
while high-frequency (saturated) audiences show diminishing or negative returns.
"""
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "campaign_segment", "historical_channel_ctr", "ad_intensity",
    "market_competition_index", "audience_segment", "device_mix_share",
    "time_since_last_campaign_days", "offer_value",
    "creative_personalization_score", "reach_frequency", "region_index",
]
CATEGORICAL_FEATURES = ["campaign_segment", "audience_segment", "region_index"]
NUMERICAL_FEATURES = [c for c in FEATURE_NAMES if c not in CATEGORICAL_FEATURES]
TREATMENT_COL = "treatment"
TARGET_NAME = "conversion"

_AUD_MAP = {"new": 0.60, "active": 0.30, "dormant": 0.80, "at-risk": 0.90}
_SEG_MAP = {"acquisition": 0.00, "retention": 0.10, "winback": 0.60, "cross-sell": 0.30}
_REG_MAP = {"metro": 0.80, "tier1": 0.60, "tier2": 0.40, "rural": 0.20}


def make_synthetic(n: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    campaign_segment = rng.choice(["acquisition", "retention", "winback", "cross-sell"],
                                  n, p=[0.30, 0.30, 0.20, 0.20])
    historical_channel_ctr = rng.beta(2, 80, n).round(4)
    ad_intensity = rng.lognormal(8.0, 0.6, n).clip(500, 200_000).astype(int)
    market_competition = rng.beta(2, 3, n).round(3)
    audience_segment = rng.choice(["new", "active", "dormant", "at-risk"],
                                  n, p=[0.25, 0.35, 0.25, 0.15])
    device_mix = rng.beta(5, 3, n).round(3)
    time_since_last = rng.exponential(20, n).clip(0, 120).astype(int)
    offer_value = rng.lognormal(2.8, 0.5, n).clip(5, 120).round(2)
    creative_pers = rng.beta(3, 2, n).round(3)
    reach_freq = rng.gamma(2.0, 1.2, n).clip(1, 12).round(2)
    region = rng.choice(["metro", "tier1", "tier2", "rural"], n, p=[0.4, 0.3, 0.2, 0.1])

    w = rng.binomial(1, 0.5, n)

    # normalized covariates for the data-generating process
    hctr = historical_channel_ctr
    intensity = np.log(ad_intensity) / np.log(200_000)
    comp = market_competition
    aud = np.array([_AUD_MAP[a] for a in audience_segment], dtype=float)
    tsl = np.clip(time_since_last / 120.0, 0, 1)
    offer = np.log(offer_value) / np.log(120.0)
    cpers = creative_pers
    rf = np.clip(reach_freq / 12.0, 0, 1)
    seg = np.array([_SEG_MAP[s] for s in campaign_segment], dtype=float)
    reg = np.array([_REG_MAP[r] for r in region], dtype=float)

    # baseline (control) conversion probability
    logit0 = (-2.8 + 6.0 * hctr + 0.6 * intensity + 0.5 * aud + 0.4 * reg
              + 0.3 * cpers + rng.normal(0, 0.25, n))
    mu0 = 1.0 / (1.0 + np.exp(-logit0))

    # heterogeneous CATE: personalization & low competition help; saturation hurts
    tau = (0.02 + 0.10 * seg + 0.15 * cpers - 0.10 * comp + 0.10 * aud
           + 0.06 * offer - 0.04 * rf - 0.03 * tsl
           + rng.normal(0, 0.015, n))
    tau = np.clip(tau, -0.05, 0.40)
    mu1 = np.clip(mu0 + tau, 0.0, 1.0)

    y0 = rng.binomial(1, mu0).astype(float)
    y1 = rng.binomial(1, mu1).astype(float)
    y = np.where(w == 1, y1, y0).astype(float)

    df = pd.DataFrame({
        "campaign_segment": campaign_segment, "historical_channel_ctr": historical_channel_ctr,
        "ad_intensity": ad_intensity, "market_competition_index": market_competition,
        "audience_segment": audience_segment, "device_mix_share": device_mix,
        "time_since_last_campaign_days": time_since_last, "offer_value": offer_value,
        "creative_personalization_score": creative_pers, "reach_frequency": reach_freq,
        "region_index": region, TREATMENT_COL: w, TARGET_NAME: y, "true_tau": tau,
    })
    X = df[FEATURE_NAMES].copy()
    treated = w == 1
    ate = float(y[treated].mean() - y[~treated].mean()) if treated.any() and (~treated).any() else 0.0
    return {
        "X": X, "y": y, "treatment": w.astype(float), "true_tau": tau.astype(float),
        "df": df, "features": list(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numerical_features": list(NUMERICAL_FEATURES),
        "treatment_col": TREATMENT_COL, "target_name": TARGET_NAME,
        "n_samples": int(n), "positive_rate": float(y.mean()),
        "treatment_rate": float(w.mean()), "ate": ate,
    }
