from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.stats import spearmanr

from src.data import make_synthetic
from src.model import fit_causal_models, predict_cate
from src.core import (
    qini_curve, normalized_qini, uplift_decile_lift,
    treatment_effect_distribution_stats,
)


def test_data_uses_campaign_features_with_heterogeneous_tau():
    d = make_synthetic(800, seed=0)
    # campaign-level features (NOT customer-tenure style)
    assert "campaign_segment" in d["df"].columns
    assert "creative_personalization_score" in d["df"].columns
    assert 0.40 < d["treatment_rate"] < 0.60
    assert d["true_tau"].std() > 1e-3
    assert 0.0 < d["positive_rate"] < 1.0


def test_estimators_predict_finite_cate():
    d = make_synthetic(900, seed=1)
    fit = fit_causal_models(d, seed=1)
    assert set(fit["cate_test"]) == {"X-learner", "T-learner", "Causal-forest (2-model)"}
    for name, cate in fit["cate_test"].items():
        assert cate.shape[0] == fit["n_test"]
        assert np.all(np.isfinite(cate))
    # causal-forest per-tree variance (GRF-style) works
    cf = fit["estimators"]["Causal-forest (2-model)"]
    var = cf.cate_variance(fit["X_test"].iloc[:20])
    assert var.shape == (20,) and np.all(np.isfinite(var))


def test_metrics_run_and_well_formed():
    d = make_synthetic(900, seed=2)
    fit = fit_causal_models(d, seed=2)
    yte, wte = fit["y_test"], fit["treatment_test"]
    cate = fit["cate_test"]["X-learner"]
    nq = normalized_qini(yte, wte, cate)
    f, q = qini_curve(yte, wte, cate)
    dec = uplift_decile_lift(yte, wte, cate)
    stats = treatment_effect_distribution_stats(cate)
    assert np.isfinite(nq)
    assert f.shape == q.shape and f[0] == 0.0
    assert len(dec) == 10 and all("uplift" in r for r in dec)
    assert all(k in stats for k in ("mean", "std", "share_positive", "q25", "q75"))


def test_predicted_cate_correlates_with_true_tau():
    d = make_synthetic(3000, seed=3)
    fit = fit_causal_models(d, seed=3)
    best = -1.0
    for name in fit["cate_test"]:
        rho = spearmanr(fit["cate_test"][name], fit["true_tau_test"]).correlation
        best = max(best, 0.0 if not np.isfinite(rho) else rho)
    assert best > 0.0, "at least one estimator must recover CATE signal"


def test_predict_cate_on_raw_rows():
    d = make_synthetic(700, seed=4)
    fit = fit_causal_models(d, seed=4)
    cate = predict_cate(fit["estimators"]["T-learner"], d["X"].iloc[:5])
    assert cate.shape == (5,) and np.all(np.isfinite(cate))
