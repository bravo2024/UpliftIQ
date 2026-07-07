from __future__ import annotations
"""Evaluation + reporting for UpliftIQ causal estimators."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from src.core import (
    qini_curve, normalized_qini, uplift_decile_lift,
    treatment_effect_distribution_stats, average_treatment_effect,
)


def causal_report(fit_result: dict) -> dict:
    yte, wte = fit_result["y_test"], fit_result["treatment_test"]
    ttau = fit_result["true_tau_test"]
    per_estimator = {}
    for name, cate in fit_result["cate_test"].items():
        per_estimator[name] = {
            "normalized_qini": normalized_qini(yte, wte, cate),
            "qini_area": float(normalized_qini(yte, wte, cate) * max(len(yte), 1)),
            "te_distribution": treatment_effect_distribution_stats(cate),
            "decile_lift": uplift_decile_lift(yte, wte, cate),
        }
        if ttau is not None:  # only synthetic data has an oracle effect
            rho = spearmanr(cate, ttau).correlation if len(ttau) > 2 else 0.0
            per_estimator[name]["spearman_vs_true_tau"] = float(rho) if np.isfinite(rho) else 0.0
    return {"per_estimator": per_estimator, "ate_observed": average_treatment_effect(yte, wte)}


def qini_curves_for_plot(fit_result: dict, n_points: int = 100) -> dict:
    yte, wte = fit_result["y_test"], fit_result["treatment_test"]
    out = {}
    for name, cate in fit_result["cate_test"].items():
        _, q = qini_curve(yte, wte, cate, n_points=n_points)
        out[name] = q
    # qini_curve prepends the origin, so match its length exactly
    n = len(next(iter(out.values()))) if out else n_points
    out["random"] = np.linspace(0, _total(yte, wte), n)
    return out


def _total(y, w):
    y, w = np.asarray(y, float), np.asarray(w, float)
    nt, nc = w.sum(), (1 - w).sum()
    ratio = nt / nc if nc > 0 else 0.0
    return float((y * w).sum() - (y * (1 - w)).sum() * ratio)


def _to_native(o):
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def save_metrics(metrics, path: str = "models/metrics.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_native(metrics), f, indent=2)


def print_report(report: dict) -> None:
    for name, m in report["per_estimator"].items():
        print(f"\n{'=' * 54}\n  {name}\n{'=' * 54}")
        print(f"  normalized_qini          : {m['normalized_qini']:.6f}")
        if "spearman_vs_true_tau" in m:
            print(f"  spearman vs true tau     : {m['spearman_vs_true_tau']:.4f}")
        td = m["te_distribution"]
        print(f"  CATE mean/std            : {td['mean']:.4f} / {td['std']:.4f}")
        print(f"  CATE [min, q25, q50, q75, max]: "
              f"[{td['min']:.4f}, {td['q25']:.4f}, {td['q50']:.4f}, {td['q75']:.4f}, {td['max']:.4f}]")
        print(f"  share positive CATE      : {td['share_positive']:.2%}")
        print("  decile uplift (top->bottom): "
              + ", ".join(f"{r['uplift']:+.3f}" for r in m["decile_lift"]))
    print(f"\n  observed ATE (test)      : {report['ate_observed']:.4f}")
