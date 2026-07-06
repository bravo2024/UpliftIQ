from __future__ import annotations
"""Causal-uplift evaluation metrics for UpliftIQ (implemented from scratch).

Differs from a sibling project's metric set: here we expose the **normalized
Qini** (per-capita incremental responses above the random baseline), a
**treatment-effect distribution summary**, and **uplift decile lift**.
"""
import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _sorted(y, treatment, cate):
    y = np.asarray(y, float)
    w = np.asarray(treatment, float)
    u = np.asarray(cate, float)
    order = np.argsort(-u, kind="mergesort")
    return y[order], w[order], u[order]


def _subsample(frac, vals, n_points):
    if n_points and len(frac) > n_points:
        idx = np.linspace(0, len(frac) - 1, n_points).astype(int)
        return frac[idx], vals[idx]
    return frac, vals


def _qini_total(y, treatment) -> float:
    y, w = np.asarray(y, float), np.asarray(treatment, float)
    nt, nc = w.sum(), (1 - w).sum()
    ratio = nt / nc if nc > 0 else 0.0
    return float((y * w).sum() - (y * (1 - w)).sum() * ratio)


def qini_curve(y, treatment, cate, n_points: int = 100):
    """Cumulative incremental-response curve vs. fraction targeted (origin prepended)."""
    y, w, _ = _sorted(y, treatment, cate)
    n = len(y)
    cum_t = np.cumsum(w)
    cum_c = np.cumsum(1 - w)
    cum_yt = np.cumsum(y * w)
    cum_yc = np.cumsum(y * (1 - w))
    ratio = np.divide(cum_t, cum_c, out=np.zeros_like(cum_t), where=cum_c > 0)
    qini = cum_yt - cum_yc * ratio
    frac = np.arange(1, n + 1) / n
    frac, qini = _subsample(frac, qini, n_points)
    return np.concatenate([[0.0], frac]), np.concatenate([[0.0], qini])


def _qini_area(y, treatment, cate, n_points: int = 100) -> float:
    frac, qini = qini_curve(y, treatment, cate, n_points)
    baseline = frac * _qini_total(y, treatment)
    return float(_trapezoid(qini - baseline, frac))


def normalized_qini(y, treatment, cate, n_points: int = 100) -> float:
    """Per-capita incremental responses above random targeting (≈[-1,1], >0 is skilful)."""
    n = max(len(y), 1)
    return _qini_area(y, treatment, cate, n_points) / n


def uplift_decile_lift(y, treatment, cate) -> list[dict]:
    """Per-decile treated-minus-control conversion gap, sorted by predicted CATE desc."""
    y, w, _ = _sorted(y, treatment, cate)
    n = len(y)
    rows = []
    for d, idx in enumerate(np.array_split(np.arange(n), 10), start=1):
        yd, wd = y[idx], w[idx]
        nt, nc = wd.sum(), (1 - wd).sum()
        rt = (yd * wd).sum() / nt if nt > 0 else 0.0
        rc = (yd * (1 - wd)).sum() / nc if nc > 0 else 0.0
        rows.append({
            "decile": d, "n": int(len(idx)),
            "uplift": float(rt - rc),
            "treated_rate": float(rt), "control_rate": float(rc),
        })
    return rows


def treatment_effect_distribution_stats(cate) -> dict:
    c = np.asarray(cate, float)
    return {
        "mean": float(c.mean()), "std": float(c.std()),
        "min": float(c.min()), "max": float(c.max()),
        "q25": float(np.percentile(c, 25)), "q50": float(np.percentile(c, 50)),
        "q75": float(np.percentile(c, 75)), "share_positive": float((c > 0).mean()),
    }


def average_treatment_effect(y, treatment) -> float:
    y, w = np.asarray(y, float), np.asarray(treatment, float)
    if w.sum() == 0 or (1 - w).sum() == 0:
        return 0.0
    return float(y[w == 1].mean() - y[w == 0].mean())
