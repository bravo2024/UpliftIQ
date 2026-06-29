"""train.py - build (synthetic) data, train, evaluate, persist. Runs with no downloads."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.data import make_synthetic_uplift
from src.model import SLearner, TLearner, XLearner, compute_ate, compute_qini
from src.evaluate import save_metrics
from src.persist import save_model


def main() -> None:
    data = make_synthetic_uplift(n=5000, seed=42)
    X, treatment, y = data["X"], data["treatment"], data["y"]

    ate = compute_ate(y, treatment)
    print(f"ATE = {ate['ate']:.4f} (CI: [{ate['ci_lower']:.4f}, {ate['ci_upper']:.4f}])")

    models = {
        "S_Learner": SLearner(),
        "T_Learner": TLearner(),
        "X_Learner": XLearner(),
    }
    results = {}
    for name, model in models.items():
        model.fit(X, treatment, y)
        cate = model.predict_cate(X)
        _, _, qini = compute_qini(y, treatment, cate)
        results[name] = {"qini": qini, "model": model, "cate": cate}
        print(f"  {name:12s} Qini = {qini:.4f}")

    best = max(results, key=lambda n: results[n]["qini"])
    save_model({
        "models": {n: results[n]["model"] for n in results},
        "cate_scores": {n: results[n]["cate"] for n in results},
        "best": best,
        "ate": ate,
    })
    save_metrics({
        "ate": ate,
        "qini": {n: results[n]["qini"] for n in results},
        "best_model": best,
        "n_samples": len(X),
        "n_features": X.shape[1],
    })
    print(f"\nBest model: {best} (Qini={results[best]['qini']:.4f})")
    print("Saved.")


if __name__ == "__main__":
    main()
