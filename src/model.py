from __future__ import annotations
"""Causal effect estimators for UpliftIQ.

Three structurally distinct estimators on potential-outcomes data:

  * **X-learner** (Künzel et al. 2019): estimate mu0/mu1, impute per-unit effects
    D1 = Y - mu0(X) (treated) and D0 = mu1(X) - Y (control), regress each on X,
    then combine  tau(x) = e*tau_control(x) + (1-e)*tau_treated(x)  with the
    propensity e. The weighting down-weights the imputation whose nuisance model
    was fit on the smaller arm.
  * **T-learner**: mu1(x) - mu0(x) with tree regressors.
  * **TwoModelCausalForest**: a random-forest analogue of the X-learner (honest
    tree ensembles on the imputed effects, Wager & Athey 2018 spirit) that also
    exposes a per-tree CATE variance estimate.

Encoding uses sklearn's ColumnTransformer + OneHotEncoder (different from the
sibling SalesUplift project, which uses a pandas get_dummies encoder).
"""
import numpy as np

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:  # pragma: no cover
    _HAS_LGB = False

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def _gbm_reg(seed: int):
    if _HAS_LGB:
        return lgb.LGBMRegressor(
            n_estimators=250, max_depth=4, num_leaves=15, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
            reg_lambda=1.0, random_state=seed, n_jobs=1, verbose=-1,
        )
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        min_samples_leaf=20, random_state=seed,
    )


def _rf_reg(seed: int, n_estimators: int = 150):
    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=8, min_samples_leaf=10,
        max_features="sqrt", bootstrap=True, random_state=seed, n_jobs=1,
    )


def make_encoder(categorical_features, numerical_features) -> ColumnTransformer:
    """One-hot encode categoricals, pass numericals through (ignores unseen levels)."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical_features)),
        ("num", "passthrough", list(numerical_features)),
    ])


class _BaseCausal:
    """Shared estimator interface: fit(X, y, treatment) / predict_cate(X)."""

    name = "base"

    def __init__(self, categorical_features, numerical_features, seed=42):
        self.cat = list(categorical_features)
        self.num = list(numerical_features)
        self.seed = seed
        self.enc: ColumnTransformer | None = None
        self.propensity = 0.5

    def _encode_fit(self, X, y, treatment):
        self.enc = make_encoder(self.cat, self.num)
        self.enc.fit(X)
        Xe = self.enc.transform(X)
        t = np.asarray(treatment)
        ya = np.asarray(y, dtype=float)
        self.propensity = float(t.mean())
        return Xe, ya, t


class XLearner(_BaseCausal):
    name = "X-learner"

    def __init__(self, categorical_features, numerical_features, seed=42):
        super().__init__(categorical_features, numerical_features, seed)
        self.mu0 = _gbm_reg(seed)
        self.mu1 = _gbm_reg(seed + 1)
        self.tau_treated = _gbm_reg(seed + 2)   # regresses D1 = Y - mu0  (treated units)
        self.tau_control = _gbm_reg(seed + 3)   # regresses D0 = mu1 - Y  (control units)

    def fit(self, X, y, treatment):
        Xe, ya, t = self._encode_fit(X, y, treatment)
        self.mu0.fit(Xe[t == 0], ya[t == 0])
        self.mu1.fit(Xe[t == 1], ya[t == 1])
        self.tau_treated.fit(Xe[t == 1], ya[t == 1] - self.mu0.predict(Xe[t == 1]))
        self.tau_control.fit(Xe[t == 0], self.mu1.predict(Xe[t == 0]) - ya[t == 0])
        return self

    def predict_cate(self, X) -> np.ndarray:
        Xe = self.enc.transform(X)
        e = self.propensity
        return e * self.tau_control.predict(Xe) + (1.0 - e) * self.tau_treated.predict(Xe)


class TLearner(_BaseCausal):
    name = "T-learner"

    def __init__(self, categorical_features, numerical_features, seed=42):
        super().__init__(categorical_features, numerical_features, seed)
        self.mu0 = _gbm_reg(seed)
        self.mu1 = _gbm_reg(seed + 1)

    def fit(self, X, y, treatment):
        Xe, ya, t = self._encode_fit(X, y, treatment)
        self.mu0.fit(Xe[t == 0], ya[t == 0])
        self.mu1.fit(Xe[t == 1], ya[t == 1])
        return self

    def predict_cate(self, X) -> np.ndarray:
        Xe = self.enc.transform(X)
        return self.mu1.predict(Xe) - self.mu0.predict(Xe)


class TwoModelCausalForest(_BaseCausal):
    """Random-forest X-learner approximation with per-tree CATE variance."""
    name = "Causal-forest (2-model)"

    def __init__(self, categorical_features, numerical_features, seed=42, n_estimators=150):
        super().__init__(categorical_features, numerical_features, seed)
        self.mu0 = _rf_reg(seed, n_estimators)
        self.mu1 = _rf_reg(seed + 1, n_estimators)
        self.tau_treated = _rf_reg(seed + 2, n_estimators)
        self.tau_control = _rf_reg(seed + 3, n_estimators)

    def fit(self, X, y, treatment):
        Xe, ya, t = self._encode_fit(X, y, treatment)
        self.mu0.fit(Xe[t == 0], ya[t == 0])
        self.mu1.fit(Xe[t == 1], ya[t == 1])
        self.tau_treated.fit(Xe[t == 1], ya[t == 1] - self.mu0.predict(Xe[t == 1]))
        self.tau_control.fit(Xe[t == 0], self.mu1.predict(Xe[t == 0]) - ya[t == 0])
        return self

    def predict_cate(self, X) -> np.ndarray:
        Xe = self.enc.transform(X)
        e = self.propensity
        return e * self.tau_control.predict(Xe) + (1.0 - e) * self.tau_treated.predict(Xe)

    def cate_variance(self, X) -> np.ndarray:
        """Variance of CATE across individual trees (GRF-style uncertainty)."""
        Xe = self.enc.transform(X)
        e = self.propensity
        tt = np.asarray([tr.predict(Xe) for tr in self.tau_treated.estimators_])
        tc = np.asarray([tr.predict(Xe) for tr in self.tau_control.estimators_])
        combined = e * tc + (1.0 - e) * tt
        return combined.var(axis=0)


def fit_causal_models(data: dict, seed: int = 42, test_size: float = 0.25) -> dict:
    cat, num = data["categorical_features"], data["numerical_features"]
    X, y = data["X"], np.asarray(data["y"], dtype=float)
    w = np.asarray(data["treatment"], dtype=float)
    true_tau = np.asarray(data["true_tau"], dtype=float)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=w, random_state=seed)
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte, wtr, wte = y[tr], y[te], w[tr], w[te]

    estimators = {
        "X-learner": XLearner(cat, num, seed=seed),
        "T-learner": TLearner(cat, num, seed=seed),
        "Causal-forest (2-model)": TwoModelCausalForest(cat, num, seed=seed),
    }
    for est in estimators.values():
        est.fit(Xtr, ytr, wtr)

    cate_test = {name: predict_cate(est, Xte) for name, est in estimators.items()}
    return {
        "estimators": estimators,
        "X_train": Xtr, "X_test": Xte,
        "y_train": ytr, "y_test": yte,
        "treatment_train": wtr, "treatment_test": wte,
        "true_tau_test": true_tau[te],
        "cate_test": cate_test,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "categorical_features": list(cat),
        "numerical_features": list(num),
        "features": list(X.columns),
    }


def predict_cate(estimator, X) -> np.ndarray:
    return np.asarray(estimator.predict_cate(X), dtype=float)
