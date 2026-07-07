# UpliftIQ — Campaign Causal Uplift

Estimates the **incremental (causal) lift** of a marketing campaign per customer and
decides *who is worth targeting*. This is causal inference, not classification: the
question is not "who will convert" but "whose behaviour the campaign actually changes."

## Data

The primary dataset is the real **Hillstrom e-mail experiment** (64,000 customers,
randomized e-mail vs holdout — the standard public uplift benchmark). It downloads
once and is cached under `data/`. Treatment is "received an e-mail," outcome is a
site visit within two weeks.

Measured on the held-out quarter: **observed ATE ≈ +5.7pp visit rate**, mean
predicted CATE ≈ 0.06 across all three estimators, normalized Qini ≈ 0.0015–0.0016
(positive ⇒ beats random targeting; small because e-mail lifts nearly everyone, so
ranking who to target is genuinely hard on this dataset — an honest result, not a
model failure).

A synthetic campaign A/B generator with a *known* heterogeneous CATE is kept
alongside (`--synthetic` / sidebar toggle), because only simulated data lets you
check estimators against the true effect (Spearman vs oracle tau).

## Methodology

### Estimators
- **X-learner** (Künzel et al. 2019) — fit `mu0`, `mu1`; impute `D1 = Y − mu0(X)`
  for treated and `D0 = mu1(X) − Y` for control; regress each on `X`; combine
  `tau(x) = e·tau_control(x) + (1−e)·tau_treated(x)` with propensity `e`.
- **T-learner** — `mu1(x) − mu0(x)` with gradient-boosted regressors.
- **Two-model causal forest** — a random-forest analogue of the X-learner
  (honest ensembles on the imputed effects, in the spirit of Wager & Athey 2018)
  that also exposes a per-tree CATE variance estimate for uncertainty.

### Metrics (from scratch)
- **Qini curve** — cumulative incremental responses vs. fraction targeted.
- **Normalized Qini** — per-capita incremental responses above the random baseline
  (≈[−1, 1], > 0 beats random).
- **Uplift decile lift** — per-decile treated−control conversion gap.
- **Treatment-effect distribution stats** — mean/std/quantiles/share-positive of
  predicted CATE.

## Project layout
```
src/data.py      Hillstrom loader (cached download) + synthetic A/B generator
src/model.py     X-learner, T-learner, TwoModelCausalForest (ColumnTransformer encoding)
src/core.py      qini_curve, normalized_qini, uplift_decile_lift, TE distribution stats
src/evaluate.py  held-out causal report + JSON metrics
tests/test_smoke.py  domain smoke tests (CATE recovery, metric shapes)
app.py           Streamlit dashboard
```

## Run
```sh
pip install -r requirements.txt
python train.py              # Hillstrom (downloads ~5 MB once)
python train.py --synthetic  # oracle-CATE simulation instead
pytest -q
streamlit run app.py
```

## References
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). *Metalearners for
  estimating heterogeneous treatment effects*. PNAS.
- Wager, S., & Athey, S. (2018). *Estimation and inference of heterogeneous
  treatment effects using random forests*. JASA.
- Radcliffe, N. J., & Surry, P. (2011). *Real uplift* (Qini curve).
