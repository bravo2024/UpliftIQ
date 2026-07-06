# UpliftIQ — Campaign Causal Uplift

Estimate the **incremental (causal) lift** of a marketing-campaign variant at the
audience-cell level and decide *which cells to target*. Like a sibling uplift
project, this is causal inference — not classification — but it uses a **different
data domain** (campaign/creative/audience features rather than customer tenure) and
a **different estimator family** (X-learner + two-model causal forest) with a
**different metric set** (normalized Qini, treatment-effect distribution stats,
uplift decile lift).

## Methodology

### Data model
Each row is a campaign × audience-cell exposure with a randomized treatment `W`
(~50%). Potential outcomes `y0 ~ Bernoulli(mu0(x))`, `y1 ~ Bernoulli(mu1(x))` with
`mu1 = clip(mu0 + tau(x), 0, 1)`. The CATE `tau(x)` is heterogeneous: personalized
creatives in low-competition markets lift strongly; high-frequency (saturated)
audiences show diminishing/negative returns.

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
src/data.py      synthetic campaign A/B data with heterogeneous CATE
src/model.py     X-learner, T-learner, TwoModelCausalForest (ColumnTransformer encoding)
src/core.py      qini_curve, normalized_qini, uplift_decile_lift, TE distribution stats
src/evaluate.py  held-out causal report + JSON metrics
tests/test_smoke.py  domain smoke tests (CATE recovery, metric shapes)
app.py           Streamlit dashboard
```

## Run
```sh
make install
make train
make test
streamlit run app.py
```

## References
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). *Metalearners for
  estimating heterogeneous treatment effects*. PNAS.
- Wager, S., & Athey, S. (2018). *Estimation and inference of heterogeneous
  treatment effects using random forests*. JASA.
- Radcliffe, N. J., & Surry, P. (2011). *Real uplift* (Qini curve).
