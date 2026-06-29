# UpliftIQ

> Causal uplift modeling platform for treatment effect estimation and campaign optimisation.

Implements S-Learner, T-Learner, and X-Learner from scratch on the Hillstrom Email Marketing dataset (randomized experiment with 64,000 customers). Computes Average Treatment Effect (ATE), Qini coefficients for model evaluation, segment-level uplift analysis, and persuasion category identification (Persuadables, Sure Things, Lost Causes, Sleeping Dogs).

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

| Metric | Value |
|---|---|
| Qini (T-Learner) | 0.529 |
| Qini (S-Learner) | 0.348 |
| Qini (X-Learner) | 0.297 |
| ATE | −0.012 (p > 0.05, not significant) |
| Best model | T-Learner |

5,000 samples, 10 features. Hillstrom dataset (randomized 3-arm experiment).

## Features

| Component | What it does |
|---|---|
| **ATE Analysis** | Average Treatment Effect with confidence intervals, treatment/control balance check |
| **Uplift Models** | S-Learner, T-Learner, X-Learner comparison with Qini curves and uplift decile charts |
| **Segment Analysis** | Uplift by customer segment, persuasion category pie chart, targeting optimisation |
| **Campaign ROI** | Revenue simulation based on uplift model targeting, budget allocation |
| **Data Explorer** | Hillstrom dataset exploration, feature distributions, treatment group balance |

## Repo Structure

```
UpliftIQ/
  src/         data, model, evaluate, visualizations modules
  train.py     uplift model training (S/T/X-Learners)
  app.py       Streamlit dashboard (390 lines)
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Hillstrom Email Marketing Dataset: 64,000 customers, randomized 3-arm experiment (mens/womens email vs control). Features: recency, history, channel, segment, zip code, and purchase indicators.

## License

MIT
