"""visualizations.py - Uplift modeling visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#0e1117",
        "axes.edgecolor": "#333",
        "axes.labelcolor": "#fafafa",
        "text.color": "#fafafa",
        "xtick.color": "#aaa",
        "ytick.color": "#aaa",
        "grid.color": "#333",
        "grid.alpha": 0.4,
        "font.size": 10,
    })


def plot_qini_curves(qini_data_dict: Dict) -> plt.Figure:
    """Overlay Qini curves for multiple models."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#22d3ee", "#a78bfa", "#f97316", "#f43f5e"]

    for i, (name, data) in enumerate(qini_data_dict.items()):
        ax.plot(data["fractions"], data["uplift_curve"], linewidth=2,
                label=f'{name} (Qini={data["qini"]:.3f})',
                color=colors[i % len(colors)])

    # Random targeting line
    for name, data in qini_data_dict.items():
        ax.plot(data["fractions"], data["random_curve"], "--", color="#666",
                linewidth=1, alpha=0.5)
        break

    ax.set_xlabel("Population Fraction (targeted)")
    ax.set_ylabel("Uplift")
    ax.set_title("Qini Curves", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--")
    ax.set_xlim([0, 1.02])
    fig.tight_layout()
    return fig


def plot_uplift_curve(fractions: np.ndarray, uplift_curve: np.ndarray,
                      random_curve: np.ndarray, model_name: str = "") -> plt.Figure:
    """Single model uplift curve with shaded area."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(fractions, uplift_curve, color="#22d3ee", linewidth=2, label=model_name)
    ax.plot(fractions, random_curve, "--", color="#666", linewidth=1, label="Random")
    ax.fill_between(fractions, uplift_curve, random_curve, alpha=0.2, color="#22d3ee")

    ax.set_xlabel("Population Fraction (targeted)")
    ax.set_ylabel("Cumulative Uplift")
    ax.set_title(f"Uplift Curve — {model_name}", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--")
    ax.set_xlim([0, 1.02])
    fig.tight_layout()
    return fig


def plot_ate_comparison(ate_dict: Dict) -> plt.Figure:
    """Bar chart comparing ATE estimates across treatment groups."""
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(ate_dict.keys())
    ates = [ate_dict[n]["ate"] for n in names]
    ci_lowers = [ate_dict[n]["ci_lower"] for n in names]
    ci_uppers = [ate_dict[n]["ci_upper"] for n in names]
    errors = [[a - cl for a, cl in zip(ates, ci_lowers)],
              [cu - a for a, cu in zip(ates, ci_uppers)]]

    colors = ["#22d3ee", "#a78bfa", "#f97316"]
    bars = ax.bar(names, ates, color=colors[:len(names)], width=0.5, edgecolor="#333",
                  yerr=errors, capsize=5)

    for bar, ate in zip(bars, ates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{ate:.4f}", ha="center", fontsize=10, fontweight="bold")

    ax.axhline(y=0, color="#666", linewidth=1, linestyle="--")
    ax.set_ylabel("Average Treatment Effect (ATE)")
    ax.set_title("ATE Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--")
    fig.tight_layout()
    return fig


def plot_segment_analysis(segments: Dict) -> plt.Figure:
    """Bar chart of uplift by segment with actual vs predicted."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(segments.keys())
    actual = [segments[n]["actual_uplift"] for n in names]
    predicted = [segments[n]["avg_predicted_uplift"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, actual, width, label="Actual Uplift", color="#22d3ee", edgecolor="#333")
    bars2 = ax.bar(x + width / 2, predicted, width, label="Predicted Uplift", color="#a78bfa", edgecolor="#333")

    ax.set_xlabel("Predicted Uplift Segment")
    ax.set_ylabel("Treatment Effect")
    ax.set_title("Actual vs Predicted Uplift by Segment", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.axhline(y=0, color="#666", linewidth=1, linestyle="--")
    ax.grid(axis="y", linestyle="--")
    fig.tight_layout()
    return fig


def plot_uplift_distribution(uplift_scores: np.ndarray) -> plt.Figure:
    """Histogram of predicted uplift scores."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(uplift_scores, bins=50, color="#a78bfa", edgecolor="#333", alpha=0.8)
    ax.axvline(x=0, color="#f43f5e", linewidth=2, linestyle="--", label="τ = 0")
    ax.axvline(x=uplift_scores.mean(), color="#22d3ee", linewidth=2,
               linestyle="-", label=f"Mean = {uplift_scores.mean():.4f}")
    ax.set_xlabel("Predicted Uplift (τ)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predicted Treatment Effects", fontsize=12,
                 fontweight="bold", pad=10)
    ax.legend()
    ax.grid(True, linestyle="--")
    fig.tight_layout()
    return fig


def plot_persuasion_pie(categories: Dict) -> plt.Figure:
    """Pie chart of persuasion categories."""
    _style()
    fig, ax = plt.subplots(figsize=(6, 5))
    names = list(categories.keys())
    sizes = [categories[n]["count"] for n in names]
    colors = ["#22d3ee", "#22c55e", "#666", "#f43f5e"]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=names, colors=colors, autopct="%1.1f%%",
        pctdistance=0.85, startangle=90, textprops={"fontsize": 10}
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")

    ax.set_title("Customer Categories", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_treatment_balance(treatment: np.ndarray, features: np.ndarray,
                           feature_names: list, top_n: int = 6) -> plt.Figure:
    """Check treatment balance across feature distributions.

    In a randomized experiment, feature distributions should be similar
    across treatment and control groups (balance check).
    """
    _style()
    n = min(top_n, len(feature_names))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        treat_vals = features[treatment == 1, i]
        ctrl_vals = features[treatment == 0, i]

        ax.hist(ctrl_vals, bins=30, alpha=0.5, color="#22d3ee", label="Control", density=True)
        ax.hist(treat_vals, bins=30, alpha=0.5, color="#f97316", label="Treatment", density=True)
        ax.set_title(feature_names[i], fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--")

    fig.suptitle("Treatment Balance Check (should overlap)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
