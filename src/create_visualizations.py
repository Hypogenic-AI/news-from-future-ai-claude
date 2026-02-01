"""
Create visualizations for the News from the Future experiment results.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory
figures_dir = Path("results/figures")
figures_dir.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load experiment results from JSON files."""
    with open("results/experiment_summary.json") as f:
        summary = json.load(f)
    with open("results/evaluation_results.json") as f:
        eval_results = json.load(f)
    with open("results/all_articles.json") as f:
        articles = json.load(f)
    return summary, eval_results, articles


def plot_mode_comparison(summary):
    """Create bar chart comparing generation modes."""
    mode_data = summary["mode_comparison"]

    modes = list(mode_data.keys())
    modes_display = ["Zero-Shot", "Probability\nConditioned", "Scenario\nPositive", "Scenario\nNegative"]

    metrics = ["avg_plausibility", "avg_authenticity", "avg_calibration", "avg_overall"]
    metric_names = ["Plausibility", "Authenticity", "Calibration", "Overall"]

    x = np.arange(len(modes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = []
        for mode in modes:
            values.append(mode_data[mode][metric])
        bars = ax.bar(x + i * width, values, width, label=name)

    ax.set_ylabel('Score (1-5)', fontsize=12)
    ax.set_xlabel('Generation Mode', fontsize=12)
    ax.set_title('Quality Metrics by Generation Mode', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(modes_display)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 5.5)

    # Add value labels
    for i, mode in enumerate(modes):
        overall = mode_data[mode]["avg_overall"]
        ax.annotate(f'{overall:.2f}',
                    xy=(i + 1.5 * width, mode_data[mode]["avg_overall"]),
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / "mode_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir}/mode_comparison.png")


def plot_calibration_analysis(summary):
    """Create visualization of calibration results."""
    calibration = summary["calibration"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Calibration score by probability range
    ranges = ["very_low", "high", "very_high"]
    range_labels = ["0-20%", "60-80%", "80-100%"]
    cal_scores = [calibration[r]["avg_calibration_score"] for r in ranges]
    counts = [calibration[r]["count"] for r in ranges]

    ax1 = axes[0]
    colors = sns.color_palette("Blues", n_colors=3)
    bars = ax1.bar(range_labels, cal_scores, color=colors)
    ax1.set_ylabel('Average Calibration Score (1-5)', fontsize=11)
    ax1.set_xlabel('Probability Range', fontsize=11)
    ax1.set_title('Calibration Score by Probability Range', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 5.5)

    for bar, count, score in zip(bars, counts, cal_scores):
        ax1.annotate(f'n={count}\n{score:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10)

    # Plot 2: Confidence ratio by probability range
    ax2 = axes[1]
    conf_ratios = [calibration[r]["avg_confidence_ratio"] for r in ranges]
    high_markers = [calibration[r]["avg_high_conf_markers"] for r in ranges]
    low_markers = [calibration[r]["avg_low_conf_markers"] for r in ranges]

    x = np.arange(len(ranges))
    width = 0.35

    bars1 = ax2.bar(x - width/2, high_markers, width, label='High Confidence Markers', color='#2ecc71')
    bars2 = ax2.bar(x + width/2, low_markers, width, label='Low Confidence Markers', color='#e74c3c')

    ax2.set_ylabel('Average Marker Count', fontsize=11)
    ax2.set_xlabel('Probability Range', fontsize=11)
    ax2.set_title('Confidence Language by Probability Range', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(range_labels)
    ax2.legend()

    # Add overall correlation annotation
    corr = calibration.get("overall_correlation", 0)
    fig.text(0.5, 0.02, f'Overall Probability-Confidence Correlation: r = {corr:.3f}',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(figures_dir / "calibration_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir}/calibration_analysis.png")


def plot_quality_distribution(eval_results):
    """Create histogram of quality scores."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ("plausibility_score", "Plausibility Score"),
        ("authenticity_score", "Authenticity Score"),
        ("calibration_score", "Calibration Score"),
        ("overall_quality", "Overall Quality Score")
    ]

    for ax, (metric, title) in zip(axes.flatten(), metrics):
        scores = [r[metric] for r in eval_results]

        ax.hist(scores, bins=np.arange(0.5, 6, 1), edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(scores):.2f}')
        ax.set_xlabel('Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, 6))
        ax.legend()

    plt.suptitle('Distribution of Evaluation Scores (n=60)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "quality_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir}/quality_distribution.png")


def plot_probability_vs_quality(eval_results):
    """Create scatter plot of input probability vs quality metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    probs = [r["event_probability"] for r in eval_results]
    quality = [r["overall_quality"] for r in eval_results]
    modes = [r["generation_mode"] for r in eval_results]

    mode_colors = {
        "zero_shot": "#3498db",
        "probability_conditioned": "#2ecc71",
        "scenario_positive": "#e74c3c",
        "scenario_negative": "#9b59b6"
    }

    mode_labels = {
        "zero_shot": "Zero-Shot",
        "probability_conditioned": "Prob. Conditioned",
        "scenario_positive": "Scenario Positive",
        "scenario_negative": "Scenario Negative"
    }

    for mode in mode_colors:
        mask = [m == mode for m in modes]
        ax.scatter(
            [p for p, m in zip(probs, mask) if m],
            [q for q, m in zip(quality, mask) if m],
            c=mode_colors[mode],
            label=mode_labels[mode],
            alpha=0.7,
            s=100
        )

    ax.set_xlabel('Input Probability', fontsize=12)
    ax.set_ylabel('Overall Quality Score', fontsize=12)
    ax.set_title('Quality vs. Input Probability by Generation Mode', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.5, 5.5)

    # Add trend line
    z = np.polyfit(probs, quality, 1)
    p = np.poly1d(z)
    ax.plot([0, 1], [p(0), p(1)], "k--", alpha=0.5, linewidth=2)

    # Calculate correlation
    corr = np.corrcoef(probs, quality)[0, 1]
    ax.annotate(f'r = {corr:.3f}', xy=(0.02, 5.2), fontsize=11)

    plt.tight_layout()
    plt.savefig(figures_dir / "probability_vs_quality.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir}/probability_vs_quality.png")


def plot_summary_radar(summary):
    """Create radar chart summarizing mode performance."""
    mode_data = summary["mode_comparison"]

    categories = ['Plausibility', 'Authenticity', 'Calibration', 'Lexical\nDiversity\n(×5)']
    num_cats = len(categories)

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    mode_colors = {
        "zero_shot": "#3498db",
        "probability_conditioned": "#2ecc71",
        "scenario_positive": "#e74c3c",
        "scenario_negative": "#9b59b6"
    }

    mode_labels = {
        "zero_shot": "Zero-Shot",
        "probability_conditioned": "Prob. Conditioned",
        "scenario_positive": "Scenario Positive",
        "scenario_negative": "Scenario Negative"
    }

    for mode, color in mode_colors.items():
        values = [
            mode_data[mode]["avg_plausibility"],
            mode_data[mode]["avg_authenticity"],
            mode_data[mode]["avg_calibration"],
            mode_data[mode]["avg_lexical_diversity"] * 5,  # Scale to 0-5 range
        ]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=mode_labels[mode], color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_title('Generation Mode Performance Comparison', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(figures_dir / "performance_radar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir}/performance_radar.png")


def create_all_visualizations():
    """Create all visualizations."""
    print("Creating visualizations...")

    summary, eval_results, articles = load_results()

    plot_mode_comparison(summary)
    plot_calibration_analysis(summary)
    plot_quality_distribution(eval_results)
    plot_probability_vs_quality(eval_results)
    plot_summary_radar(summary)

    print(f"\nAll visualizations saved to {figures_dir}/")


if __name__ == "__main__":
    create_all_visualizations()
