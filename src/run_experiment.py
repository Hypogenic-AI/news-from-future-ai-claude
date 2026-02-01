"""
Main Experiment Runner for News from the Future

This script runs the complete experiment:
1. Fetch prediction market events
2. Generate news articles with different approaches
3. Evaluate generated articles
4. Analyze and visualize results
"""

import os
import sys
import json
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Local imports
from prediction_markets import fetch_prediction_events, get_sample_events, PredictionEvent
from news_generator import NewsGenerator, GeneratedArticle, GenerationMode, run_generation_experiment
from evaluation import ArticleEvaluator, EvaluationResult, run_evaluation, analyze_calibration, compare_generation_modes


def select_diverse_events(events: List[PredictionEvent], n: int = 15) -> List[PredictionEvent]:
    """
    Select a diverse subset of events across categories and probability ranges.
    """
    # Group by probability range
    prob_ranges = [
        (0.0, 0.3),   # unlikely
        (0.3, 0.5),   # uncertain-low
        (0.5, 0.7),   # uncertain-high
        (0.7, 1.0),   # likely
    ]

    selected = []
    seen_questions = set()

    for low, high in prob_ranges:
        range_events = [
            e for e in events
            if low <= e.probability < high and e.question not in seen_questions
        ]
        # Select up to n/4 from each range
        sample_size = min(len(range_events), n // 4 + 1)
        sampled = random.sample(range_events, sample_size) if range_events else []
        for e in sampled:
            seen_questions.add(e.question)
        selected.extend(sampled)

    # Fill remaining slots if needed
    remaining = [e for e in events if e.question not in seen_questions]
    if len(selected) < n and remaining:
        additional = random.sample(remaining, min(len(remaining), n - len(selected)))
        selected.extend(additional)

    return selected[:n]


def run_full_experiment(
    use_live_api: bool = True,
    max_events: int = 15,
    models: List[str] = None,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Run the complete experiment pipeline.

    Args:
        use_live_api: Whether to fetch from Polymarket API
        max_events: Maximum number of events to process
        models: List of model providers (default: ["openai"])
        output_dir: Directory for saving results

    Returns:
        Dictionary containing all experiment results
    """
    if models is None:
        models = ["openai"]  # Default to just OpenAI for faster execution

    print("=" * 70)
    print("NEWS FROM THE FUTURE - EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Random seed: {SEED}")
    print(f"Max events: {max_events}")
    print(f"Models: {models}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Record experiment configuration
    config = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "max_events": max_events,
        "models": models,
        "use_live_api": use_live_api,
        "generation_modes": [m.value for m in GenerationMode]
    }

    # ===== PHASE 1: DATA COLLECTION =====
    print("\n" + "=" * 70)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 70)

    # Fetch events
    all_events = fetch_prediction_events(use_live_api=use_live_api)
    print(f"Total events available: {len(all_events)}")

    # Select diverse subset
    selected_events = select_diverse_events(all_events, n=max_events)
    print(f"Selected {len(selected_events)} diverse events")

    # Also add sample events for guaranteed coverage
    sample_events = get_sample_events()[:5]
    combined_events = selected_events + sample_events
    # Deduplicate
    seen = set()
    unique_events = []
    for e in combined_events:
        if e.question not in seen:
            seen.add(e.question)
            unique_events.append(e)
    events = unique_events[:max_events]

    print(f"\nFinal event count: {len(events)}")

    # Print event summary
    print("\nSelected Events:")
    for i, event in enumerate(events):
        print(f"  {i+1}. [{event.probability:.0%}] {event.question[:60]}...")

    # Save events
    events_file = os.path.join(output_dir, "events.json")
    with open(events_file, 'w') as f:
        json.dump([e.to_dict() for e in events], f, indent=2)
    print(f"\nSaved events to {events_file}")

    # ===== PHASE 2: ARTICLE GENERATION =====
    print("\n" + "=" * 70)
    print("PHASE 2: ARTICLE GENERATION")
    print("=" * 70)

    all_articles = []
    generation_modes = [
        GenerationMode.ZERO_SHOT,
        GenerationMode.PROBABILITY_CONDITIONED,
        GenerationMode.SCENARIO_POSITIVE,
        GenerationMode.SCENARIO_NEGATIVE
    ]

    for model_provider in models:
        print(f"\n--- Generating with {model_provider} ---")
        try:
            generator = NewsGenerator(model_provider=model_provider)
            articles = generator.generate_batch(events, generation_modes, verbose=True)
            all_articles.extend(articles)

            # Save model-specific results
            model_file = os.path.join(output_dir, f"articles_{model_provider}.json")
            with open(model_file, 'w') as f:
                json.dump([a.to_dict() for a in articles], f, indent=2)
            print(f"Saved {len(articles)} articles to {model_file}")

        except Exception as e:
            print(f"Error with {model_provider}: {e}")

    print(f"\nTotal articles generated: {len(all_articles)}")

    # Save all articles
    all_articles_file = os.path.join(output_dir, "all_articles.json")
    with open(all_articles_file, 'w') as f:
        json.dump([a.to_dict() for a in all_articles], f, indent=2)

    # ===== PHASE 3: EVALUATION =====
    print("\n" + "=" * 70)
    print("PHASE 3: EVALUATION")
    print("=" * 70)

    eval_results, analysis = run_evaluation(all_articles, output_dir=output_dir)

    # ===== PHASE 4: ANALYSIS =====
    print("\n" + "=" * 70)
    print("PHASE 4: ANALYSIS SUMMARY")
    print("=" * 70)

    # Mode comparison
    print("\nGeneration Mode Comparison:")
    mode_comparison = analysis.get("mode_comparison", {})
    for mode, stats in mode_comparison.items():
        print(f"\n  {mode}:")
        print(f"    Count: {stats.get('count', 0)}")
        print(f"    Avg Plausibility: {stats.get('avg_plausibility', 0):.2f}/5")
        print(f"    Avg Authenticity: {stats.get('avg_authenticity', 0):.2f}/5")
        print(f"    Avg Calibration: {stats.get('avg_calibration', 0):.2f}/5")
        print(f"    Avg Overall: {stats.get('avg_overall', 0):.2f}/5")

    # Calibration analysis
    print("\nCalibration Analysis:")
    calibration = analysis.get("calibration", {})
    if "overall_correlation" in calibration:
        print(f"  Probability-Confidence Correlation: {calibration['overall_correlation']:.3f}")

    for range_name, stats in calibration.items():
        if isinstance(stats, dict) and "probability_range" in stats:
            print(f"\n  {range_name} ({stats['probability_range']}):")
            print(f"    Articles: {stats['count']}")
            print(f"    Avg Calibration Score: {stats['avg_calibration_score']:.2f}/5")
            print(f"    Avg Confidence Ratio: {stats['avg_confidence_ratio']:.2f}")

    # Save final summary
    summary = {
        "config": config,
        "events_count": len(events),
        "articles_count": len(all_articles),
        "evaluations_count": len(eval_results),
        "mode_comparison": mode_comparison,
        "calibration": calibration,
        "avg_overall_quality": analysis.get("avg_quality", 0)
    }

    summary_file = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved experiment summary to {summary_file}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - events.json: {len(events)} prediction events")
    print(f"  - all_articles.json: {len(all_articles)} generated articles")
    print(f"  - evaluation_results.json: {len(eval_results)} evaluation results")
    print(f"  - evaluation_analysis.json: Analysis summary")
    print(f"  - experiment_summary.json: Full experiment summary")

    return {
        "events": events,
        "articles": all_articles,
        "eval_results": eval_results,
        "analysis": analysis,
        "summary": summary
    }


def print_example_articles(articles: List[GeneratedArticle], n: int = 3):
    """Print example generated articles."""
    print("\n" + "=" * 70)
    print("EXAMPLE GENERATED ARTICLES")
    print("=" * 70)

    # Group by mode
    mode_examples = {}
    for article in articles:
        mode = article.generation_mode
        if mode not in mode_examples:
            mode_examples[mode] = []
        mode_examples[mode].append(article)

    for mode, mode_articles in mode_examples.items():
        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*60}")

        for article in mode_articles[:1]:  # One example per mode
            print(f"\nEvent: {article.event_question}")
            print(f"Probability: {article.event_probability:.0%}")
            print(f"\nHEADLINE: {article.headline}")
            print(f"\nARTICLE:\n{article.article_text[:600]}...")
            print("-" * 40)


if __name__ == "__main__":
    # Run the full experiment
    results = run_full_experiment(
        use_live_api=True,
        max_events=15,
        models=["openai"],  # Using OpenAI for faster execution
        output_dir="results"
    )

    # Print example articles
    if results["articles"]:
        print_example_articles(results["articles"], n=4)

    print("\nExperiment completed successfully!")
