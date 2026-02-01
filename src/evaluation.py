"""
Evaluation Module for Generated News Articles

Evaluates generated articles for:
- Plausibility and news authenticity
- Calibration (does confidence match probability?)
- Text quality metrics
"""

import os
import json
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from collections import Counter
import time

import openai
import numpy as np
from tqdm import tqdm

from news_generator import GeneratedArticle, GenerationMode


# Confidence markers for calibration analysis
HIGH_CONFIDENCE_MARKERS = [
    "will", "certain", "definitely", "confirmed", "officially",
    "announced", "expected to", "set to", "poised to", "inevitable",
    "clear", "undoubtedly", "sure to", "guaranteed", "decided"
]

LOW_CONFIDENCE_MARKERS = [
    "may", "might", "could", "possibly", "potentially",
    "uncertain", "unclear", "speculation", "rumored", "alleged",
    "if", "whether", "remains to be seen", "question", "doubt",
    "perhaps", "reportedly", "sources say", "unconfirmed"
]

HEDGING_PHRASES = [
    "according to sources", "it is believed", "analysts suggest",
    "some experts", "there are indications", "appears to",
    "seems to", "is thought to", "is expected", "likely to"
]


@dataclass
class EvaluationResult:
    """Results from evaluating a single article."""
    article_id: str
    event_question: str
    event_probability: float
    generation_mode: str
    model: str

    # LLM-as-judge scores (1-5)
    plausibility_score: float
    authenticity_score: float
    calibration_score: float
    overall_quality: float

    # LLM-as-judge explanations
    plausibility_explanation: str
    authenticity_explanation: str
    calibration_explanation: str

    # Automated metrics
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    lexical_diversity: float  # type-token ratio

    # Confidence markers
    high_confidence_count: int
    low_confidence_count: int
    hedging_count: int
    confidence_ratio: float  # high / (high + low + 1)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArticleEvaluator:
    """Evaluates generated news articles."""

    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.eval_model = "gpt-4o"  # Using GPT-4o for evaluation

    def _count_markers(self, text: str) -> tuple[int, int, int]:
        """Count confidence markers in text."""
        text_lower = text.lower()

        high_count = sum(1 for marker in HIGH_CONFIDENCE_MARKERS if marker in text_lower)
        low_count = sum(1 for marker in LOW_CONFIDENCE_MARKERS if marker in text_lower)
        hedge_count = sum(1 for phrase in HEDGING_PHRASES if phrase in text_lower)

        return high_count, low_count, hedge_count

    def _compute_text_metrics(self, text: str) -> dict:
        """Compute basic text quality metrics."""
        # Word count
        words = text.split()
        word_count = len(words)

        # Sentence count
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        # Average sentence length
        avg_sentence_length = word_count / max(sentence_count, 1)

        # Lexical diversity (type-token ratio)
        unique_words = set(w.lower() for w in words)
        lexical_diversity = len(unique_words) / max(word_count, 1)

        # Confidence markers
        high_conf, low_conf, hedging = self._count_markers(text)
        confidence_ratio = high_conf / (high_conf + low_conf + 1)

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "lexical_diversity": lexical_diversity,
            "high_confidence_count": high_conf,
            "low_confidence_count": low_conf,
            "hedging_count": hedging,
            "confidence_ratio": confidence_ratio
        }

    def _evaluate_with_llm(
        self,
        article: GeneratedArticle
    ) -> dict:
        """Use LLM as judge to evaluate article quality."""

        eval_prompt = f"""You are an expert evaluator of news article quality.
Evaluate the following generated news article based on three criteria.

ORIGINAL EVENT/QUESTION: {article.event_question}
PREDICTION MARKET PROBABILITY: {article.event_probability:.0%}
GENERATION MODE: {article.generation_mode}

HEADLINE: {article.headline}

ARTICLE:
{article.article_text}

Please evaluate on a scale of 1-5:

1. PLAUSIBILITY (1-5): How believable is this news article? Would a reasonable person accept this as a plausible future scenario?
   - 1: Completely implausible, fantastical
   - 3: Somewhat plausible but with issues
   - 5: Highly plausible and realistic

2. AUTHENTICITY (1-5): Does this read like a real news article? Consider journalistic style, structure, sourcing.
   - 1: Clearly fake/AI-generated, poor style
   - 3: Passable but imperfect
   - 5: Indistinguishable from professional journalism

3. CALIBRATION (1-5): Does the article's level of certainty/hedging match the input probability?
   - For 80%+ probability: Article should be confident
   - For 40-60% probability: Article should show uncertainty
   - For <20% probability: Article should hedge or frame as surprising
   - Score 5 if well-matched, 1 if completely mismatched

Respond in JSON format:
{{
    "plausibility_score": <1-5>,
    "plausibility_explanation": "<one sentence>",
    "authenticity_score": <1-5>,
    "authenticity_explanation": "<one sentence>",
    "calibration_score": <1-5>,
    "calibration_explanation": "<one sentence>"
}}
"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.eval_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Respond only in JSON."},
                    {"role": "user", "content": eval_prompt}
                ],
                max_tokens=500,
                temperature=0.3,
            )

            response_text = response.choices[0].message.content

            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            result = json.loads(response_text.strip())

            return {
                "plausibility_score": float(result.get("plausibility_score", 3)),
                "authenticity_score": float(result.get("authenticity_score", 3)),
                "calibration_score": float(result.get("calibration_score", 3)),
                "plausibility_explanation": result.get("plausibility_explanation", ""),
                "authenticity_explanation": result.get("authenticity_explanation", ""),
                "calibration_explanation": result.get("calibration_explanation", "")
            }

        except Exception as e:
            print(f"LLM evaluation error: {e}")
            return {
                "plausibility_score": 0,
                "authenticity_score": 0,
                "calibration_score": 0,
                "plausibility_explanation": f"Error: {str(e)}",
                "authenticity_explanation": "",
                "calibration_explanation": ""
            }

    def evaluate_article(self, article: GeneratedArticle) -> EvaluationResult:
        """Evaluate a single generated article."""
        # Get text metrics
        text_metrics = self._compute_text_metrics(article.article_text)

        # Get LLM evaluation
        llm_eval = self._evaluate_with_llm(article)

        # Compute overall quality
        overall_quality = np.mean([
            llm_eval["plausibility_score"],
            llm_eval["authenticity_score"],
            llm_eval["calibration_score"]
        ])

        return EvaluationResult(
            article_id=f"{article.event_id}_{article.generation_mode}_{article.model}",
            event_question=article.event_question,
            event_probability=article.event_probability,
            generation_mode=article.generation_mode,
            model=article.model,
            plausibility_score=llm_eval["plausibility_score"],
            authenticity_score=llm_eval["authenticity_score"],
            calibration_score=llm_eval["calibration_score"],
            overall_quality=overall_quality,
            plausibility_explanation=llm_eval["plausibility_explanation"],
            authenticity_explanation=llm_eval["authenticity_explanation"],
            calibration_explanation=llm_eval["calibration_explanation"],
            **text_metrics
        )

    def evaluate_batch(
        self,
        articles: List[GeneratedArticle],
        verbose: bool = True
    ) -> List[EvaluationResult]:
        """Evaluate multiple articles."""
        results = []

        iterator = tqdm(articles, desc="Evaluating articles", disable=not verbose)

        for article in iterator:
            try:
                result = self.evaluate_article(article)
                results.append(result)

                if verbose:
                    iterator.set_postfix({
                        "quality": f"{result.overall_quality:.2f}",
                        "mode": article.generation_mode[:10]
                    })

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"Error evaluating article: {e}")
                continue

        return results


def analyze_calibration(results: List[EvaluationResult]) -> Dict[str, Any]:
    """
    Analyze calibration between input probability and linguistic confidence.

    Returns statistics on how well article confidence matches input probability.
    """
    # Group by probability range
    prob_ranges = {
        "very_low": (0.0, 0.2),
        "low": (0.2, 0.4),
        "medium": (0.4, 0.6),
        "high": (0.6, 0.8),
        "very_high": (0.8, 1.0)
    }

    analysis = {}

    for range_name, (low, high) in prob_ranges.items():
        range_results = [
            r for r in results
            if low <= r.event_probability < high
        ]

        if range_results:
            avg_confidence_ratio = np.mean([r.confidence_ratio for r in range_results])
            avg_calibration_score = np.mean([r.calibration_score for r in range_results])
            avg_high_conf_markers = np.mean([r.high_confidence_count for r in range_results])
            avg_low_conf_markers = np.mean([r.low_confidence_count for r in range_results])

            analysis[range_name] = {
                "probability_range": f"{low:.0%}-{high:.0%}",
                "count": len(range_results),
                "avg_confidence_ratio": avg_confidence_ratio,
                "avg_calibration_score": avg_calibration_score,
                "avg_high_conf_markers": avg_high_conf_markers,
                "avg_low_conf_markers": avg_low_conf_markers
            }

    # Compute correlation
    if results:
        probs = [r.event_probability for r in results]
        conf_ratios = [r.confidence_ratio for r in results]

        correlation = np.corrcoef(probs, conf_ratios)[0, 1]
        analysis["overall_correlation"] = float(correlation) if not np.isnan(correlation) else 0.0

    return analysis


def compare_generation_modes(results: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
    """Compare performance across different generation modes."""
    modes = set(r.generation_mode for r in results)
    comparison = {}

    for mode in modes:
        mode_results = [r for r in results if r.generation_mode == mode]

        if mode_results:
            comparison[mode] = {
                "count": len(mode_results),
                "avg_plausibility": np.mean([r.plausibility_score for r in mode_results]),
                "avg_authenticity": np.mean([r.authenticity_score for r in mode_results]),
                "avg_calibration": np.mean([r.calibration_score for r in mode_results]),
                "avg_overall": np.mean([r.overall_quality for r in mode_results]),
                "std_overall": np.std([r.overall_quality for r in mode_results]),
                "avg_word_count": np.mean([r.word_count for r in mode_results]),
                "avg_lexical_diversity": np.mean([r.lexical_diversity for r in mode_results])
            }

    return comparison


def run_evaluation(
    articles: List[GeneratedArticle],
    output_dir: str = "results"
) -> tuple[List[EvaluationResult], Dict[str, Any]]:
    """
    Run full evaluation pipeline.

    Returns:
        Tuple of (evaluation results, analysis summary)
    """
    print(f"Evaluating {len(articles)} articles...")

    evaluator = ArticleEvaluator()
    results = evaluator.evaluate_batch(articles, verbose=True)

    # Compute analysis
    calibration_analysis = analyze_calibration(results)
    mode_comparison = compare_generation_modes(results)

    analysis = {
        "calibration": calibration_analysis,
        "mode_comparison": mode_comparison,
        "total_articles": len(results),
        "avg_quality": np.mean([r.overall_quality for r in results]) if results else 0
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    analysis_file = os.path.join(output_dir, "evaluation_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"Saved evaluation results to {results_file}")
    print(f"Saved analysis to {analysis_file}")

    return results, analysis


if __name__ == "__main__":
    # Test evaluation with sample data
    from news_generator import GeneratedArticle

    # Create a sample article for testing
    sample_article = GeneratedArticle(
        event_id="test_1",
        event_question="Will the Federal Reserve cut interest rates by 50 basis points in 2026?",
        event_probability=0.65,
        generation_mode="probability_conditioned",
        model="gpt-4o",
        headline="Fed Signals Rate Cut Amid Economic Uncertainty",
        article_text="""
        WASHINGTON - Federal Reserve officials are expected to announce a significant
        interest rate cut in their upcoming meeting, according to sources close to the matter.

        The anticipated 50 basis point reduction would mark one of the largest single cuts
        in recent years, reflecting concerns about economic growth.

        "We are monitoring the situation closely," said one Federal Reserve official,
        speaking on condition of anonymity. "The data suggests action may be needed."

        Market analysts have been predicting this move for months, with many citing
        slowing job growth and persistent inflation concerns.
        """,
        timestamp="2026-01-31T12:00:00",
        prompt_used="test prompt",
        generation_time_seconds=2.5
    )

    evaluator = ArticleEvaluator()
    result = evaluator.evaluate_article(sample_article)

    print(f"\nEvaluation Results:")
    print(f"  Plausibility: {result.plausibility_score}/5 - {result.plausibility_explanation}")
    print(f"  Authenticity: {result.authenticity_score}/5 - {result.authenticity_explanation}")
    print(f"  Calibration: {result.calibration_score}/5 - {result.calibration_explanation}")
    print(f"  Overall Quality: {result.overall_quality:.2f}/5")
    print(f"\n  Word Count: {result.word_count}")
    print(f"  Confidence Ratio: {result.confidence_ratio:.2f}")
