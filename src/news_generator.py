"""
News Article Generation Pipeline

Generates plausible news articles about future events using LLMs,
conditioned on prediction market probabilities.
"""

import os
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import time

import openai
import anthropic
from tqdm import tqdm

from prediction_markets import PredictionEvent, fetch_prediction_events, get_sample_events


class GenerationMode(Enum):
    """Different modes for generating news articles."""
    ZERO_SHOT = "zero_shot"  # No probability conditioning
    PROBABILITY_CONDITIONED = "probability_conditioned"  # Direct probability input
    SCENARIO_POSITIVE = "scenario_positive"  # Assume event happens
    SCENARIO_NEGATIVE = "scenario_negative"  # Assume event doesn't happen


@dataclass
class GeneratedArticle:
    """Represents a generated news article."""
    event_id: str
    event_question: str
    event_probability: float
    generation_mode: str
    model: str
    article_text: str
    headline: str
    timestamp: str
    prompt_used: str
    generation_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NewsGenerator:
    """Generates future news articles using LLMs."""

    def __init__(self, model_provider: str = "openai"):
        """
        Initialize the news generator.

        Args:
            model_provider: "openai" or "anthropic"
        """
        self.model_provider = model_provider
        self.openai_client = None
        self.anthropic_client = None

        if model_provider == "openai":
            self.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.model = "gpt-4o"  # Using GPT-4o as it's available and high quality
        elif model_provider == "anthropic":
            self.anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            self.model = "claude-sonnet-4-20250514"  # Claude Sonnet 4.5
        else:
            raise ValueError(f"Unknown provider: {model_provider}")

    def _get_probability_description(self, probability: float) -> str:
        """Convert probability to natural language description."""
        if probability >= 0.9:
            return "highly likely (over 90% probability)"
        elif probability >= 0.75:
            return "likely (around 75-90% probability)"
        elif probability >= 0.6:
            return "moderately likely (around 60-75% probability)"
        elif probability >= 0.4:
            return "uncertain, roughly even odds (around 40-60% probability)"
        elif probability >= 0.25:
            return "somewhat unlikely (around 25-40% probability)"
        elif probability >= 0.1:
            return "unlikely (around 10-25% probability)"
        else:
            return "very unlikely (under 10% probability)"

    def _build_prompt(
        self,
        event: PredictionEvent,
        mode: GenerationMode
    ) -> str:
        """Build the prompt for article generation based on mode."""

        base_context = f"""You are a skilled news journalist writing for a major publication.
Your task is to write a realistic, professional news article.

The article should:
- Be written in standard news article format (inverted pyramid structure)
- Include a compelling headline
- Be 200-400 words
- Include realistic quotes from plausible experts or officials
- Use professional journalistic language
- Be dated in 2026 (specify a realistic date)
- Feel authentic and believable

Event/Topic: {event.question}
Additional context: {event.description}
Category: {event.category}
"""

        if mode == GenerationMode.ZERO_SHOT:
            prompt = f"""{base_context}

Write a news article about this topic, imagining a plausible future outcome.
The article should read as if it's reporting on the actual outcome of this event.

Format your response as:
HEADLINE: [Your headline]

ARTICLE:
[Your article text]
"""

        elif mode == GenerationMode.PROBABILITY_CONDITIONED:
            prob_desc = self._get_probability_description(event.probability)
            prompt = f"""{base_context}

IMPORTANT: According to prediction markets, this event is considered {prob_desc}.
Current probability estimate: {event.probability:.0%}

Write a news article that reflects this probability level.
- If the probability is high (>70%), write as if the event has occurred or is about to occur with high confidence.
- If the probability is moderate (30-70%), write with appropriate hedging and uncertainty.
- If the probability is low (<30%), you may write about either:
  a) The event occurring as a surprising development, OR
  b) Why the event is unlikely to occur

Your narrative confidence should match the probability level.

Format your response as:
HEADLINE: [Your headline]

ARTICLE:
[Your article text]
"""

        elif mode == GenerationMode.SCENARIO_POSITIVE:
            prompt = f"""{base_context}

SCENARIO: Assume this event HAS HAPPENED or WILL HAPPEN.
Write a news article reporting on this outcome as if it has occurred.

Current market probability: {event.probability:.0%} (for context)

Write the article from the perspective that "{event.outcome_yes}" is the outcome.
Make it feel like breaking news or a recent development.

Format your response as:
HEADLINE: [Your headline]

ARTICLE:
[Your article text]
"""

        elif mode == GenerationMode.SCENARIO_NEGATIVE:
            prompt = f"""{base_context}

SCENARIO: Assume this event has NOT HAPPENED or WILL NOT HAPPEN.
Write a news article reporting on this outcome.

Current market probability: {event.probability:.0%} (for context)

Write the article from the perspective that "{event.outcome_no}" is the outcome.
This could be news about the event not occurring, or about it failing/being prevented.

Format your response as:
HEADLINE: [Your headline]

ARTICLE:
[Your article text]
"""

        return prompt

    def _generate_with_openai(self, prompt: str) -> str:
        """Generate text using OpenAI API."""
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional news journalist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def _generate_with_anthropic(self, prompt: str) -> str:
        """Generate text using Anthropic API."""
        response = self.anthropic_client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[
                {"role": "user", "content": prompt}
            ],
            system="You are a professional news journalist.",
        )
        return response.content[0].text

    def _parse_article(self, raw_text: str) -> tuple[str, str]:
        """Parse headline and article body from generated text."""
        headline = ""
        article = ""

        lines = raw_text.strip().split('\n')
        in_article = False

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.upper().startswith("HEADLINE:"):
                headline = line_stripped[9:].strip()
            elif line_stripped.upper() == "ARTICLE:":
                in_article = True
            elif in_article or (headline and not line_stripped.upper().startswith("HEADLINE")):
                article += line + "\n"

        # Fallback if parsing fails
        if not headline:
            # Try to extract first non-empty line as headline
            for line in lines:
                if line.strip():
                    headline = line.strip()
                    break
        if not article:
            article = raw_text

        return headline.strip(), article.strip()

    def generate_article(
        self,
        event: PredictionEvent,
        mode: GenerationMode
    ) -> GeneratedArticle:
        """Generate a single news article for an event."""
        prompt = self._build_prompt(event, mode)

        start_time = time.time()

        if self.model_provider == "openai":
            raw_text = self._generate_with_openai(prompt)
        else:
            raw_text = self._generate_with_anthropic(prompt)

        generation_time = time.time() - start_time

        headline, article_body = self._parse_article(raw_text)

        return GeneratedArticle(
            event_id=event.id,
            event_question=event.question,
            event_probability=event.probability,
            generation_mode=mode.value,
            model=self.model,
            article_text=article_body,
            headline=headline,
            timestamp=datetime.now().isoformat(),
            prompt_used=prompt,
            generation_time_seconds=generation_time
        )

    def generate_batch(
        self,
        events: List[PredictionEvent],
        modes: List[GenerationMode],
        verbose: bool = True
    ) -> List[GeneratedArticle]:
        """Generate articles for multiple events and modes."""
        articles = []
        total = len(events) * len(modes)

        iterator = tqdm(
            [(event, mode) for event in events for mode in modes],
            total=total,
            desc="Generating articles",
            disable=not verbose
        )

        for event, mode in iterator:
            try:
                article = self.generate_article(event, mode)
                articles.append(article)
                if verbose:
                    iterator.set_postfix({"event": event.question[:30], "mode": mode.value})

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"Error generating for {event.question[:50]}: {e}")
                continue

        return articles


def run_generation_experiment(
    events: List[PredictionEvent],
    models: List[str] = ["openai"],
    modes: List[GenerationMode] = None,
    max_events: int = 20,
    output_dir: str = "results"
) -> Dict[str, List[GeneratedArticle]]:
    """
    Run a full generation experiment.

    Args:
        events: List of prediction events
        models: List of model providers to test
        modes: Generation modes to test
        max_events: Maximum events to process
        output_dir: Directory to save results

    Returns:
        Dictionary mapping model names to generated articles
    """
    if modes is None:
        modes = [
            GenerationMode.ZERO_SHOT,
            GenerationMode.PROBABILITY_CONDITIONED,
            GenerationMode.SCENARIO_POSITIVE,
            GenerationMode.SCENARIO_NEGATIVE
        ]

    # Limit events
    events = events[:max_events]
    print(f"Running experiment with {len(events)} events and {len(modes)} modes")

    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for model_provider in models:
        print(f"\n{'='*60}")
        print(f"Generating with {model_provider}")
        print(f"{'='*60}")

        try:
            generator = NewsGenerator(model_provider=model_provider)
            articles = generator.generate_batch(events, modes, verbose=True)
            all_results[model_provider] = articles

            # Save intermediate results
            output_file = os.path.join(output_dir, f"articles_{model_provider}.json")
            with open(output_file, 'w') as f:
                json.dump([a.to_dict() for a in articles], f, indent=2)
            print(f"Saved {len(articles)} articles to {output_file}")

        except Exception as e:
            print(f"Error with {model_provider}: {e}")
            all_results[model_provider] = []

    return all_results


if __name__ == "__main__":
    # Test with sample events
    print("Testing News Generator with sample events...")

    # Get sample events
    events = get_sample_events()[:3]

    # Test with OpenAI
    generator = NewsGenerator(model_provider="openai")

    for event in events:
        print(f"\n{'='*60}")
        print(f"Event: {event.question}")
        print(f"Probability: {event.probability:.0%}")
        print(f"{'='*60}")

        # Generate with probability conditioning
        article = generator.generate_article(event, GenerationMode.PROBABILITY_CONDITIONED)

        print(f"\nHEADLINE: {article.headline}")
        print(f"\nARTICLE:\n{article.article_text[:500]}...")
        print(f"\nGeneration time: {article.generation_time_seconds:.2f}s")
