"""
Prediction Market Data Collection Module

Fetches current prediction market data from Polymarket API
for use in generating future news articles.
"""

import httpx
import json
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
import os


@dataclass
class PredictionEvent:
    """Represents a prediction market event."""
    id: str
    question: str
    description: str
    probability: float  # 0.0 to 1.0
    resolution_date: Optional[str]
    category: str
    outcome_yes: str
    outcome_no: str
    source: str  # e.g., "polymarket", "metaculus"

    def to_dict(self):
        return {
            "id": self.id,
            "question": self.question,
            "description": self.description,
            "probability": self.probability,
            "resolution_date": self.resolution_date,
            "category": self.category,
            "outcome_yes": self.outcome_yes,
            "outcome_no": self.outcome_no,
            "source": self.source
        }


class PolymarketClient:
    """Client for fetching data from Polymarket's public API."""

    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self):
        self.client = httpx.Client(timeout=30.0)

    def _get_markets(self, limit: int = 50, active: bool = True) -> List[dict]:
        """Fetch markets from Polymarket API."""
        try:
            params = {
                "limit": limit,
                "active": str(active).lower(),
                "closed": "false",
            }
            response = self.client.get(f"{self.BASE_URL}/markets", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching from Polymarket: {e}")
            return []

    def get_events(self, limit: int = 30) -> List[PredictionEvent]:
        """Fetch and parse prediction events."""
        markets = self._get_markets(limit=limit)
        events = []

        for market in markets:
            try:
                # Extract probability from outcomes
                prob = 0.5  # default
                if "outcomePrices" in market:
                    prices = market.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        try:
                            prices = json.loads(prices)
                        except:
                            prices = [0.5, 0.5]
                    if prices and len(prices) > 0:
                        prob = float(prices[0])

                # Parse resolution date
                resolution_date = market.get("endDate") or market.get("resolutionDate")

                # Determine category from tags or condition
                category = "general"
                if "tags" in market and market["tags"]:
                    tags = market["tags"]
                    if isinstance(tags, list) and len(tags) > 0:
                        category = tags[0].lower()

                event = PredictionEvent(
                    id=str(market.get("id", "")),
                    question=market.get("question", ""),
                    description=market.get("description", ""),
                    probability=prob,
                    resolution_date=resolution_date,
                    category=category,
                    outcome_yes=market.get("outcomes", ["Yes", "No"])[0] if market.get("outcomes") else "Yes",
                    outcome_no=market.get("outcomes", ["Yes", "No"])[1] if market.get("outcomes") and len(market.get("outcomes")) > 1 else "No",
                    source="polymarket"
                )

                # Only include events with valid questions
                if event.question and len(event.question) > 10:
                    events.append(event)

            except Exception as e:
                print(f"Error parsing market: {e}")
                continue

        return events

    def close(self):
        self.client.close()


def get_sample_events() -> List[PredictionEvent]:
    """
    Return a curated set of sample prediction events for testing.
    These represent realistic prediction market questions across categories.
    """
    sample_events = [
        PredictionEvent(
            id="sample_1",
            question="Will the Federal Reserve cut interest rates by at least 50 basis points in 2026?",
            description="This market resolves YES if the Federal Reserve cuts the federal funds rate by 50 basis points or more cumulatively in 2026.",
            probability=0.65,
            resolution_date="2026-12-31",
            category="economics",
            outcome_yes="Rate cut of 50+ bps",
            outcome_no="Less than 50 bps cut",
            source="sample"
        ),
        PredictionEvent(
            id="sample_2",
            question="Will GPT-5 be released by OpenAI before July 2026?",
            description="This market resolves YES if OpenAI publicly releases a model officially named GPT-5 before July 1, 2026.",
            probability=0.45,
            resolution_date="2026-07-01",
            category="technology",
            outcome_yes="GPT-5 released",
            outcome_no="Not released by date",
            source="sample"
        ),
        PredictionEvent(
            id="sample_3",
            question="Will global temperatures in 2026 exceed 1.5C above pre-industrial levels?",
            description="This market resolves YES if the global mean temperature for 2026 exceeds 1.5 degrees Celsius above pre-industrial baseline.",
            probability=0.72,
            resolution_date="2027-01-15",
            category="climate",
            outcome_yes="Exceeds 1.5C",
            outcome_no="Below 1.5C",
            source="sample"
        ),
        PredictionEvent(
            id="sample_4",
            question="Will Bitcoin reach $150,000 before June 2026?",
            description="This market resolves YES if Bitcoin reaches or exceeds $150,000 USD on any major exchange before June 1, 2026.",
            probability=0.28,
            resolution_date="2026-06-01",
            category="crypto",
            outcome_yes="BTC >= $150K",
            outcome_no="BTC < $150K",
            source="sample"
        ),
        PredictionEvent(
            id="sample_5",
            question="Will SpaceX successfully land a Starship booster on the launch tower in 2026?",
            description="This market resolves YES if SpaceX successfully catches a Starship Super Heavy booster on the launch tower ('chopsticks') in 2026.",
            probability=0.85,
            resolution_date="2026-12-31",
            category="space",
            outcome_yes="Successful tower catch",
            outcome_no="No successful catch",
            source="sample"
        ),
        PredictionEvent(
            id="sample_6",
            question="Will a major tech company announce layoffs of 10,000+ employees in Q1 2026?",
            description="This market resolves YES if any company with market cap > $100B announces layoffs of 10,000 or more employees in Q1 2026.",
            probability=0.55,
            resolution_date="2026-04-01",
            category="business",
            outcome_yes="Major layoffs announced",
            outcome_no="No major layoffs",
            source="sample"
        ),
        PredictionEvent(
            id="sample_7",
            question="Will the US unemployment rate exceed 5% in 2026?",
            description="This market resolves YES if the US unemployment rate exceeds 5% at any point during 2026.",
            probability=0.35,
            resolution_date="2026-12-31",
            category="economics",
            outcome_yes="Unemployment > 5%",
            outcome_no="Unemployment <= 5%",
            source="sample"
        ),
        PredictionEvent(
            id="sample_8",
            question="Will a new COVID-19 variant cause WHO to declare a health emergency in 2026?",
            description="This market resolves YES if WHO declares a Public Health Emergency of International Concern related to a new COVID-19 variant in 2026.",
            probability=0.18,
            resolution_date="2026-12-31",
            category="health",
            outcome_yes="WHO emergency declared",
            outcome_no="No emergency declared",
            source="sample"
        ),
        PredictionEvent(
            id="sample_9",
            question="Will autonomous vehicles be approved for commercial operation in 5+ US states by end of 2026?",
            description="This market resolves YES if fully autonomous vehicles (no safety driver) are approved for commercial passenger service in 5 or more US states by December 31, 2026.",
            probability=0.62,
            resolution_date="2026-12-31",
            category="technology",
            outcome_yes="5+ states approve",
            outcome_no="Fewer than 5 states",
            source="sample"
        ),
        PredictionEvent(
            id="sample_10",
            question="Will there be a ceasefire in the Russia-Ukraine conflict lasting 30+ days in 2026?",
            description="This market resolves YES if there is an official ceasefire between Russia and Ukraine that lasts at least 30 consecutive days in 2026.",
            probability=0.40,
            resolution_date="2026-12-31",
            category="geopolitics",
            outcome_yes="30+ day ceasefire",
            outcome_no="No lasting ceasefire",
            source="sample"
        ),
        PredictionEvent(
            id="sample_11",
            question="Will lab-grown meat receive FDA approval for restaurant sales in 2026?",
            description="This market resolves YES if the FDA approves lab-grown (cultured) meat products for sale in restaurants in the United States during 2026.",
            probability=0.48,
            resolution_date="2026-12-31",
            category="food",
            outcome_yes="FDA approves",
            outcome_no="Not approved",
            source="sample"
        ),
        PredictionEvent(
            id="sample_12",
            question="Will any AI system pass the Turing test as judged by expert panel in 2026?",
            description="This market resolves YES if any AI system passes a formal Turing test administered by a panel of AI experts using rigorous methodology in 2026.",
            probability=0.25,
            resolution_date="2026-12-31",
            category="technology",
            outcome_yes="AI passes Turing test",
            outcome_no="No AI passes",
            source="sample"
        ),
        PredictionEvent(
            id="sample_13",
            question="Will Apple release AR glasses in 2026?",
            description="This market resolves YES if Apple releases consumer AR glasses (distinct from Vision Pro VR/MR headset) in 2026.",
            probability=0.38,
            resolution_date="2026-12-31",
            category="technology",
            outcome_yes="Apple AR glasses released",
            outcome_no="Not released",
            source="sample"
        ),
        PredictionEvent(
            id="sample_14",
            question="Will the S&P 500 close above 7,000 at any point in 2026?",
            description="This market resolves YES if the S&P 500 index closes above 7,000 points on any trading day in 2026.",
            probability=0.58,
            resolution_date="2026-12-31",
            category="finance",
            outcome_yes="S&P 500 > 7,000",
            outcome_no="S&P 500 <= 7,000",
            source="sample"
        ),
        PredictionEvent(
            id="sample_15",
            question="Will nuclear fusion generate net energy output commercially by end of 2026?",
            description="This market resolves YES if any organization demonstrates commercial-scale nuclear fusion with sustained net energy gain by December 31, 2026.",
            probability=0.12,
            resolution_date="2026-12-31",
            category="energy",
            outcome_yes="Commercial fusion achieved",
            outcome_no="Not achieved",
            source="sample"
        ),
        PredictionEvent(
            id="sample_16",
            question="Will California experience a major earthquake (7.0+) in 2026?",
            description="This market resolves YES if California experiences an earthquake of magnitude 7.0 or greater on the Richter scale in 2026.",
            probability=0.22,
            resolution_date="2026-12-31",
            category="natural_disaster",
            outcome_yes="Major earthquake occurs",
            outcome_no="No major earthquake",
            source="sample"
        ),
        PredictionEvent(
            id="sample_17",
            question="Will the EU impose new AI regulations affecting major tech companies in 2026?",
            description="This market resolves YES if the European Union enacts new regulations specifically targeting AI systems that affect companies with market cap > $500B.",
            probability=0.78,
            resolution_date="2026-12-31",
            category="regulation",
            outcome_yes="New EU AI regulations",
            outcome_no="No new regulations",
            source="sample"
        ),
        PredictionEvent(
            id="sample_18",
            question="Will China's GDP growth exceed 5% in 2026?",
            description="This market resolves YES if China's official GDP growth rate for 2026 exceeds 5%.",
            probability=0.42,
            resolution_date="2027-01-31",
            category="economics",
            outcome_yes="GDP growth > 5%",
            outcome_no="GDP growth <= 5%",
            source="sample"
        ),
        PredictionEvent(
            id="sample_19",
            question="Will a quantum computer solve a real-world problem faster than classical computers in 2026?",
            description="This market resolves YES if a quantum computer demonstrates practical quantum advantage for a commercially relevant problem in 2026.",
            probability=0.52,
            resolution_date="2026-12-31",
            category="technology",
            outcome_yes="Quantum advantage demonstrated",
            outcome_no="Not demonstrated",
            source="sample"
        ),
        PredictionEvent(
            id="sample_20",
            question="Will the US pass federal data privacy legislation in 2026?",
            description="This market resolves YES if the US Congress passes comprehensive federal data privacy legislation in 2026.",
            probability=0.32,
            resolution_date="2026-12-31",
            category="regulation",
            outcome_yes="Privacy law passed",
            outcome_no="No law passed",
            source="sample"
        ),
    ]
    return sample_events


def fetch_prediction_events(use_live_api: bool = True) -> List[PredictionEvent]:
    """
    Fetch prediction events from live API or return sample events.

    Args:
        use_live_api: If True, attempt to fetch from Polymarket. Falls back to samples.

    Returns:
        List of PredictionEvent objects.
    """
    events = []

    if use_live_api:
        try:
            client = PolymarketClient()
            events = client.get_events(limit=30)
            client.close()
            print(f"Fetched {len(events)} events from Polymarket API")
        except Exception as e:
            print(f"Failed to fetch from API: {e}")

    # If no events from API or API disabled, use samples
    if not events:
        events = get_sample_events()
        print(f"Using {len(events)} sample prediction events")

    return events


if __name__ == "__main__":
    # Test the module
    events = fetch_prediction_events(use_live_api=True)
    print(f"\nFetched {len(events)} events:")
    for i, event in enumerate(events[:5]):
        print(f"\n{i+1}. {event.question}")
        print(f"   Probability: {event.probability:.1%}")
        print(f"   Category: {event.category}")
        print(f"   Resolution: {event.resolution_date}")
