# News from the Future

Generating plausible news articles about future events using LLMs conditioned on prediction market probabilities.

## Key Findings

- **Probability-conditioned generation achieves 4.53/5 quality** (33% improvement over zero-shot baseline)
- **Perfect calibration (5.0/5)**: LLMs effectively translate numerical probabilities into appropriately confident narrative language
- **High authenticity (4.33/5)**: Generated articles read like professional journalism
- **Positive correlation (r=0.21)** between input probability and linguistic confidence markers

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run full experiment
python src/run_experiment.py

# Create visualizations
python src/create_visualizations.py
```

## Project Structure

```
.
├── src/
│   ├── prediction_markets.py    # Polymarket API data collection
│   ├── news_generator.py        # Article generation with 4 modes
│   ├── evaluation.py            # LLM-as-judge evaluation
│   ├── run_experiment.py        # Main experiment runner
│   └── create_visualizations.py # Generate result plots
├── results/
│   ├── all_articles.json        # 60 generated articles
│   ├── evaluation_results.json  # Evaluation scores
│   ├── experiment_summary.json  # Summary statistics
│   └── figures/                 # Visualization plots
├── planning.md                  # Research plan
├── REPORT.md                    # Full research report
└── README.md
```

## Generation Modes

| Mode | Description | Avg Quality |
|------|-------------|-------------|
| **Probability-Conditioned** | Explicit probability input | 4.53/5 |
| Scenario-Negative | Assume event doesn't occur | 4.29/5 |
| Scenario-Positive | Assume event occurs | 3.53/5 |
| Zero-Shot | No probability input | 3.40/5 |

## Example Output

**Event**: "Will the U.S. collect between $100b and $200b in revenue in 2025?"
**Probability**: 6%

**Generated (Probability-Conditioned)**:
> *Headline*: "U.S. Customs Revenue Unlikely to Hit $100 Billion Mark in 2025, Experts Say"
>
> *Article*: "The U.S. government's customs revenue... has fallen short of the ambitious $100 billion benchmark... Experts suggest that significant hurdles remain..."

The article appropriately uses hedging language matching the low probability.

## Requirements

- Python 3.10+
- OpenAI API key
- Dependencies: openai, anthropic, httpx, pandas, numpy, matplotlib, seaborn

## Reproducibility

```bash
# Install dependencies
source .venv/bin/activate
uv pip install openai anthropic httpx pandas numpy matplotlib seaborn tqdm datasets

# Set API keys
export OPENAI_API_KEY="your-key"

# Run experiment (seed=42)
python src/run_experiment.py
```

## Full Report

See [REPORT.md](REPORT.md) for complete methodology, results, and analysis.

## Citation

```
@misc{newsfromfuture2026,
  title={News from the Future: Combining LLMs with Prediction Markets},
  author={Claude Code Research},
  year={2026},
  note={Experimental research on probability-conditioned news generation}
}
```
