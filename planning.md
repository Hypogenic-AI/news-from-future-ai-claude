# Research Plan: News from the Future

## Motivation & Novelty Assessment

### Why This Research Matters

The ability to generate plausible news articles about future events has significant applications:

1. **Scenario Planning**: Organizations need to envision multiple future scenarios for strategic planning. Automated generation of realistic future news helps stakeholders visualize possible outcomes.

2. **Educational Value**: Understanding how prediction markets estimate probabilities of future events, combined with narrative presentation, helps users grasp probabilistic thinking.

3. **Information Synthesis**: Prediction markets aggregate distributed knowledge effectively (Brier score ~0.121 for crowd median), but numerical probabilities are hard for humans to interpret. News narratives make this information accessible.

4. **Risk Communication**: Presenting potential future outcomes as narratives (rather than raw statistics) improves comprehension and engagement with forecast information.

### Gap in Existing Work (Based on Literature Review)

The literature review reveals:

1. **No direct combination exists**: No existing work combines prediction markets with news generation.
2. **LLM forecasting is established**: ForecastBench shows LLMs achieve Brier ~0.122-0.136, comparable to crowd predictions.
3. **Fake news generation is well-studied**: FakeGPT demonstrates LLMs can generate highly realistic news content.
4. **Future-oriented generation is unexplored**: Most synthetic news research focuses on misinformation about past/present events.

**The gap**: While LLMs can forecast (ForecastBench) and generate realistic news (FakeGPT), no one has combined prediction market probabilities with news generation to create probability-conditioned future news.

### Our Novel Contribution

We propose and evaluate a system that:
1. **Sources probabilities from prediction markets** (Polymarket, Metaculus)
2. **Conditions LLM generation on these probabilities** to create plausible future news
3. **Evaluates calibration** - does the generated content's confidence match input probabilities?
4. **Evaluates plausibility** - do humans find the generated articles believable?

This is the first systematic study of probability-conditioned future news generation.

### Experiment Justification

| Experiment | Why Needed |
|------------|------------|
| **Baseline: Zero-shot generation** | Establishes performance without probability conditioning |
| **Probability-conditioned generation** | Tests main hypothesis - does probability input improve plausibility? |
| **Multi-scenario generation** | Tests ability to generate contrasting scenarios for same question |
| **Human evaluation** | Automated metrics insufficient for news plausibility |
| **Calibration analysis** | Tests if narrative confidence matches input probability |

---

## Research Question

**Primary Question**: Can LLMs generate plausible news articles about future events when conditioned on prediction market probabilities?

**Sub-questions**:
1. Does probability conditioning improve perceived plausibility vs. unconditioned generation?
2. Are generated narratives well-calibrated (e.g., 80% probability events described with appropriate confidence)?
3. What prompting strategies produce the most realistic future news?
4. How do different LLMs compare in this task?

---

## Background and Motivation

Prediction markets aggregate distributed knowledge about future events into probability estimates. However, raw probabilities (e.g., "62% chance of X") are difficult for most people to interpret and engage with. News articles, by contrast, are a familiar format for conveying information.

This research explores whether LLMs can bridge this gap by generating realistic news articles about future events, conditioned on prediction market probabilities. The hypothesis is that such a system could:
- Make probabilistic forecasts more accessible
- Support scenario planning and decision-making
- Demonstrate a novel application of LLM capabilities

---

## Hypothesis Decomposition

### H1: LLMs can generate syntactically and stylistically correct news articles about future events
- **Metric**: News style scoring, grammatical correctness
- **Baseline**: XSum dataset articles

### H2: Probability conditioning improves the plausibility of generated content
- **Metric**: Human plausibility ratings (1-5 scale)
- **Comparison**: Conditioned vs. unconditioned generation

### H3: Generated content reflects input probabilities appropriately (calibration)
- **Metric**: Linguistic confidence markers correlate with input probability
- **Measurement**: Extract hedging language, certainty expressions

### H4: Generated news is distinguishable from real news but not implausibly so
- **Metric**: Human detection rate
- **Target**: 60-80% detection (too easy = implausible, too hard = potential misuse concern)

---

## Proposed Methodology

### Approach

We will build a pipeline that:
1. **Fetches prediction market data** (Polymarket API)
2. **Generates news articles** conditioned on event questions and probabilities
3. **Evaluates quality** through automated metrics and human evaluation

### Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Prediction Market  │────▶│   LLM Generation     │────▶│   Evaluation        │
│  API (Polymarket)   │     │   Pipeline           │     │   Framework         │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
        │                            │                            │
        ▼                            ▼                            ▼
  - Event question            - Prompt engineering          - Automated metrics
  - Current probability       - Multiple models             - Human evaluation
  - Resolution date           - Scenario generation         - Calibration analysis
```

### Experimental Steps

**Step 1: Data Collection**
- Fetch 20-30 prediction market events from Polymarket API
- Select diverse topics: politics, economics, technology, sports
- Record: question, probability, resolution date, context

**Step 2: Baseline Generation (Zero-shot)**
- Generate news articles without probability input
- Use GPT-4.1 and Claude Sonnet 4.5
- Simple prompt: "Write a news article about: [event question]"

**Step 3: Probability-Conditioned Generation**
- Include probability in prompt
- Test multiple prompting strategies:
  - Direct probability mention: "Given 75% probability..."
  - Implicit conditioning: "This outcome is considered likely..."
  - Scenario-based: "Assuming this event occurs..."

**Step 4: Multi-Scenario Generation**
- For each event, generate contrasting articles:
  - High-probability outcome (event happens)
  - Low-probability outcome (event doesn't happen)
- Evaluate internal consistency

**Step 5: Evaluation**
- Automated: Perplexity, lexical diversity, confidence markers
- Human: Plausibility ratings, detectability, calibration perception

### Baselines

| Baseline | Description | Purpose |
|----------|-------------|---------|
| **Zero-shot** | No probability input | Tests if probability helps |
| **Random probability** | Use unrelated probability | Tests specificity of conditioning |
| **Real news** | Actual news articles | Upper bound for style quality |

### Evaluation Metrics

**Automated Metrics**:
- **Perplexity**: Lower = more fluent (measure with small LM)
- **Lexical diversity**: Type-token ratio, vocabulary richness
- **Confidence markers**: Count hedging words vs. certainty expressions
- **News style adherence**: Structure, quote patterns, source references

**Human Evaluation**:
- **Plausibility** (1-5): How believable is this article?
- **News authenticity** (1-5): Does this read like real news?
- **Calibration perception**: Does the confidence level match the claimed probability?
- **Detection**: Can evaluators identify which articles are generated?

### Statistical Analysis Plan

- **Significance level**: α = 0.05
- **Tests**:
  - Paired t-tests for within-model comparisons (conditioned vs. unconditioned)
  - ANOVA for multi-model comparisons
  - Spearman correlation for calibration analysis
- **Effect sizes**: Cohen's d for pairwise comparisons
- **Multiple comparison correction**: Bonferroni where applicable

---

## Expected Outcomes

### Supporting the Hypothesis
- Probability-conditioned articles rated significantly more plausible (p < 0.05)
- Strong correlation (r > 0.5) between input probability and extracted confidence markers
- Humans rate articles as moderately authentic (3-4 out of 5)

### Refuting the Hypothesis
- No significant difference between conditioned and unconditioned generation
- Poor calibration (confidence markers don't correlate with probability)
- Very low plausibility ratings (< 2 out of 5)

---

## Timeline and Milestones

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1 | Environment setup, data collection | 20-30 min |
| Phase 2 | Implement generation pipeline | 45-60 min |
| Phase 3 | Run experiments | 30-45 min |
| Phase 4 | Analysis and evaluation | 30-45 min |
| Phase 5 | Documentation | 20-30 min |

---

## Potential Challenges

| Challenge | Mitigation |
|-----------|------------|
| **API rate limits** | Cache responses, batch requests |
| **Prediction market data quality** | Filter for well-defined events |
| **LLM safety filters** | Use legitimate research framing |
| **Human evaluation resources** | Use LLM-as-judge for initial pass |
| **Calibration measurement** | Use established confidence lexicons |

---

## Success Criteria

1. **Minimum**: Generate 50+ news articles across 20+ events
2. **Target**: Show statistically significant improvement from probability conditioning
3. **Stretch**: Demonstrate good calibration (r > 0.5 between probability and confidence)

---

## Ethical Considerations

- All generated content will be clearly labeled as synthetic
- No intention to deceive or spread misinformation
- Focus on demonstrating capabilities for legitimate use cases (scenario planning, education)
- Discussion of potential misuse and safeguards in final report
