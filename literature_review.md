# Literature Review: News from the Future

## Research Hypothesis
A website that combines large language models (LLMs) with prediction markets can generate plausible news articles about future events.

---

## 1. Research Area Overview

This research lies at the intersection of three rapidly evolving fields:
1. **LLM-based text generation** - particularly news article generation
2. **Forecasting and prediction markets** - probabilistic prediction of future events
3. **Synthetic content generation** - generating realistic but hypothetical content

The hypothesis combines LLMs' generative capabilities with prediction markets' probability estimates to create plausible future news. This is a novel application that draws from recent advances in each area.

---

## 2. Key Papers

### 2.1 LLM Forecasting Capabilities

#### ForecastBench (Karger et al., 2024) - ICLR 2025
- **arXiv**: 2409.19839
- **Key Contribution**: Dynamic benchmark for evaluating LLM forecasting against human superforecasters
- **Methodology**: 1,000 forecasting questions from prediction markets (Metaculus, Polymarket, Manifold) and datasets (ACLED, FRED, Wikipedia)
- **Key Results**:
  - Superforecasters achieve Brier score of 0.096
  - Best LLM (Claude 3.5 Sonnet with scratchpad + freeze values): 0.122
  - LLMs perform significantly worse than expert forecasters (p < 0.001)
  - Linear correlation between general LLM capability and forecasting accuracy (r = -0.68)
- **Relevance**: Establishes baseline for LLM forecasting accuracy; shows LLMs can approximate crowd predictions but not expert-level accuracy
- **Code**: https://github.com/forecastingresearch/forecastbench

#### AI-Augmented Predictions (Schoenegger et al., 2024)
- **arXiv**: 2402.07862
- **Key Contribution**: LLM assistants improve human forecasting accuracy by 24-41%
- **Methodology**: N=991 participants with two types of LLM assistants (superforecasting vs. noisy)
- **Key Results**:
  - Even noisy LLM assistants improved human accuracy by 24%
  - Superforecasting-prompted assistants improved accuracy by 41%
  - Human-AI collaboration outperforms either alone
- **Relevance**: Demonstrates that LLMs can augment human judgment for forecasting, suggesting hybrid approaches

#### OpenEP: Open-Ended Future Event Prediction (Guan et al., 2024)
- **arXiv**: 2408.06578
- **Key Contribution**: Framework for predicting future events in open-ended (free-form) format
- **Methodology**:
  - Questions from 7 perspectives: time, location, event development, outcome, impact, response, other
  - Uses hot topics from Zhihu (Chinese) and Google News (English)
  - 15-day prediction window
  - LLM-based evaluation metrics (accuracy, completeness, relevance, specificity, reasonableness)
- **Key Results**: Accurately predicting future events in open-ended settings remains challenging for LLMs
- **Relevance**: Directly relevant - provides framework for generating predictions about future events

### 2.2 Fake News and Synthetic News Generation

#### FakeGPT (Huang et al., 2023)
- **arXiv**: 2310.05046
- **Key Contribution**: Comprehensive study of ChatGPT for fake news generation, explanation, and detection
- **Methodology**:
  - Four prompting methods for generation: (a) altering text meaning, (b) inventing stories, (c) creating imaginary text, (d) multiple prompts
  - Nine features characterizing fake news identified
  - Reason-aware prompting for detection
- **Key Results**:
  - Humans achieved only 54.8% accuracy in identifying LLM-generated fake news
  - Self-evaluation: ChatGPT detected 72.5% of its own generated fake news
  - Multiple prompts method most effective at bypassing safety filters
  - Nine features of fake news: emotional bias, lack of evidence, conflicting facts, informal language, insufficient support, lack of context, misinterpretation, oversimplification, doctored media
- **Key Datasets Used**: COVID-19, FakeNewsAMT, Celebrity, LIAR, Weibo21, Kaggle, Chinese Rumor, FakeNewsNet, Twitter15&16
- **Relevance**: Provides methodology for generating realistic news content; identifies quality features to optimize

#### Synthetic News Generation for Fake News Classification (2025)
- **arXiv**: 2503.24206
- **Relevance**: Recent work on synthetic news generation for training fake news detectors

#### On LLMs-Driven Synthetic Data Generation (Long et al., 2024)
- **arXiv**: 2406.15126
- **Key Contribution**: Comprehensive survey on LLM-driven synthetic data generation, curation, and evaluation
- **Relevance**: Provides best practices for generating high-quality synthetic content

### 2.3 Controllable Text Generation

#### Air-Decoding (Wang et al., 2023)
- **arXiv**: 2310.14892
- **Key Contribution**: Attribute distribution reconstruction for decoding-time controllable text generation
- **Relevance**: Methods for controlling attributes (like news style, topic, tone) during generation

#### Controllable Stylistic Text Generation (2025)
- **arXiv**: 2510.06386
- **Key Contribution**: Train-time attribute-regularized diffusion for controllable generation
- **Relevance**: Techniques for ensuring generated news matches target style

#### Controllable Mixed-Initiative Dialogue Generation (2023)
- **arXiv**: 2305.04147
- **Relevance**: Prompting methods for controllable generation

### 2.4 Hallucination and Factuality

#### Hallucination Detection with Small Language Models (2025)
- **arXiv**: 2506.22486
- **Key Contribution**: Detecting hallucinated content in LLM outputs
- **Relevance**: Critical for ensuring generated "future news" remains plausible rather than fantastical

#### LLM-Generated Text Detection Survey (2023)
- **arXiv**: 2310.14724
- **Key Contribution**: Survey on detecting LLM-generated content
- **Relevance**: Understanding what makes generated content detectable vs. realistic

---

## 3. Common Methodologies

### 3.1 Prompting Strategies
- **Zero-shot**: Direct prompting without examples
- **Scratchpad**: Chain-of-thought reasoning for forecasting
- **Multiple prompts**: Progressive prompting to bypass safety filters (FakeGPT)
- **Reason-aware prompts**: Including feature descriptions to guide generation/detection

### 3.2 Retrieval Augmentation
- News API integration for current events
- Knowledge base retrieval for factual grounding
- Historical event retrieval for pattern recognition (OpenEP)

### 3.3 Evaluation Metrics
- **Forecasting**: Brier score (lower is better, 0.25 = uninformed)
- **Text Quality**: ROUGE scores, BLEU, perplexity
- **Fake News**: Accuracy, F1, human evaluation
- **Open-Ended**: LLM-based evaluation on accuracy, completeness, relevance, specificity, reasonableness

---

## 4. Standard Baselines

| Baseline | Description | Typical Performance |
|----------|-------------|---------------------|
| Superforecasters | Expert human forecasters | Brier ~0.096 |
| Crowd median | Aggregated public forecasts | Brier ~0.121 |
| GPT-4/Claude 3.5 | Frontier LLMs with RAG | Brier ~0.122-0.136 |
| LLM ensemble | Multiple models aggregated | Similar to single best |
| Random/Uninformed | 50% probability | Brier = 0.25 |

---

## 5. Datasets in the Literature

| Dataset | Task | Source | Size |
|---------|------|--------|------|
| ForecastBench | Forecasting | Metaculus, Polymarket, etc. | 1000+ questions |
| LIAR | Fake news detection | PolitiFact | 12.8K statements |
| FakeNewsNet | Fake news + social | GossipCop, PolitiFact | Large |
| XSum | News summarization | BBC | 226K articles |
| COVID-19 | Fake news | Various | Medium |
| OpenEPBench | Event prediction | Zhihu, Google News | Growing |

---

## 6. Gaps and Opportunities

### 6.1 Current Gaps
1. **No direct combination**: No existing work combines prediction markets with news generation
2. **Future-oriented generation**: Most fake news research focuses on misinformation about past/present, not future
3. **Plausibility calibration**: No framework for calibrating "plausibility" of generated future scenarios
4. **Dynamic content**: Existing work is static; no system updates predictions based on new information

### 6.2 Research Opportunities
1. **Probability-conditioned generation**: Generate news conditional on probability estimates from markets
2. **Scenario branching**: Multiple news versions for different outcome probabilities
3. **Temporal consistency**: Ensuring generated news is temporally consistent with prediction timelines
4. **Uncertainty communication**: Incorporating uncertainty into news narrative

---

## 7. Recommendations for Experiment Design

### 7.1 Recommended Datasets
1. **ForecastBench** (primary): Use for sourcing prediction market questions and probabilities
2. **XSum**: For fine-tuning news generation style
3. **LIAR/FakeNewsNet**: For evaluating realism of generated content

### 7.2 Recommended Baselines
1. **LLM-only generation**: Generate news without probability conditioning
2. **Human-written hypothetical news**: For quality comparison
3. **Random probability assignment**: To test if probability information helps

### 7.3 Recommended Metrics
1. **Plausibility**: Human evaluation of how plausible the news seems
2. **Consistency**: Does news match the probability (e.g., high-probability events described confidently)?
3. **Calibration**: Across many generations, do described outcomes match input probabilities?
4. **Detectability**: Can humans/detectors identify content as synthetic?

### 7.4 Methodological Considerations
1. **Prompt engineering**: Start with multiple prompts approach from FakeGPT
2. **Retrieval augmentation**: Include current context (similar to ForecastBench methodology)
3. **Probability injection**: Condition generation on prediction market probabilities
4. **Evaluation pipeline**: Combine automated metrics with human evaluation

---

## 8. Key Technical Challenges

1. **Hallucination control**: Preventing LLMs from generating implausible details
2. **Probability translation**: Converting numerical probabilities to appropriate narrative confidence
3. **Temporal grounding**: Ensuring future dates and timelines are consistent
4. **Style matching**: Generating content that matches real news article style
5. **Ethical considerations**: Clearly labeling synthetic content, preventing misuse

---

## 9. Summary

The "News from the Future" research combines three mature research areas in a novel way:

- **LLM forecasting** shows LLMs can approximate (but not match) human expert predictions
- **Fake news generation** demonstrates LLMs can produce highly realistic news content
- **Controllable generation** provides techniques for conditioning output on specific attributes

The key innovation would be using prediction market probabilities to condition news generation, creating probabilistically-calibrated hypothetical news about future events. This has not been attempted in existing literature and represents a genuine research contribution.

**Critical success factors**:
1. Using ForecastBench methodology for probability sourcing
2. Applying FakeGPT-style prompting for realistic generation
3. Implementing strong evaluation including human judgment
4. Ensuring generated content is clearly labeled as hypothetical
