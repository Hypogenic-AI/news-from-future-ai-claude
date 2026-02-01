# Resources Catalog

## Summary

This document catalogs all resources gathered for the "News from the Future" research project, including papers, datasets, and code repositories.

**Research Hypothesis**: A website that combines large language models (LLMs) with prediction markets can generate plausible news articles about future events.

---

## Papers

**Total papers downloaded**: 37

### Core Papers (Highly Relevant)

| Title | Authors | Year | File | Key Relevance |
|-------|---------|------|------|---------------|
| FakeGPT: Fake News Generation, Explanation and Detection of Large Language Models | Huang et al. | 2023 | papers/2310.05046_FakeGPT_*.pdf | Methodology for LLM news generation |
| ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities | Karger et al. | 2024 | papers/2409.19839_ForecastBench_*.pdf | LLM forecasting benchmark, prediction market data |
| OpenEP: Open-Ended Future Event Prediction | Guan et al. | 2024 | papers/2408.06578_OpenEP_*.pdf | Future event prediction framework |
| AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Accuracy | Schoenegger et al. | 2024 | papers/2402.07862_AI-Augmented_*.pdf | Human-AI forecasting collaboration |
| On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey | Long et al. | 2024 | papers/2406.15126_On_LLMs-Driven_*.pdf | Synthetic content generation survey |
| Wisdom of the Crowds in Forecasting | 2025 | papers/2502.08205_Wisdom_*.pdf | Forecast aggregation methods |
| A Survey on LLM-Generated Text Detection | 2023 | papers/2310.14724_A_Survey_on_LLM-Generated_*.pdf | Detection of synthetic content |

### Controllable Generation Papers

| Title | Authors | Year | File | Key Relevance |
|-------|---------|------|------|---------------|
| Controllable Stylistic Text Generation | 2025 | papers/2510.06386_Controllable_Stylistic_*.pdf | Style-controlled generation |
| Air-Decoding: Attribute Distribution Reconstruction | Wang et al. | 2023 | papers/2310.14892_Air-Decoding_*.pdf | Decoding-time control |
| Controllable Mixed-Initiative Dialogue Generation | 2023 | papers/2305.04147_Controllable_Mixed-Initiative_*.pdf | Prompting for control |
| Towards Attribute-Entangled Controllable Text Generation | 2022 | papers/2210.16557_Towards_Attribute-Entangled_*.pdf | Attribute control |

### Fake News Detection Papers

| Title | Authors | Year | File | Key Relevance |
|-------|---------|------|------|---------------|
| On the Coherence of Fake News Articles | Singh et al. | 2019 | papers/1906.11126_On_the_Coherence_*.pdf | Fake news characteristics |
| A Benchmark Study of ML Models for Online Fake News Detection | Khan et al. | 2019 | papers/1905.04749_A_Benchmark_Study_*.pdf | Detection baselines |
| Exploring Text-transformers in COVID-19 Fake News Detection | 2021 | papers/2101.02359_Exploring_Text-transformers_*.pdf | Transformer-based detection |

### LLM Reasoning Papers

| Title | Authors | Year | File | Key Relevance |
|-------|---------|------|------|---------------|
| Making Large Language Models Better Reasoners with Alignment | 2023 | papers/2309.02144_Making_Large_*.pdf | Improving LLM reasoning |
| Hallucination Detection with Small Language Models | 2025 | papers/2506.22486_Hallucination_*.pdf | Hallucination control |
| Response: Emergent analogical reasoning | 2023 | papers/2308.16118_Response_Emergent_*.pdf | LLM reasoning capabilities |

See `papers/README.md` for complete paper catalog.

---

## Datasets

**Total datasets downloaded**: 2 (locally available)

### Downloaded Datasets

| Name | Source | Size | Task | Location | Status |
|------|--------|------|------|----------|--------|
| XSum | HuggingFace (EdinburghNLP/xsum) | 226K articles | News summarization | datasets/xsum/ | Downloaded |
| Fake News English | HuggingFace (community-datasets) | 492 samples | Fake news detection | datasets/fake_news_english/ | Downloaded |

### Recommended Additional Datasets

| Name | Source | Size | Task | Notes |
|------|--------|------|------|-------|
| LIAR | cs.ucsb.edu | 12.8K | Political fact-checking | Standard benchmark |
| FakeNewsNet | GitHub (KaiDMML) | Large | Fake news + social | Includes GossipCop |
| ForecastBench Dataset | GitHub | Growing | Forecasting evaluation | Essential for this research |
| OpenEPBench | Contact authors | N/A | Event prediction | Chinese + English |

See `datasets/README.md` for download instructions.

---

## Code Repositories

**Total repositories cloned**: 3

| Name | URL | Purpose | Location | Key Features |
|------|-----|---------|----------|--------------|
| ForecastBench | github.com/forecastingresearch/forecastbench | LLM forecasting benchmark | code/forecastbench/ | Question bank, evaluation, leaderboard |
| LLM Forecasting | github.com/dannyallover/llm_forecasting | LLM forecasting experiments | code/llm_forecasting/ | Prompting strategies, RAG |
| LLM Misinformation | github.com/llm-misinformation/llm-misinformation | Misinformation generation/detection | code/llm-misinformation/ | LLMFake dataset, generation methods |

See `code/README.md` for repository details.

---

## Resource Gathering Notes

### Search Strategy

1. **arXiv searches**: Multiple queries covering LLM forecasting, prediction markets, news generation, fake news, controllable generation
2. **Web searches**: Targeted searches for HuggingFace datasets, GitHub repositories, and recent research
3. **Paper-finder**: Attempted (service unavailable), fell back to manual arXiv API search

### Selection Criteria

Papers selected based on:
- Direct relevance to forecasting + LLMs
- Relevance to news/text generation
- Recency (2023-2025 preferred)
- Citation count and venue quality
- Code/data availability

### Challenges Encountered

1. **Paper-finder service**: Not running, required manual search
2. **Some HuggingFace datasets**: Script-based loading deprecated, required alternative sources
3. **Synthetic news datasets**: Limited availability; most fake news datasets focus on detection

### Gaps and Workarounds

- **No direct "future news generation" dataset**: Will need to create evaluation framework
- **ForecastBench dataset**: Available via GitHub, not pre-downloaded due to size
- **Prediction market APIs**: Will need direct API access for live probability data

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **ForecastBench**: Use for forecasting questions and probability estimates
- **XSum**: For fine-tuning news generation style
- **Custom evaluation set**: Create human evaluation framework

### 2. Baseline Methods
- Zero-shot LLM generation (no probability input)
- Probability-conditioned generation
- Human-written hypothetical news
- Random probability baseline

### 3. Evaluation Metrics
- **Plausibility**: Human ratings (1-5 scale)
- **Calibration**: Do generated outcomes match input probabilities?
- **News quality**: ROUGE, perplexity, coherence
- **Detectability**: Can detectors identify as synthetic?

### 4. Code to Adapt/Reuse
- **ForecastBench**: Question generation, probability sourcing
- **LLM Forecasting**: Prompting strategies, RAG pipeline
- **LLM Misinformation**: Generation methods, evaluation

---

## File Structure

```
project/
├── papers/
│   ├── *.pdf (37 papers)
│   ├── pages/ (chunked PDFs for reading)
│   ├── arxiv_search_results.json
│   └── selected_papers.json
├── datasets/
│   ├── xsum/
│   ├── fake_news_english/
│   ├── .gitignore
│   └── README.md
├── code/
│   ├── forecastbench/
│   ├── llm_forecasting/
│   └── llm-misinformation/
├── literature_review.md
├── resources.md
└── pyproject.toml
```

---

## Quick Start for Experiment Runner

### Environment Setup
```bash
source .venv/bin/activate
```

### Load Datasets
```python
from datasets import load_from_disk

# News summarization dataset
xsum = load_from_disk("datasets/xsum")

# Fake news dataset
fake_news = load_from_disk("datasets/fake_news_english")
```

### Access ForecastBench
```python
# Clone ForecastBench datasets
# git clone https://github.com/forecastingresearch/forecastbench-datasets.git

# Or use API directly (see code/forecastbench for examples)
```

### Key Papers to Reference
1. `papers/2310.05046_FakeGPT_*.pdf` - Generation methodology
2. `papers/2409.19839_ForecastBench_*.pdf` - Forecasting benchmark
3. `papers/2408.06578_OpenEP_*.pdf` - Event prediction framework
