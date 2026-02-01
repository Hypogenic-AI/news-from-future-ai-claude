# Downloaded Datasets

This directory contains datasets for the "News from the Future" research project. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: XSum (BBC News Summarization)

### Overview
- **Source**: HuggingFace (EdinburghNLP/xsum)
- **Size**: 226,711 samples total (204,045 train, 11,332 validation, 11,334 test)
- **Format**: HuggingFace Dataset
- **Task**: News article summarization (extreme summarization)
- **License**: See HuggingFace page

### Download Instructions

```python
from datasets import load_dataset
xsum = load_dataset("EdinburghNLP/xsum")
xsum.save_to_disk("datasets/xsum")
```

### Loading the Dataset

```python
from datasets import load_from_disk
xsum = load_from_disk("datasets/xsum")
```

### Sample Data

Example from training set:
```json
{
  "document": "The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed...",
  "summary": "Clean-up operations are continuing across the Scottish Borders and Dumfries and Galloway after flooding caused by Storm Frank.",
  "id": "35232142"
}
```

### Notes
- Contains BBC news articles with one-sentence extreme summaries
- Average article length: 431 words (~20 sentences)
- Average summary length: 23 words
- Useful for: training news generation/summarization models, understanding news structure

---

## Dataset 2: Fake News English

### Overview
- **Source**: HuggingFace (community-datasets/fake_news_english)
- **Size**: 492 samples
- **Format**: HuggingFace Dataset
- **Task**: Fake news detection
- **Features**: article URL, fake/satire label, rebutting article URL

### Download Instructions

```python
from datasets import load_dataset
fake_news = load_dataset("community-datasets/fake_news_english")
fake_news.save_to_disk("datasets/fake_news_english")
```

### Loading the Dataset

```python
from datasets import load_from_disk
fake_news = load_from_disk("datasets/fake_news_english")
```

### Notes
- Contains URLs to fake news articles with classification labels
- Includes rebutting article URLs for verification
- Small dataset, good for validation/testing

---

## Recommended Additional Datasets

For comprehensive research on "News from the Future", consider also downloading:

### 1. LIAR Dataset (Fake News Benchmark)
- **Source**: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- **Task**: Political statement fact-checking
- **Size**: 12.8K labeled statements from PolitiFact
- **Labels**: pants-fire, false, barely-true, half-true, mostly-true, true

### 2. FakeNewsNet (Mentioned in FakeGPT paper)
- **Source**: https://github.com/KaiDMML/FakeNewsNet
- **Task**: Fake news detection with social context
- **Includes**: GossipCop and PolitiFact datasets

### 3. ForecastBench Dataset
- **Source**: https://github.com/forecastingresearch/forecastbench-datasets
- **Task**: LLM forecasting evaluation
- **Contains**: Forecasting questions, human/LLM predictions, resolutions

### 4. OpenEPBench (Open-Ended Event Prediction)
- **Source**: Contact authors of arXiv:2408.06578
- **Task**: Open-ended future event prediction
- **Languages**: Chinese and English

### 5. Metaculus/Polymarket Data
- **Source**: API access from respective platforms
- **Task**: Prediction market data
- **Useful for**: Training forecasting models

---

## Data Usage Notes

1. **XSum**: Use for understanding news article structure and training summarization components
2. **Fake News English**: Use for validation of generated content authenticity
3. **ForecastBench**: Essential for evaluating forecasting accuracy
4. **Prediction Market Data**: Core for the "prediction markets + LLMs" hypothesis

## Citation

If using these datasets, please cite the original sources:
- XSum: Narayan et al., 2018 "Don't Give Me the Details, Just the Summary!"
- ForecastBench: Karger et al., 2024 "ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities" (ICLR 2025)
