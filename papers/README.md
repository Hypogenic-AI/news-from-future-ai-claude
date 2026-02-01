# Downloaded Papers

Total papers downloaded: 37

## Core Papers (Most Relevant)

These papers are essential reading for the "News from the Future" research:

### LLM Forecasting
1. **ForecastBench** (2409.19839) - Dynamic benchmark for LLM forecasting, ICLR 2025
2. **AI-Augmented Predictions** (2402.07862) - LLM assistants improve human forecasting
3. **OpenEP** (2408.06578) - Open-ended future event prediction

### News Generation & Fake News
4. **FakeGPT** (2310.05046) - Fake news generation, explanation, and detection with LLMs
5. **On LLMs-Driven Synthetic Data Generation** (2406.15126) - Survey on synthetic generation
6. **Survey on LLM-Generated Text Detection** (2310.14724) - Detection methods

### Controllable Generation
7. **Air-Decoding** (2310.14892) - Attribute-controlled generation
8. **Controllable Stylistic Text Generation** (2510.06386) - Style control

## Paper List

1. **FakeGPT: Fake News Generation, Explanation and Detection of Large Lang...**
   - arXiv: 2310.05046
   - Year: 2023
   - Categories: cs.CL

2. **Synthetic News Generation for Fake News Classification...**
   - arXiv: 2503.24206
   - Year: 2025
   - Categories: cs.CL

3. **Wisdom of the Crowds in Forecasting: Forecast Summarization for Suppor...**
   - arXiv: 2502.08205
   - Year: 2025
   - Categories: cs.LG, cs.CL, cs.IR

4. **AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Acc...**
   - arXiv: 2402.07862
   - Year: 2024
   - Categories: cs.CY, cs.AI, cs.CL

5. **On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A ...**
   - arXiv: 2406.15126
   - Year: 2024
   - Categories: cs.CL

6. **V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning...**
   - arXiv: 2503.11495
   - Year: 2025
   - Categories: cs.CV

7. **Visual Generation Unlocks Human-Like Reasoning through Multimodal Worl...**
   - arXiv: 2601.19834
   - Year: 2026
   - Categories: cs.AI

8. **OpenEP: Open-Ended Future Event Prediction...**
   - arXiv: 2408.06578
   - Year: 2024
   - Categories: cs.CL

9. **Controllable Stylistic Text Generation with Train-Time Attribute-Regul...**
   - arXiv: 2510.06386
   - Year: 2025
   - Categories: cs.CL

10. **Analogy Generation by Prompting Large Language Models: A Case Study of...**
   - arXiv: 2210.04186
   - Year: 2022
   - Categories: cs.CL, cs.AI

11. **Proceedings of the First International Workshop on Next-Generation Lan...**
   - arXiv: 2410.05339
   - Year: 2024
   - Categories: cs.AI

12. **Proceedings of the Second International Workshop on Next-Generation La...**
   - arXiv: 2511.09575
   - Year: 2025
   - Categories: cs.AI

13. **Air-Decoding: Attribute Distribution Reconstruction for Decoding-Time ...**
   - arXiv: 2310.14892
   - Year: 2023
   - Categories: cs.CL

14. **PB-LLM: Partially Binarized Large Language Models...**
   - arXiv: 2310.00034
   - Year: 2023
   - Categories: cs.LG, cs.AI, cs.CL

15. **Northeastern Uni at Multilingual Counterspeech Generation: Enhancing C...**
   - arXiv: 2412.15453
   - Year: 2024
   - Categories: cs.CL, cs.AI

16. **Hierarchical Multi-agent Large Language Model Reasoning for Autonomous...**
   - arXiv: 2512.13930
   - Year: 2025
   - Categories: cond-mat.mtrl-sci, cs.AI, cs.CL

17. **Towards Attribute-Entangled Controllable Text Generation: A Pilot Stud...**
   - arXiv: 2210.16557
   - Year: 2022
   - Categories: cs.CL

18. **The Future of Scientific Publishing: Automated Article Generation...**
   - arXiv: 2404.17586
   - Year: 2024
   - Categories: cs.HC, cs.AI, cs.ET

19. **ARS: Adaptive Reasoning Suppression for Efficient Large Reasoning Lang...**
   - arXiv: 2510.00071
   - Year: 2025
   - Categories: cs.AI, cs.CL

20. **Large Language Models Reasoning Abilities Under Non-Ideal Conditions A...**
   - arXiv: 2508.04848
   - Year: 2025
   - Categories: cs.AI

21. **Learning Adaptive Parallel Reasoning with Language Models...**
   - arXiv: 2504.15466
   - Year: 2025
   - Categories: cs.AI, cs.CL

22. **Response: Emergent analogical reasoning in large language models...**
   - arXiv: 2308.16118
   - Year: 2023
   - Categories: cs.CL, cs.AI

23. **Demystifying AI Platform Design for Distributed Inference of Next-Gene...**
   - arXiv: 2406.01698
   - Year: 2024
   - Categories: cs.AR, cs.AI, cs.DC

24. **Exploring Text-transformers in AAAI 2021 Shared Task: COVID-19 Fake Ne...**
   - arXiv: 2101.02359
   - Year: 2021
   - Categories: cs.CL, cs.AI, cs.LG

25. **Generative World Modelling for Humanoids: 1X World Model Challenge Tec...**
   - arXiv: 2510.07092
   - Year: 2025
   - Categories: cs.LG, cs.AI, cs.RO

26. **PediatricsGPT: Large Language Models as Chinese Medical Assistants for...**
   - arXiv: 2405.19266
   - Year: 2024
   - Categories: cs.CL

27. **ALERT-Transformer: Bridging Asynchronous and Synchronous Machine Learn...**
   - arXiv: 2402.01393
   - Year: 2024
   - Categories: cs.CV, cs.LG, cs.NE

28. **A Survey on LLM-Generated Text Detection: Necessity, Methods, and Futu...**
   - arXiv: 2310.14724
   - Year: 2023
   - Categories: cs.CL, cs.AI

29. **Efficient (Soft) Q-Learning for Text Generation with Limited Good Data...**
   - arXiv: 2106.07704
   - Year: 2021
   - Categories: cs.CL, cs.LG

30. **OPT-R: Exploring the Role of Explanations in Finetuning and Prompting ...**
   - arXiv: 2305.12001
   - Year: 2023
   - Categories: cs.CL


## Usage

Papers are organized by arXiv ID. To read a specific paper:

```python
# Use the PDF chunker for large papers
python .claude/skills/paper-finder/scripts/pdf_chunker.py papers/<paper>.pdf --pages-per-chunk 3
```

Then read chunk files in `papers/pages/`.

## Citations

See individual paper files for citation information. Key papers to cite:

```bibtex
@inproceedings{karger2025forecastbench,
  title={ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities},
  author={Karger, Ezra and Bastani, Houtan and Chen, Yueh-Han and others},
  booktitle={ICLR},
  year={2025}
}

@article{huang2023fakegpt,
  title={FakeGPT: Fake News Generation, Explanation and Detection of Large Language Models},
  author={Huang, Yue and Shu, Kai and Yu, Philip S and Sun, Lichao},
  journal={arXiv preprint arXiv:2310.05046},
  year={2023}
}
```
