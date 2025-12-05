# Multimodal Benchmarks

The open evaluation suite for multimodal retrieval systems.

Standard datasets, queries, and relevance judgments for benchmarking retrieval across video, image, audio, and document modalities—particularly in regulated and high-stakes domains.

## Why This Exists

Most retrieval benchmarks assume text-only search on clean web data. Real-world multimodal retrieval is harder:

- **Medical device IFUs** with nested tables, diagrams, and regulatory language
- **SEC filings** with embedded charts, footnotes, and cross-references
- **Warehouse safety videos** requiring temporal understanding
- **Ad creatives** spanning image, video, and audio brand safety signals

This repo provides ground-truth evaluation sets for these verticals—so you can measure what actually matters.

## Benchmarks

| Benchmark | Modalities | Queries | Status |
|-----------|------------|---------|--------|
| `medical-device-ifu` | Document, Image | 500+ | ✅ Available |
| `finance-sec` | Document, Table | 400+ | ✅ Available |
| `ad-safety` | Video, Image, Audio | 300+ | 🚧 Coming Soon |
| `warehouse-safety-video` | Video | 200+ | 🚧 Coming Soon |

## Structure

```
multimodal-benchmarks/
├── medical-device-ifu/
│   ├── data/                    # Source documents
│   ├── queries.json             # Natural language queries
│   ├── relevance_judgments.json # Ground-truth labels
│   ├── metrics.yaml             # Evaluation config
│   └── baselines/               # Published baseline results
├── finance-sec/
├── ad-safety/
└── warehouse-safety-video/
```

## Quick Start

### Install

```bash
pip install multimodal-benchmarks
```

### Evaluate Your Retriever

```python
from multimodal_benchmarks import load_benchmark, evaluate

# Load a benchmark
benchmark = load_benchmark("medical-device-ifu")

# Your retrieval function
def my_retriever(query: str) -> list[str]:
    # Returns ranked list of document IDs
    ...

# Evaluate
results = evaluate(
    retriever=my_retriever,
    benchmark=benchmark,
    metrics=["ndcg@10", "recall@5", "mrr"]
)

print(results)
# {'ndcg@10': 0.72, 'recall@5': 0.68, 'mrr': 0.81}
```

### Run via CLI

```bash
# Evaluate and output results
multimodal-bench eval \
  --benchmark medical-device-ifu \
  --retriever your_config.yaml \
  --output results.json
```

## Metrics

Each benchmark supports:

| Metric | Description |
|--------|-------------|
| `ndcg@k` | Normalized Discounted Cumulative Gain |
| `recall@k` | Proportion of relevant docs in top-k |
| `mrr` | Mean Reciprocal Rank |
| `precision@k` | Precision at cutoff k |
| `map` | Mean Average Precision |

## Contributing a Benchmark

We welcome contributions from researchers and practitioners working on vertical-specific retrieval.

### Requirements

1. **Minimum 100 queries** with relevance judgments
2. **Clear licensing** for underlying data
3. **Reproducible baseline** using at least one open retriever
4. **Documentation** describing the domain and evaluation protocol

### Submission Process

1. Fork this repo
2. Add your benchmark under a new directory
3. Include all required files (see structure above)
4. Open a PR with benchmark description

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## Baselines

We publish baseline results for each benchmark using common retrieval approaches:

| Retriever | medical-device-ifu (nDCG@10) | finance-sec (nDCG@10) |
|-----------|------------------------------|----------------------|
| BM25 | 0.41 | 0.38 |
| ColBERT v2 | 0.58 | 0.52 |
| OpenAI text-embedding-3-large | 0.63 | 0.61 |
| Mixpeek Multimodal | **0.78** | **0.74** |

Full baseline reproduction scripts available in each benchmark's `baselines/` directory.

## Leaderboard

Submit your results to appear on the public leaderboard:

🏆 **[View Leaderboard →](https://mixpeek.com/benchmarks)**

## Citation

If you use these benchmarks in your research:

```bibtex
@misc{multimodal-benchmarks,
  title={Multimodal Benchmarks: Evaluation Suite for Vertical Retrieval Systems},
  author={Mixpeek},
  year={2025},
  url={https://github.com/mixpeek/multimodal-benchmarks}
}
```

## License

Benchmark code: MIT License

Datasets: Individual licensing per benchmark (see each benchmark's `LICENSE` file)

---

Built by [Mixpeek](https://mixpeek.com) — Multimodal AI infrastructure for regulated industries.
