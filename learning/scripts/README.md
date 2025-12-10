# Example Scripts

Standalone scripts demonstrating key concepts from our multimodal retrieval system.

## Scripts

| Script | What It Demonstrates | Prerequisites |
|--------|---------------------|---------------|
| [01_embeddings_intro.py](01_embeddings_intro.py) | How embeddings represent meaning | `sentence-transformers` |
| [02_similarity_search.py](02_similarity_search.py) | Basic semantic search | `numpy`, `sentence-transformers` |
| [03_hyde_demo.py](03_hyde_demo.py) | HyDE query enhancement | `sentence-transformers` |
| [04_multi_vector.py](04_multi_vector.py) | Multi-vector retrieval | `numpy`, `sentence-transformers` |
| [05_rrf_fusion.py](05_rrf_fusion.py) | Reciprocal Rank Fusion | `numpy` |
| [06_evaluation_metrics.py](06_evaluation_metrics.py) | NDCG, MRR, Recall calculation | `numpy` |

## Quick Start

```bash
# Install dependencies
pip install sentence-transformers numpy

# Run any script
python 01_embeddings_intro.py
```

## Learning Path

1. Start with `01_embeddings_intro.py` to understand embeddings
2. Move to `02_similarity_search.py` to see semantic search in action
3. Try `03_hyde_demo.py` to see how HyDE improves results
4. Explore `04_multi_vector.py` for multi-modal concepts
5. Learn fusion with `05_rrf_fusion.py`
6. Understand evaluation with `06_evaluation_metrics.py`

Each script is self-contained with detailed comments explaining what's happening.
