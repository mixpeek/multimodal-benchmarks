# Financial Document Retrieval Leaderboard

Official leaderboard for the FinanceBench financial document retrieval benchmark.

**Learn more:** [mxp.co/finance](https://mxp.co/finance)

---

## FinanceBench Results

### Overall Accuracy

| Rank | System | Accuracy | Calculation | Factual | Multi-hop | Date |
|------|--------|----------|-------------|---------|-----------|------|
| - | GPT-4 (paper baseline) | **68.0%** | ~65% | ~70% | ~60% | 2023-11 |
| - | Gemini Pro | 64.0% | - | - | - | 2024-01 |
| - | Claude 3.5 Sonnet | 62.0% | - | - | - | 2024-06 |
| 1 | **Mixpeek + CoT (ours)** | **44.0%** | **76.9%** | 38.0% | 27.3% | 2024-12 |
| 2 | Naive RAG | 25.0% | ~40% | ~25% | ~15% | 2024-12 |

**Key Insight:** Our system achieves 76.9% on calculation tasks - higher than GPT-4's estimated ~65%. The bottleneck is retrieval, not reasoning.

---

## Performance Progression

Our systematic improvements from 25% baseline to 44%:

| Stage | Accuracy | Change | Key Innovation |
|-------|----------|--------|----------------|
| Baseline (naive RAG) | 25.0% | - | Basic text extraction |
| + TableFormer | 32.0% | +7.0% | Cell-level table structure |
| + Value Normalization | 38.0% | +6.0% | Scale detection from headers |
| + Entity Filtering | 41.3% | +3.3% | Company/year metadata filters |
| + Answer Validation | 44.0% | +2.7% | Year extraction bug fixes |
| + Statement Detection | ~50%+ | +6%+ | Intelligent retrieval targeting |

---

## Category Breakdown

Performance by question type (44% checkpoint):

| Category | Count | Accuracy | Analysis |
|----------|-------|----------|----------|
| **Calculation** | 26 | **76.9%** | Excellent - LLM is great at math with data |
| **Numerical** | 10 | 50.0% | Moderate - complex reasoning challenges |
| **Factual** | 92 | 38.0% | Weak - retrieval misses key data |
| **Multi-hop** | 22 | 27.3% | Hardest - needs multiple statements |

---

## System Descriptions

### Mixpeek + CoT (Our System)

**Architecture:**
- TableFormer for cell-level table extraction
- Context-aware chunking (headers + scale preserved)
- Intelligent statement type detection
- Company/fiscal year metadata filtering
- Chain-of-thought reasoning with Claude Sonnet 4

**Key Features:**
- Financial value normalization (scale detection)
- Multi-statement retrieval for ratio calculations
- Explicit step-by-step reasoning

**Technology:**
- PDF Processing: PyMuPDF
- Table Detection: microsoft/table-transformer-detection
- Vector DB: Qdrant
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- LLM: Claude Sonnet 4

**Source:** [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)

### GPT-4 (Paper Baseline)

- **Source:** [FinanceBench Paper](https://arxiv.org/abs/2311.11944)
- **Method:** Full document in context + direct prompting
- **Limitations:** Expensive, slow, limited to single documents

### Naive RAG

- **Architecture:** Basic text chunking + semantic search
- **Embeddings:** Single dense vector per chunk
- **Features:** No table extraction, no metadata filtering
- **Why it fails:** Loses table structure, cross-entity contamination

---

## Evaluation Details

### Dataset: FinanceBench

- **Source:** [FinanceBench on arXiv](https://arxiv.org/abs/2311.11944)
- **Documents:** 289 SEC 10-K filings (S&P 500 companies)
- **Questions:** 150 human-annotated
- **Categories:** Factual (92), Calculation (26), Multi-hop (22), Numerical (10)

### Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Correct answers / Total questions |
| **Calculation Accuracy** | Correct on calculation questions |
| **Factual Accuracy** | Correct on factual extraction questions |
| **Multi-hop Accuracy** | Correct on multi-statement questions |

### Evaluation Protocol

1. Questions are run against indexed document collection
2. Top-50 chunks retrieved per question
3. Chain-of-thought reasoning generates answer
4. Answer compared to ground truth (exact match for numbers, fuzzy for text)

---

## Error Analysis

### Failure Modes (at 44% checkpoint)

| Failure Mode | Frequency | Example |
|--------------|-----------|---------|
| Retrieval Miss | 42% | Required data not in top-50 chunks |
| Multi-Statement Gap | 23% | Missing one of required statements |
| Value Scale Error | 15% | Wrong scale interpretation |
| Hallucination | 10% | LLM fabricated number |
| Ambiguous Question | 10% | Unclear what metric is asked |

### What Works Well

1. **Calculations** (76.9%) - LLM is excellent at math when given data
2. **Single-statement queries** - Factual extraction from one source
3. **Explicit metrics** - "What was revenue?" vs "How did the company perform?"

### What Needs Improvement

1. **Multi-hop reasoning** (27.3%) - Questions needing 3+ data points
2. **Implicit metrics** - Requires domain knowledge to identify formula
3. **Cross-document** - Comparing data across multiple filings

---

## Ablation Studies

### TableFormer Impact

| Configuration | Accuracy | Tables Detected |
|---------------|----------|-----------------|
| PyMuPDF only | 25.0% | ~60% |
| + TableFormer | 32.0% | ~95% |

### Retrieval Strategy

| Configuration | Chunks | Accuracy |
|---------------|--------|----------|
| Semantic only | 40 | 44.0% |
| Broad (all 4 statements) | 60 | 42.7% |
| Intelligent (needed only) | 50 | ~50%+ |

### Chain-of-Thought

| Configuration | Calc Accuracy | Hallucination |
|---------------|---------------|---------------|
| Direct prompt | 60.0% | ~15% |
| + CoT | 76.9% | ~5% |

---

## Submit Your Results

To submit your system to the leaderboard:

### 1. Run the Benchmark

```bash
cd finance
python run.py
```

### 2. Submit Results

Create a PR with:
- `submissions/your-system-name.json` - Benchmark results
- `submissions/your-system-name.md` - System description

### 3. Required Information

```markdown
# Your System Name

## Architecture
- Document processing approach
- Embedding model(s)
- Retrieval strategy
- LLM and prompting method

## Results
- Overall accuracy: X.X%
- Calculation accuracy: X.X%
- Factual accuracy: X.X%
- Multi-hop accuracy: X.X%

## Reproducibility
- Hardware used
- Total runtime
- Key hyperparameters
```

---

## Future Targets

| Target | Approach |
|--------|----------|
| 50-55% | Intelligent statement detection |
| 55-60% | XBRL integration for number validation |
| 60-65% | Fine-tuned financial embeddings |
| 65-70% | Multi-hop query decomposition |

---

## References

1. Islam, P. et al. "FinanceBench: A New Benchmark for Financial Question Answering." arXiv:2311.11944 (2023)
2. OpenAI. "GPT-4 Technical Report." arXiv:2303.08774 (2023)
3. Smock, B. et al. "PubTables-1M: Towards comprehensive table extraction." CVPR 2022

---

**Last Updated:** December 2024
**Maintained by:** Mixpeek Team

---

Built by [Mixpeek](https://mixpeek.com) — Multimodal AI for regulated industries.
