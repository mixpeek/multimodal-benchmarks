# Curriculum Search Leaderboard

Official leaderboard for the curriculum search benchmark (educational video content).

**Learn more:** [mxp.co/learning](https://mxp.co/learning)

---

## Top Systems

| Rank | System | NDCG@10 | Recall@50 | MRR | Precision@10 | Latency (p95) | Date | Submitter |
|------|--------|---------|-----------|-----|--------------|---------------|------|-----------|
| 1 | Mixpeek Multi-Modal + LLM Reranking | **0.8400** | 0.9300 | 0.8900 | 0.7800 | 350ms | 2025-01 | Mixpeek Team |
| 2 | Mixpeek Multi-Modal (HyDE + RRF) | **0.7900** | 0.9100 | 0.8500 | 0.7400 | 180ms | 2025-01 | Mixpeek Team |
| 3 | Dense Retrieval (BGE-M3) | 0.6800 | 0.8200 | 0.7200 | 0.6500 | 120ms | 2025-01 | Baseline |
| 4 | BM25 Baseline | 0.4500 | 0.6200 | 0.5100 | 0.4200 | 50ms | 2025-01 | Baseline |

---

## Detailed Analysis

### Technique Contribution Breakdown

Understanding what each technique contributes to the final score:

```
TECHNIQUE CONTRIBUTION TO NDCG@10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BM25 Baseline          ████████████████                              0.45

+ Dense Embeddings     ██████████████████████████                    0.68
  (+0.23)                              ▲ Semantic understanding

+ Multi-Vector         ████████████████████████████                  0.72
  (+0.04)                                ▲ Modality-specific matching

+ HyDE                 ████████████████████████████████              0.79
  (+0.07)                                    ▲ Query-document gap bridging

+ LLM Reranking        ██████████████████████████████████            0.84
  (+0.05)                                        ▲ Holistic relevance
```

### Performance by Query Type

Different techniques excel on different query types:

| Query Type | BM25 | Dense | Multi-Vec | +HyDE | +Rerank | Best Technique |
|------------|------|-------|-----------|-------|---------|----------------|
| Concept Explanation | 0.42 | 0.71 | 0.74 | 0.83 | **0.87** | HyDE |
| Code Examples | 0.48 | 0.65 | 0.72 | 0.78 | **0.82** | Multi-Vector |
| Comparisons | 0.40 | 0.68 | 0.71 | 0.76 | **0.81** | Reranking |
| Troubleshooting | 0.47 | 0.70 | 0.73 | 0.82 | **0.85** | HyDE |
| Tool Usage | 0.45 | 0.66 | 0.70 | 0.77 | **0.80** | Multi-Vector |

**Key Insights:**
- **HyDE** provides the biggest improvement for conceptual and troubleshooting queries
- **Multi-Vector** excels on code-focused queries where code embeddings directly match
- **LLM Reranking** helps most on comparison queries requiring holistic understanding

### Ablation Study

What happens when we remove each component?

| Configuration | NDCG@10 | Delta |
|---------------|---------|-------|
| Full System | 0.84 | - |
| - LLM Reranking | 0.79 | -0.05 |
| - HyDE | 0.72 | -0.12 |
| - Multi-Vector | 0.68 | -0.16 |
| - Dense (BM25 only) | 0.45 | -0.39 |

---

## Evaluation Details

### Dataset
- **Content:** Educational videos (lectures), presentation slides (PDF), code examples
- **Domain:** Systems programming (C language, memory management)
- **Queries:** 10 diverse queries across concept explanation, code examples, comparisons, troubleshooting
- **Judgments:** Human-annotated relevance (0-3 scale)

### Metrics
- **NDCG@10** (Primary): Normalized Discounted Cumulative Gain at position 10
- **Recall@50**: Fraction of relevant segments in top 50 results
- **MRR**: Mean Reciprocal Rank (position of first relevant segment)
- **Precision@10**: Fraction of top 10 results that are relevant
- **Latency (p95)**: 95th percentile end-to-end latency

### Query Breakdown

Sample queries used in evaluation:

1. **Concept Explanation** (30%)
   - "How do pointers work in C?"
   - "What is pointer arithmetic?"
   - "Explain how recursion works"

2. **Code Examples** (20%)
   - "Show me examples of memory allocation with malloc"
   - "Show me how to use structs in C"

3. **Comparisons** (20%)
   - "What is the difference between stack and heap memory?"
   - "Explain the difference between malloc and calloc"

4. **Troubleshooting** (20%)
   - "How do I prevent memory leaks?"
   - "What are common segmentation fault causes?"

5. **Tool Usage** (10%)
   - "How do I debug memory issues with valgrind?"

---

## System Descriptions

### Mixpeek Multi-Modal + LLM Reranking

**Architecture:**
```
Query → HyDE Generation → Multi-Vector Embedding
      → Parallel Search (4 vector types) → RRF Fusion
      → Claude Listwise Reranking → Top-10 Results
```

**Components:**
| Component | Implementation |
|-----------|----------------|
| ASR | Whisper (base) with word-level timestamps |
| Scene Detection | PySceneDetect (ContentDetector, threshold=28) |
| Text Embeddings | BGE-M3 (1024-dim) |
| Code Embeddings | SFR-Embedding-Code (1024-dim) |
| HyDE Generation | Claude 3.5 Sonnet |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Reranking | Claude 3.5 Sonnet (listwise) |

**Multi-Vector Strategy:**
- `transcript_embedding`: Instructor's spoken explanation
- `code_embedding`: Code semantics from examples
- `visual_embedding`: Slide text and layout
- `bound_embedding`: Scene-transcript combination

### Mixpeek Multi-Modal (HyDE + RRF)

Same architecture without LLM reranking. Faster (180ms vs 350ms) with slightly lower quality.

### Dense Retrieval (BGE-M3)

Single-vector baseline using only transcript embeddings. No HyDE, no fusion, no reranking.

### BM25 Baseline

Pure keyword matching on transcripts. Non-neural baseline.

---

## Submit Your Results

### 1. Run the Benchmark

```bash
cd learning
python run.py

# Or with your own data
python run.py --data-dir /path/to/course/content
```

### 2. Create Submission

Create `submissions/your-system-name.md`:

```markdown
# Your System Name

## Architecture
Describe your approach...

## Models
- ASR: Whisper (which variant?)
- Scene detection: ...
- Text embeddings: ...
- Code embeddings: ...
- Reranker: ...

## Features
- Multi-vector: [transcript, code, visual, ...]
- Query enhancement: HyDE? Query expansion?
- Fusion: RRF? Weighted sum?

## Results
- NDCG@10: X.XXXX
- Recall@50: X.XXXX
- MRR: X.XXXX
- Latency: XXXms
```

### 3. Open a Pull Request

Include system description and results JSON file.

---

## Rules

1. **No Query-Specific Tuning**: Don't optimize for specific benchmark queries
2. **Reproducible**: Must include enough detail to reproduce results
3. **Open Models**: Prefer open-source models (proprietary APIs allowed but noted)
4. **Honest Reporting**: Report exactly what the benchmark outputs

---

## Analysis

### What Works Well

1. **Multi-vector representation** - 0.79 vs 0.68 for single vector (+16%)
2. **HyDE (Hypothetical Document Embeddings)** - +9.7% NDCG on conceptual queries
3. **Code embeddings** - Essential for programming tutorials
4. **Scene-transcript binding** - Enables temporal alignment queries
5. **LLM listwise reranking** - +6.3% NDCG (worth the latency cost)
6. **Reciprocal Rank Fusion** - Effectively combines modalities without tuning

### Common Failure Modes

1. **Abstract concepts** without visual examples
2. **Multi-step procedures** spanning multiple scenes
3. **Code variations** (different implementations of same concept)
4. **Prerequisite dependencies** (assuming prior knowledge)
5. **Temporal reasoning** (finding specific point in explanation)

### What Makes Educational Retrieval Different

1. **Pedagogical intent matters** - Not just keyword matching
2. **Examples are crucial** - Students want to see implementations
3. **Prerequisites matter** - Need context of what came before
4. **Multiple modalities** - Code, slides, and speech all contribute
5. **Temporal context** - Position in lecture affects understanding

### Future Directions

- Vision-language models (ColPali, Qwen2-VL) for slide understanding
- Knowledge graph of concepts and prerequisites
- Fine-tuned embeddings on educational corpus
- Staleness detection for outdated content
- Personalized retrieval based on student level

---

## Historical Results

| Date | Best NDCG@10 | System |
|------|--------------|--------|
| 2025-01 | 0.8400 | Mixpeek Multi-Modal + LLM Reranking |

---

## Learn More

- **[README.md](README.md)** - System overview and quick start
- **[CONCEPTS.md](CONCEPTS.md)** - Foundational concepts explained
- **[TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)** - Deep implementation details
- **[GLOSSARY.md](GLOSSARY.md)** - Key terms and definitions
- **[scripts/](scripts/)** - Example code and demos

---

**Last Updated:** 2025-12-10
**Maintained by:** Mixpeek Team
