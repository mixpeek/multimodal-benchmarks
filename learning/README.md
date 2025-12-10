# Curriculum Search Benchmark

**Achieving Near State-of-the-Art Results in Educational Content Retrieval**

Learn more: [mxp.co/learning](https://mxp.co/learning)

---

## What We Built

A multimodal retrieval system that searches educational video content (lectures, slides, code examples) and achieves **0.84 NDCG@10** - near state-of-the-art performance for educational content retrieval.

| System | NDCG@10 | Improvement |
|--------|---------|-------------|
| BM25 (keyword baseline) | 0.45 | - |
| Dense Retrieval (single vector) | 0.68 | +51% |
| **Our System (multi-vector + HyDE)** | 0.79 | +76% |
| **+ LLM Reranking** | **0.84** | +87% |

---

## How to Use This Module

This learning module is designed for **both technical and non-technical audiences**. Choose your path:

### For Everyone (Start Here)
- **[CONCEPTS.md](CONCEPTS.md)** - Foundational concepts explained simply
- **[GLOSSARY.md](GLOSSARY.md)** - Key terms and definitions

### For Technical Readers
- **[TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)** - Deep dive into implementation
- **[scripts/](scripts/)** - Runnable example code
- **[LEADERBOARD.md](LEADERBOARD.md)** - Detailed benchmark results

### Quick Links
- [Why This Matters](#why-this-matters)
- [The Problem We Solved](#the-problem-we-solved)
- [Our Approach](#our-approach-the-5-key-techniques)
- [Results](#results)
- [Try It Yourself](#try-it-yourself)

---

## Why This Matters

Imagine a student watching a 90-minute lecture on memory management in C. They have a question: *"How do I prevent memory leaks?"*

**Traditional search** would require them to scrub through the entire video or hope the instructor added timestamps.

**Our system** can instantly find the exact 2-minute segment where the instructor explains memory leak prevention, shows the code, and demonstrates the fix - even if they never said the exact words "memory leak."

This technology enables:
- **Smart educational platforms** that answer student questions with precise video clips
- **Corporate training systems** that help employees find relevant content instantly
- **Research tools** that make video knowledge as searchable as text

---

## The Problem We Solved

Educational content is *hard* to search because:

1. **Multiple modalities**: Information is split across what the instructor says, what's on their slides, and the code they show
2. **Terminology mismatch**: Students ask "How do I fix my program crashing?" when the lecture uses "segmentation fault"
3. **Visual context matters**: "As you can see here..." only makes sense with the slide/code on screen
4. **Semantic understanding**: "Memory management" should match content about malloc, free, pointers, even if those words aren't in the query

```
Traditional Search:
Query: "memory leak prevention"
         ↓
Keyword Match: "memory" AND "leak" AND "prevention"
         ↓
Result: Only finds segments with exact words (misses 70%+ of relevant content)

Our System:
Query: "memory leak prevention"
         ↓
Multi-Modal Understanding: What IS a memory leak conceptually?
         ↓
Searches: Transcript (what instructor says)
          Code (malloc without free patterns)
          Slides (memory diagrams)
         ↓
Result: Finds all relevant content, even without exact word matches
```

---

## Our Approach: The 5 Key Techniques

We achieved our results by combining five state-of-the-art techniques:

### 1. Multi-Modal Extraction

We don't just transcribe video - we extract structured information from every modality:

```
Input: Educational Video
         ↓
┌─────────────────────────────────────────┐
│         Multi-Modal Extraction          │
├─────────────┬─────────────┬─────────────┤
│   AUDIO     │   VISUAL    │    CODE     │
│  (Whisper)  │  (Scenes)   │  (Analysis) │
│             │             │             │
│ Transcript  │ Keyframes   │ Functions   │
│ + timestamps│ + detection │ + imports   │
│ + words     │ + context   │ + patterns  │
└─────────────┴─────────────┴─────────────┘
```

**Key Insight**: Each modality contains unique information. The transcript captures *explanations*, the slides capture *structure*, and the code captures *implementation details*.

### 2. Multi-Vector Embeddings

Instead of creating one embedding per content segment, we create **4-5 separate embeddings**:

| Vector Type | What It Captures | Model Used |
|-------------|------------------|------------|
| `transcript_embedding` | Instructor's spoken explanation | BGE-M3 |
| `code_embedding` | Code semantics and patterns | SFR-Embedding-Code |
| `visual_embedding` | Slide text and layout | BGE-M3 |
| `bound_embedding` | Scene-transcript combination | BGE-M3 |

**Why?** Different queries need different representations:
- "Explain pointers" → matches transcript embedding
- "malloc example" → matches code embedding
- "Show me the diagram" → matches visual embedding

### 3. HyDE (Hypothetical Document Embeddings)

The **single biggest improvement** (+11% NDCG) came from HyDE.

```
Traditional:
Query: "How do I prevent memory leaks?"
         ↓
Embed query directly
         ↓
Search (query embedding may not match document style)

With HyDE:
Query: "How do I prevent memory leaks?"
         ↓
Generate hypothetical answer:
   "To prevent memory leaks in C, ensure every malloc() has a
    corresponding free(). Use tools like valgrind to detect leaks.
    Common patterns include: tracking allocations, using RAII-like
    cleanup functions, and always freeing in reverse allocation order."
         ↓
Embed the hypothetical answer
         ↓
Search (embedding matches document style much better!)
```

**Key Insight**: User queries are short and vague. Documents are long and detailed. HyDE bridges this gap by generating what a good answer *would look like*.

### 4. Reciprocal Rank Fusion (RRF)

We search each vector type independently and combine rankings:

```
Query: "malloc free examples"

Transcript Search: [Seg_A, Seg_C, Seg_B, Seg_D]
Code Search:       [Seg_B, Seg_A, Seg_D, Seg_C]
Visual Search:     [Seg_C, Seg_B, Seg_A, Seg_D]
                           ↓
              Reciprocal Rank Fusion
                           ↓
Final Ranking:     [Seg_B, Seg_A, Seg_C, Seg_D]
```

**The Math**: `RRF_score(d) = Σ 1/(k + rank_i(d))` where `k=60`

This formula gives credit to documents that rank highly across multiple searches, without being dominated by any single modality.

### 5. LLM Listwise Reranking

The final boost comes from having an LLM rerank the top results:

```
Top 20 from RRF → Claude Reranking → Final Top 10
```

The LLM considers:
- Does this segment actually answer the question?
- Is there working code with explanations?
- Is the content beginner-friendly?
- Is this the best segment or is another one more complete?

**Result**: +5 points NDCG@10 (0.79 → 0.84)

---

## Results

### Performance Comparison

| System | NDCG@10 | Recall@50 | MRR | P@10 | Latency |
|--------|---------|-----------|-----|------|---------|
| BM25 Baseline | 0.45 | 0.62 | 0.51 | 0.42 | 50ms |
| Dense (BGE-M3) | 0.68 | 0.82 | 0.72 | 0.65 | 120ms |
| Multi-Vector | 0.72 | 0.85 | 0.78 | 0.68 | 140ms |
| + HyDE | 0.79 | 0.91 | 0.85 | 0.74 | 180ms |
| **+ LLM Reranking** | **0.84** | **0.93** | **0.89** | **0.78** | 350ms |

### What Each Technique Contributed

```
Starting Point (BM25):          0.45
                                  │
Dense Retrieval:                +0.23  (semantic understanding)
                                  │
Multi-Vector:                   +0.04  (modality-specific matching)
                                  │
HyDE:                           +0.07  (query-document gap bridging)
                                  │
LLM Reranking:                  +0.05  (holistic relevance assessment)
                                  │
Final:                          0.84
```

### Query Type Breakdown

| Query Type | Example | Our NDCG@10 |
|------------|---------|-------------|
| Concept Explanation | "How do pointers work?" | 0.87 |
| Code Examples | "Show me malloc examples" | 0.82 |
| Comparisons | "Stack vs heap?" | 0.81 |
| Troubleshooting | "Fix memory leaks" | 0.85 |
| Tool Usage | "Using valgrind" | 0.80 |

---

## Try It Yourself

### Quick Demo (No Setup Required)

```bash
# Clone and navigate
cd learning

# Run demo with sample data
python run.py --quick
```

### Full Benchmark

```bash
# Install dependencies
pip install -r ../shared/requirements.txt
brew install ffmpeg poppler  # macOS

# Run full benchmark
python run.py

# View results
cat results/benchmark_results.json | jq '.aggregate_metrics'
```

### With Your Own Content

```bash
# Prepare content
# your-course/
#   ├── video.mp4
#   ├── slides.pdf
#   └── code.zip

# Run benchmark
python run.py --data-dir /path/to/your-course
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: Course Content                         │
│              Video (MP4) + Slides (PDF) + Code (ZIP)                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EXTRACTION LAYER                                │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │   Whisper   │   │ PySceneDetect│   │    Code    │               │
│  │     ASR     │   │   Scenes    │   │   Parser   │               │
│  │  + words    │   │ + keyframes │   │ + analysis │               │
│  └─────────────┘   └─────────────┘   └─────────────┘               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EMBEDDING LAYER                                  │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │              Multi-Vector Embeddings (BGE-M3)              │     │
│  │  transcript_emb │ code_emb │ visual_emb │ bound_emb        │     │
│  └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL LAYER                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │    HyDE     │ → │ Multi-Vector│ → │     RRF     │               │
│  │  Generation │   │   Search    │   │   Fusion    │               │
│  └─────────────┘   └─────────────┘   └─────────────┘               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RERANKING LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │           LLM Listwise Reranking (Claude)                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       OUTPUT: Ranked Results                         │
│                      Top-K relevant segments                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Files

| File | Description |
|------|-------------|
| [run.py](run.py) | Main benchmark script |
| [CONCEPTS.md](CONCEPTS.md) | Foundational concepts for non-technical readers |
| [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) | Deep implementation details |
| [GLOSSARY.md](GLOSSARY.md) | Key terms and definitions |
| [LEADERBOARD.md](LEADERBOARD.md) | Benchmark results and analysis |
| [scripts/](scripts/) | Example code and demos |

---

## Learn More

### Research Papers
- [HyDE Paper](https://arxiv.org/abs/2212.10496) - Hypothetical Document Embeddings
- [BGE-M3 Paper](https://arxiv.org/abs/2402.03216) - Multi-functionality embeddings
- [ColBERT Paper](https://arxiv.org/abs/2004.12832) - Late interaction retrieval

### Implementation
- **Source Code**: [/extractors/curriculum](../../../extractors/curriculum)
- **Benchmark Suite**: [github.com/mixpeek/benchmarks](https://github.com/mixpeek/benchmarks)

### Citation

```bibtex
@misc{mixpeek-curriculum-benchmark,
  title={Curriculum Search Benchmark for Educational Video Content},
  author={Mixpeek},
  year={2025},
  url={https://mxp.co/learning}
}
```

---

Built by [Mixpeek](https://mixpeek.com) - Multimodal AI for regulated industries.
