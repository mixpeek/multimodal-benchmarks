# Glossary of Terms

A comprehensive reference of technical terms used in this benchmark and in multimodal retrieval systems.

---

## A

### ASR (Automatic Speech Recognition)
The technology that converts spoken language into text. In our system, we use **Whisper** for ASR.

**Example**: Converting "Hello, today we'll learn about pointers" from audio into text.

### AST (Abstract Syntax Tree)
A tree representation of the structure of source code. Used to understand code semantics beyond just the text.

**Example**: The code `int x = 5;` becomes a tree with nodes for "declaration", "type=int", "name=x", "value=5".

---

## B

### BGE-M3
A state-of-the-art embedding model from BAAI that produces three types of embeddings in one pass: dense, sparse, and ColBERT vectors.

**Why it matters**: Most versatile open-source embedding model for text retrieval.

### Bi-encoder
An architecture where queries and documents are encoded separately, then compared using similarity (like cosine). Fast but less accurate than cross-encoders.

**Trade-off**: Fast (can pre-compute document embeddings) but can miss nuanced relevance.

### Bound Embedding
An embedding that combines scene visual information with transcript text, capturing the temporal relationship between what's shown and what's said.

**Example**: "IDE showing useEffect hook" + "Now let's look at how cleanup works" → combined embedding.

---

## C

### ColBERT (Contextualized Late Interaction over BERT)
A retrieval approach where both query and document get per-token embeddings, and relevance is computed via late interaction (MaxSim).

**Why it matters**: Catches exact term matches that dense embeddings blur together.

### Cosine Similarity
A measure of similarity between two vectors, ranging from -1 to 1. Computed as the dot product divided by the product of magnitudes.

**Formula**: `cos(θ) = (A · B) / (||A|| × ||B||)`

**Interpretation**: 1.0 = identical, 0.0 = unrelated, -1.0 = opposite.

### Cross-encoder
An architecture where query and document are encoded together, allowing deep interaction. More accurate than bi-encoders but slower.

**Use case**: Reranking top results after initial retrieval.

---

## D

### Dense Embedding
A fixed-length vector (e.g., 1024 dimensions) where every dimension has a value. Captures semantic meaning.

**Contrast with**: Sparse embeddings where most values are zero.

### DCG (Discounted Cumulative Gain)
A ranking metric that measures quality by summing relevance scores with a logarithmic discount for position.

**Formula**: `DCG = Σ (2^rel - 1) / log2(position + 1)`

**Example**: A highly relevant result at position 1 contributes more than at position 10.

---

## E

### Embedding
A numerical representation of text, code, or images as a list of numbers (vector). Embeddings capture semantic meaning so similar items have similar numbers.

**Analogy**: Like GPS coordinates for concepts in a high-dimensional space.

### Extraction
The process of pulling structured information from raw content (video, PDFs, code files).

**Our extractors**: Video (Whisper + scene detection), Slides (PDF + OCR), Code (parsing + analysis).

---

## F

### Fusion
Combining results from multiple search strategies into a single ranking. We use Reciprocal Rank Fusion (RRF).

**Why**: Different search methods find different relevant content; fusion gets the best of all.

---

## G

### Ground Truth
The known correct answers for evaluation. For retrieval, this means human-annotated relevance judgments.

**Scale**: 0 (not relevant) to 3 (highly relevant).

---

## H

### HNSW (Hierarchical Navigable Small World)
An algorithm for approximate nearest neighbor search in high-dimensional spaces. Used in vector databases for fast similarity search.

**Trade-off**: Slight accuracy loss for massive speed improvement.

### HyDE (Hypothetical Document Embeddings)
A technique that generates a hypothetical answer to a query, then embeds that answer for retrieval. Bridges the semantic gap between short queries and long documents.

**Impact**: +9.7% NDCG improvement in our system.

---

## I

### IDCG (Ideal DCG)
The best possible DCG score if results were perfectly ranked. Used to normalize DCG into NDCG.

### Intent (Query Intent)
The type of information a user is seeking. We classify intents to adjust retrieval strategy.

**Types**: concept_explanation, code_example, comparison, troubleshooting, tool_usage.

---

## K

### Keyframe
A representative frame from a video scene, typically from the middle of the scene. Used for visual analysis and thumbnails.

---

## L

### Late Interaction
A retrieval approach where query and document embeddings are computed separately, but similarity is computed at the token level (not just one score per document).

**Example**: ColBERT computes max similarity for each query token across all document tokens.

### Listwise Reranking
Reranking where the model considers all candidates together, rather than scoring each independently.

**Advantage**: Can consider relationships between candidates (e.g., "this is more complete than that").

### LLM (Large Language Model)
AI models like Claude or GPT that can understand and generate text. We use LLMs for HyDE generation and reranking.

---

## M

### MAP (Mean Average Precision)
A ranking metric that averages precision at each relevant result position.

**Interpretation**: Higher = relevant results appear earlier and more consistently.

### Modal / Modality
A type of information. Our system handles three modalities: audio (transcript), visual (slides/scenes), and code.

### MRR (Mean Reciprocal Rank)
The average of 1/rank where rank is the position of the first relevant result.

**Example**: If first relevant result is at position 2, reciprocal rank = 0.5.

### Multi-Vector
An approach where each document gets multiple embeddings (e.g., one for transcript, one for code, one for visuals) instead of a single embedding.

**Why**: Different aspects of content are better captured by different embeddings.

---

## N

### NDCG (Normalized Discounted Cumulative Gain)
Our primary metric. DCG normalized by IDCG to give a score from 0 to 1.

**Interpretation**: 0.84 means 84% of the theoretical maximum ranking quality.

**Formula**: `NDCG = DCG / IDCG`

---

## O

### OCR (Optical Character Recognition)
Technology that extracts text from images. Used to get text from slides.

**Limitation**: Can struggle with code, diagrams, and complex layouts.

---

## P

### Precision@k
The fraction of top-k results that are relevant.

**Example**: If 6 of top 10 results are relevant, Precision@10 = 0.6.

### PySceneDetect
A Python library for detecting scene transitions in videos. We use it to segment lectures into coherent scenes.

---

## Q

### Query Enhancement
Techniques that improve the query before searching. HyDE is a query enhancement technique.

### Qdrant
An open-source vector database optimized for neural search. Supports multi-vector storage and HNSW indexing.

---

## R

### Recall@k
The fraction of all relevant documents that appear in the top-k results.

**Example**: If there are 10 relevant documents and 9 appear in top-50, Recall@50 = 0.9.

### Reciprocal Rank Fusion (RRF)
A method for combining multiple ranking lists that doesn't require score normalization.

**Formula**: `RRF(d) = Σ 1/(k + rank(d))` where k is typically 60.

### Relevance Judgment
A human annotation indicating how relevant a document is to a query. We use a 0-3 scale.

### Reranking
A second-stage process that reorders initial retrieval results using a more sophisticated model.

**Typical pipeline**: Bi-encoder retrieval (fast, top-100) → Cross-encoder/LLM reranking (slow, top-10).

---

## S

### Scene Detection
Identifying transition points in video where the visual content changes significantly.

**Threshold tuning**: Lower threshold = more scenes. 28 works well for slide-based lectures.

### Semantic Search
Search based on meaning rather than exact keyword matches. Enabled by embeddings.

**Example**: "memory leak" matches content about "forgetting to free allocated memory".

### SFR-Embedding-Code
A code-specialized embedding model from Salesforce, based on StarCoder.

**Use**: Better captures code semantics than general text embeddings.

### Sparse Embedding
An embedding where most values are zero, similar to traditional TF-IDF. Only meaningful terms have non-zero weights.

**Advantage**: Exact term matching, interpretable.
**Disadvantage**: Misses synonyms and semantic similarity.

### SOTA (State of the Art)
The best known performance on a task at a given time.

---

## T

### Temporal Binding
Connecting information based on when it appears in a video. Scene-transcript binding uses temporal overlap.

### Top-k
The top k results from a retrieval operation. Common values: k=10, k=20, k=50.

### Transcript
Text extracted from speech in a video. Includes timestamps for each segment and word.

---

## V

### Vector
A list of numbers. In our context, vectors are embeddings that represent semantic meaning.

**Dimensions**: Our embeddings are 1024-dimensional.

### Vector Database
A database optimized for storing and searching vectors by similarity. Examples: Qdrant, Pinecone, Vespa.

### Vector Store
Our in-memory implementation of vector storage. Production systems use vector databases.

---

## W

### Whisper
OpenAI's automatic speech recognition model. We use it for transcription with word-level timestamps.

**Models**: tiny (fastest), base, small, medium, large (most accurate).

### Word-level Timestamps
Timestamps for each individual word in a transcript, not just sentences.

**Why important**: Enables precise scene-transcript binding.

---

## Quick Reference Table

| Term | One-line Definition | Importance |
|------|---------------------|------------|
| BGE-M3 | Multi-functional text embedding model | Core component |
| HyDE | Generate hypothetical answer, then search | +9.7% NDCG |
| RRF | Combine multiple rankings without normalization | Enables multi-vector |
| NDCG@10 | Primary quality metric (0-1) | Evaluation |
| Multi-vector | Multiple embeddings per document | +5% NDCG |
| Whisper | Speech-to-text model | Extraction |
| ColBERT | Per-token embeddings for exact matching | Precision |
| Cross-encoder | Joint query-document encoding | Reranking |

---

*Need a term added? Open an issue on [GitHub](https://github.com/mixpeek/benchmarks).*
