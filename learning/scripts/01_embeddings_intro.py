#!/usr/bin/env python3
"""
01_embeddings_intro.py
======================

This script demonstrates the fundamental concept of embeddings:
converting text into numerical vectors that capture semantic meaning.

Key Concepts:
- What embeddings are
- How similar concepts get similar numbers
- Why this enables semantic search

Run: python 01_embeddings_intro.py
Requires: pip install sentence-transformers
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    print("=" * 70)
    print("EMBEDDINGS INTRODUCTION")
    print("=" * 70)
    print()

    # Load a pre-trained model
    # We use a smaller model for demo; production uses BGE-M3
    print("Loading embedding model (this may take a moment)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print()

    # =========================================================================
    # Part 1: What are embeddings?
    # =========================================================================
    print("-" * 70)
    print("PART 1: What are embeddings?")
    print("-" * 70)
    print()

    text = "Memory management in C"
    embedding = model.encode(text)

    print(f"Text: '{text}'")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 10 values: {embedding[:10].round(3)}")
    print()
    print("An embedding is just a list of numbers (a vector).")
    print("These numbers capture the 'meaning' of the text.")
    print()

    # =========================================================================
    # Part 2: Similar meanings → Similar numbers
    # =========================================================================
    print("-" * 70)
    print("PART 2: Similar meanings get similar numbers")
    print("-" * 70)
    print()

    # Define some example texts
    texts = [
        "Memory management in C",          # Original
        "Allocating and freeing memory",   # Semantically similar
        "malloc and free functions",       # Technically similar
        "Making a cup of coffee",          # Unrelated
        "The weather is nice today",       # Unrelated
    ]

    # Get embeddings for all texts
    embeddings = model.encode(texts)

    # Calculate similarities to the first text
    print(f"Reference text: '{texts[0]}'")
    print()
    print("Cosine similarities to other texts:")
    print("-" * 50)

    reference = embeddings[0]
    for i, (text, emb) in enumerate(zip(texts[1:], embeddings[1:]), 1):
        # Cosine similarity
        similarity = np.dot(reference, emb) / (np.linalg.norm(reference) * np.linalg.norm(emb))
        bar = "█" * int(similarity * 30)
        print(f"{similarity:.3f} {bar}")
        print(f"       '{text}'")
        print()

    print("Notice: Related texts (memory, malloc) have HIGH similarity (~0.5-0.7)")
    print("        Unrelated texts (coffee, weather) have LOW similarity (~0.0-0.2)")
    print()

    # =========================================================================
    # Part 3: Why this matters for search
    # =========================================================================
    print("-" * 70)
    print("PART 3: Why this matters for search")
    print("-" * 70)
    print()

    # Simulate a document collection
    documents = [
        "Pointers store memory addresses in C",
        "Use malloc() to allocate memory dynamically",
        "Always free() memory when done to prevent leaks",
        "Segmentation faults occur when accessing invalid memory",
        "Python lists are dynamic arrays",
        "JavaScript uses garbage collection for memory",
    ]

    # Create a "database" of document embeddings
    doc_embeddings = model.encode(documents)

    # User query
    query = "How do I avoid memory leaks?"
    query_embedding = model.encode(query)

    print(f"User query: '{query}'")
    print()
    print("Search results (ranked by similarity):")
    print("-" * 50)

    # Calculate similarities
    similarities = [
        np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        for doc_emb in doc_embeddings
    ]

    # Sort by similarity
    ranked = sorted(zip(similarities, documents), reverse=True)

    for rank, (sim, doc) in enumerate(ranked, 1):
        print(f"[{rank}] Score: {sim:.3f}")
        print(f"    '{doc}'")
        print()

    print("The search found relevant content about free() and memory management,")
    print("even though the query didn't contain the exact words!")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key takeaways:

1. EMBEDDINGS convert text into numbers (vectors)
   - Each text becomes a list of ~384-1024 numbers

2. SIMILAR MEANINGS get similar numbers
   - "memory management" ≈ "malloc and free"
   - "memory management" ≠ "making coffee"

3. This enables SEMANTIC SEARCH
   - Find relevant content without exact keyword matches
   - "avoid memory leaks" finds "Always free() memory..."

This is the foundation of our multimodal retrieval system!
    """)


if __name__ == "__main__":
    main()
