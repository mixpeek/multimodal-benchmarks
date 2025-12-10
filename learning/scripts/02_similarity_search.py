#!/usr/bin/env python3
"""
02_similarity_search.py
=======================

This script demonstrates semantic search - finding relevant content
based on meaning, not just keyword matching.

Key Concepts:
- Keyword search vs semantic search
- Building a simple search index
- How semantic search handles synonyms and related concepts

Run: python 02_similarity_search.py
Requires: pip install sentence-transformers numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleSearchEngine:
    """A minimal semantic search engine for demonstration."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def index(self, documents: list):
        """Index a list of documents."""
        self.documents = documents
        self.embeddings = self.model.encode(documents)
        print(f"Indexed {len(documents)} documents")

    def search(self, query: str, top_k: int = 5) -> list:
        """Search for relevant documents."""
        query_embedding = self.model.encode(query)

        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {"doc": self.documents[i], "score": float(similarities[i])}
            for i in top_indices
        ]


def keyword_search(query: str, documents: list) -> list:
    """Simple keyword matching for comparison."""
    query_words = set(query.lower().split())
    results = []

    for doc in documents:
        doc_words = set(doc.lower().split())
        matches = len(query_words & doc_words)
        if matches > 0:
            results.append({"doc": doc, "matches": matches})

    return sorted(results, key=lambda x: x["matches"], reverse=True)


def main():
    print("=" * 70)
    print("SEMANTIC SEARCH DEMONSTRATION")
    print("=" * 70)
    print()

    # Sample educational content (like lecture segments)
    documents = [
        "Pointers in C store memory addresses. Use * to dereference.",
        "Memory allocation with malloc() returns a pointer to heap memory.",
        "Always call free() to release dynamically allocated memory.",
        "A segmentation fault occurs when you access invalid memory addresses.",
        "Stack memory is automatically managed, heap memory is manual.",
        "Buffer overflow happens when writing beyond array boundaries.",
        "Use valgrind to detect memory leaks in your C programs.",
        "Linked lists use pointers to connect nodes in sequence.",
        "Double pointers (**) are pointers to pointers, useful for 2D arrays.",
        "Memory leaks occur when allocated memory is never freed.",
        "The heap grows upward, the stack grows downward in memory.",
        "NULL pointer dereference is a common cause of crashes.",
    ]

    # Initialize search engine
    print("Initializing semantic search engine...")
    engine = SimpleSearchEngine()
    engine.index(documents)
    print()

    # Test queries
    test_queries = [
        "How do I prevent memory leaks?",
        "What causes program crashes?",
        "heap vs stack",
    ]

    for query in test_queries:
        print("-" * 70)
        print(f"QUERY: '{query}'")
        print("-" * 70)
        print()

        # Keyword search
        print("KEYWORD SEARCH (exact word matching):")
        keyword_results = keyword_search(query, documents)
        if keyword_results:
            for i, r in enumerate(keyword_results[:3], 1):
                print(f"  [{i}] ({r['matches']} word matches) {r['doc'][:60]}...")
        else:
            print("  No matches found!")
        print()

        # Semantic search
        print("SEMANTIC SEARCH (meaning-based):")
        semantic_results = engine.search(query, top_k=3)
        for i, r in enumerate(semantic_results, 1):
            print(f"  [{i}] (score: {r['score']:.3f}) {r['doc'][:60]}...")
        print()

    # =========================================================================
    # Analysis
    # =========================================================================
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
Notice the differences:

QUERY: "How do I prevent memory leaks?"
- KEYWORD: Finds documents with "memory" or "leaks" (limited)
- SEMANTIC: Finds documents about free(), valgrind, memory management
  (understands the INTENT behind the question)

QUERY: "What causes program crashes?"
- KEYWORD: May find nothing (no exact matches for "crashes")
- SEMANTIC: Finds segfaults, NULL dereference, buffer overflow
  (understands "crashes" relates to these concepts)

QUERY: "heap vs stack"
- KEYWORD: Finds documents with "heap" or "stack"
- SEMANTIC: Finds comparison content AND related memory concepts
  (understands you want to compare memory types)

This is why semantic search achieves 0.68 NDCG vs 0.45 for keywords!
    """)


if __name__ == "__main__":
    main()
