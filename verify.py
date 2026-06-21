"""
Test script for the Ayush clone RAG retrieval system.

Run:
    python verify.py "your test query here"
"""

import sys
from rag import KnowledgeBase


def main():
    # If a query is provided as an argument, use it; otherwise use a default test query.
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What are your skills and tech stack?"
    )

    print("Initializing KnowledgeBase...")
    try:
        kb = KnowledgeBase()
    except Exception as e:
        print(f"Error initializing knowledge base: {e}")
        return

    print(f"Loaded {len(kb.chunks)} chunks from knowledge base.")
    print(f"Searching for query: '{query}'\n")

    # Run search with minimum score of 0.0 to show all ranked results
    results = kb.search(query, k=5, min_score=0.0)

    if not results:
        print("No matches found above the minimum threshold.")
        return

    print("Top matched chunks:")
    for idx, chunk in enumerate(results, 1):
        print(f"[{idx}] ---------------------------------------------")
        print(chunk)
        print()


if __name__ == "__main__":
    main()
