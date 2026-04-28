"""
evaluate_k.py

Compares retrieval quality across different values of k
and across strategies (top-k cosine vs MMR).

Currently uses:
  - Intra-list diversity (ILD): measures how different the retrieved captions are
    from each other. Higher = more diverse.
  - Average relevance score: mean cosine similarity of retrieved captions to query.

These are retrieval-level metrics that don't require audio.
Once the full pipeline is ready, this will be extended with
BLEU / CIDEr scores on generated captions.
"""

import numpy as np
import pandas as pd
from .datastore import Datastore
from .retriever import TopKRetriever, MMRRetriever, _cosine_similarity_matrix


# ------------------------------------------------------------------
# Retrieval-level metrics
# ------------------------------------------------------------------

def average_relevance(
    query: np.ndarray,
    retrieved_indices: list[int],
    matrix: np.ndarray,
) -> float:
    """Mean cosine similarity of retrieved items to the query."""
    sims = _cosine_similarity_matrix(query, matrix[retrieved_indices])
    return float(sims.mean())


def intra_list_diversity(
    retrieved_indices: list[int],
    matrix: np.ndarray,
) -> float:
    """
    Intra-List Diversity (ILD):
    Average pairwise dissimilarity among retrieved items.
    ILD = 1 - average pairwise cosine similarity.
    Higher = more diverse retrieved set.
    """
    if len(retrieved_indices) < 2:
        return 0.0

    sub = matrix[retrieved_indices]  # (k, d)
    norms = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-10)
    sim_matrix = norms @ norms.T  # (k, k)

    # Take upper triangle (excluding diagonal)
    k = len(retrieved_indices)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    avg_sim = np.mean([sim_matrix[i, j] for i, j in pairs])

    return float(1.0 - avg_sim)


# ------------------------------------------------------------------
# Main comparison
# ------------------------------------------------------------------

def evaluate_k(
    datastore: Datastore,
    k_values: list[int] = [1, 2, 4, 8],
    lambda_values: list[float] = [0.5, 0.7, 1.0],
    n_queries: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluate retrieval quality across k values and strategies.

    Args:
        datastore:     Datastore with embeddings built
        k_values:      list of k values to test
        lambda_values: lambda values for MMR (1.0 = same as top-k)
        n_queries:     number of random query embeddings to average over
        seed:          random seed for reproducibility

    Returns:
        DataFrame with columns: strategy, k, lambda, avg_relevance, avg_diversity
    """
    assert datastore.embeddings_matrix is not None, "Build embeddings first."

    np.random.seed(seed)
    d = datastore.embeddings_matrix.shape[1]

    # Sample random query embeddings (replace with real CLAP audio embeddings later)
    queries = np.random.randn(n_queries, d).astype(np.float32)

    matrix = datastore.embeddings_matrix
    records = []

    for k in k_values:
        # --- Top-K ---
        retriever = TopKRetriever(datastore, k=k)
        rel_scores, div_scores = [], []

        for query in queries:
            retrieved_caps = retriever.retrieve(query)
            indices = [
                i for i, e in enumerate(datastore.entries)
                if e.caption in retrieved_caps
            ][:k]

            rel_scores.append(average_relevance(query, indices, matrix))
            div_scores.append(intra_list_diversity(indices, matrix))

        records.append({
            "strategy": "topk",
            "k": k,
            "lambda": 1.0,
            "avg_relevance": np.mean(rel_scores),
            "avg_diversity": np.mean(div_scores),
        })

        # --- MMR for each lambda ---
        for lam in lambda_values:
            if lam == 1.0:
                continue  # same as top-k, already recorded

            retriever = MMRRetriever(datastore, k=k, lambda_=lam)
            rel_scores, div_scores = [], []

            for query in queries:
                retrieved_caps = retriever.retrieve(query)
                indices = [
                    i for i, e in enumerate(datastore.entries)
                    if e.caption in retrieved_caps
                ][:k]

                rel_scores.append(average_relevance(query, indices, matrix))
                div_scores.append(intra_list_diversity(indices, matrix))

            records.append({
                "strategy": "mmr",
                "k": k,
                "lambda": lam,
                "avg_relevance": np.mean(rel_scores),
                "avg_diversity": np.mean(div_scores),
            })

    df = pd.DataFrame(records)
    return df


# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    import os

    BASE = os.path.join(os.path.dirname(__file__), "..", "data", "audiocaps")

    ds = Datastore()
    ds.load_from_csv(os.path.join(BASE, "train.clean.csv"))

    # Simulate embeddings (replace with CLAP later)
    np.random.seed(0)
    ds.build_embeddings(lambda texts: np.random.randn(len(texts), 512))

    print("Running evaluation across k values and strategies...\n")
    results = evaluate_k(
        ds,
        k_values=[1, 2, 4, 8],
        lambda_values=[0.5, 0.7, 1.0],
        n_queries=50,
    )

    print(results.to_string(index=False))

    # Best k for top-k (by relevance)
    topk_results = results[results["strategy"] == "topk"]
    best_k = topk_results.loc[topk_results["avg_relevance"].idxmax(), "k"]
    print(f"\nBest k for top-k retriever (highest relevance): k={int(best_k)}")

    # Best MMR config (by diversity, with relevance above threshold)
    threshold = results["avg_relevance"].mean()
    mmr_results = results[
        (results["strategy"] == "mmr") & (results["avg_relevance"] >= threshold)
    ]
    if not mmr_results.empty:
        best_row = mmr_results.loc[mmr_results["avg_diversity"].idxmax()]
        print(
            f"Best MMR config (highest diversity above avg relevance): "
            f"k={int(best_row['k'])}, lambda={best_row['lambda']}"
        )
