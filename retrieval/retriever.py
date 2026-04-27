"""
retriever.py

Two retrieval strategies for RECAP:
  1. TopKRetriever  — standard cosine similarity top-k (same as RECAP paper)
  2. MMRRetriever   — Maximal Marginal Relevance (our contribution)
                      balances relevance + diversity for compositional audio

Both accept a query_embedding (np.ndarray) and return retrieved captions.
query_embedding comes from CLAP audio encoder — plug in later.
"""

import numpy as np

try:
    from .datastore import Datastore
except ImportError:
    from datastore import Datastore


def _cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and each row of matrix.

    Args:
        query:  (d,)
        matrix: (N, d)
    Returns:
        sims: (N,)
    """
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norms @ query_norm  # (N,)


# ------------------------------------------------------------------
# Strategy 1: Top-K Cosine (paper baseline)
# ------------------------------------------------------------------

class TopKRetriever:
    """
    Retrieve top-k captions by cosine similarity.
    Identical to the retrieval strategy in the RECAP paper.
    """

    def __init__(self, datastore: Datastore, k: int = 4):
        self.datastore = datastore
        self.k = k

    def retrieve(
        self,
        query_embedding: np.ndarray,
        exclude_access_id: str = None,
    ) -> list[str]:
        """
        Args:
            query_embedding:   (d,) audio embedding from CLAP
            exclude_access_id: access_id of the current sample to exclude
                               (avoids retrieving the ground-truth caption itself)
        Returns:
            List of top-k caption strings.
        """
        matrix = self.datastore.embeddings_matrix
        assert matrix is not None, "Datastore embeddings not built yet."

        sims = _cosine_similarity_matrix(query_embedding, matrix)

        # Exclude the current sample's own caption during training
        if exclude_access_id is not None:
            for i, entry in enumerate(self.datastore.entries):
                if entry.access_id == exclude_access_id:
                    sims[i] = -np.inf

        top_indices = np.argsort(sims)[::-1][: self.k]
        return [self.datastore.entries[i].caption for i in top_indices]


# ------------------------------------------------------------------
# Strategy 2: MMR (our contribution)
# ------------------------------------------------------------------

class MMRRetriever:
    """
    Maximal Marginal Relevance retrieval.

    At each step, selects the candidate that maximizes:
        MMR = λ · sim(query, candidate)
            - (1 - λ) · max_sim(candidate, already_selected)

    λ = 1.0  →  identical to top-k cosine
    λ = 0.5  →  equal weight on relevance and diversity
    λ = 0.0  →  pure diversity (not useful alone)

    Why this matters for audio captioning:
        Compositional audio (e.g., dog + car + rain) benefits from
        diverse retrieved captions that cover each event separately,
        rather than k near-duplicate captions about just one event.
    """

    def __init__(self, datastore: Datastore, k: int = 4, lambda_: float = 0.7):
        """
        Args:
            datastore: Datastore instance with embeddings built
            k:         number of captions to retrieve
            lambda_:   trade-off between relevance (1.0) and diversity (0.0)
        """
        self.datastore = datastore
        self.k = k
        self.lambda_ = lambda_

    def retrieve(
        self,
        query_embedding: np.ndarray,
        exclude_access_id: str = None,
    ) -> list[str]:
        """
        Args:
            query_embedding:   (d,) audio embedding from CLAP
            exclude_access_id: access_id to exclude (same as TopKRetriever)
        Returns:
            List of k caption strings, balancing relevance and diversity.
        """
        matrix = self.datastore.embeddings_matrix
        assert matrix is not None, "Datastore embeddings not built yet."

        relevance_scores = _cosine_similarity_matrix(query_embedding, matrix)

        # Exclude current sample
        excluded = set()
        if exclude_access_id is not None:
            for i, entry in enumerate(self.datastore.entries):
                if entry.access_id == exclude_access_id:
                    excluded.add(i)
                    relevance_scores[i] = -np.inf

        selected_indices = []
        candidate_indices = [
            i for i in range(len(self.datastore.entries)) if i not in excluded
        ]

        for _ in range(self.k):
            if not candidate_indices:
                break

            if not selected_indices:
                # First pick: just take the most relevant
                best = max(candidate_indices, key=lambda i: relevance_scores[i])
            else:
                selected_matrix = matrix[selected_indices]  # (num_selected, d)

                best = None
                best_score = -np.inf

                for i in candidate_indices:
                    rel = self.lambda_ * relevance_scores[i]

                    # Similarity to already-selected captions
                    cand_emb = matrix[i]  # (d,)
                    redundancy = _cosine_similarity_matrix(cand_emb, selected_matrix)
                    div = (1 - self.lambda_) * redundancy.max()

                    mmr_score = rel - div
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = i

            selected_indices.append(best)
            candidate_indices.remove(best)

        return [self.datastore.entries[i].caption for i in selected_indices]


# ------------------------------------------------------------------
# Retriever factory — easy to swap strategy
# ------------------------------------------------------------------

def get_retriever(strategy: str, datastore: Datastore, k: int, **kwargs):
    """
    Factory function to get a retriever by name.

    Args:
        strategy: "topk" or "mmr"
        datastore: Datastore instance
        k: number of captions to retrieve
        **kwargs: additional args (e.g., lambda_ for MMR)
    """
    if strategy == "topk":
        return TopKRetriever(datastore, k=k)
    elif strategy == "mmr":
        lambda_ = kwargs.get("lambda_", 0.7)
        return MMRRetriever(datastore, k=k, lambda_=lambda_)
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}. Choose 'topk' or 'mmr'.")


# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    import os

    BASE = os.path.join(os.path.dirname(__file__), "..", "data", "audiocaps")

    ds = Datastore()
    ds.load_from_csv(os.path.join(BASE, "train.clean.csv"))

    # Simulate CLAP embeddings (replace with real CLAP output later)
    np.random.seed(42)
    ds.build_embeddings(lambda texts: np.random.randn(len(texts), 512))

    # Simulate a query audio embedding
    query = np.random.randn(512)

    print("=== Top-K Retriever (k=4) ===")
    topk = TopKRetriever(ds, k=4)
    results = topk.retrieve(query)
    for i, cap in enumerate(results, 1):
        print(f"  {i}. {cap}")

    print("\n=== MMR Retriever (k=4, lambda=0.7) ===")
    mmr = MMRRetriever(ds, k=4, lambda_=0.7)
    results = mmr.retrieve(query)
    for i, cap in enumerate(results, 1):
        print(f"  {i}. {cap}")
