"""
datastore.py

Loads captions from CSV and manages the retrieval datastore.

Text embeddings must match CLAP joint projection dim (typically 512): see
``clap_embeddings.encode_texts`` or ``precompute_text_embeddings.py`` to build
``.npy`` matrices aligned with ``ClapModel.get_audio_features`` queries.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatastoreEntry:
    access_id: str
    caption: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class Datastore:
    def __init__(self):
        self.entries: list[DatastoreEntry] = []
        self._embeddings_matrix: Optional[np.ndarray] = None  # shape: (N, d)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_csv(self, csv_path: str) -> None:
        """Load captions from a clean CSV file (train/val/test)."""
        df = pd.read_csv(csv_path)

        required_cols = {"access_id", "caption"}
        assert required_cols.issubset(df.columns), (
            f"CSV must contain columns: {required_cols}. Found: {set(df.columns)}"
        )

        for _, row in df.iterrows():
            entry = DatastoreEntry(
                access_id=str(row["access_id"]),
                caption=str(row["caption"]),
            )
            self.entries.append(entry)

        print(f"[Datastore] Loaded {len(self.entries)} entries from {csv_path}")

    def load_from_multiple_csvs(self, csv_paths: list[str]) -> None:
        """Load and merge captions from multiple CSV files (e.g., train + val)."""
        for path in csv_paths:
            self.load_from_csv(path)
        print(f"[Datastore] Total entries after merging: {len(self.entries)}")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def build_embeddings(self, embed_fn) -> None:
        """
        Populate embeddings using a provided embedding function.

        Args:
            embed_fn: callable that takes a list of strings and returns
                      np.ndarray of shape (N, d).

        Example:

            from clap_embeddings import build_text_embed_fn

            datastore.build_embeddings(build_text_embed_fn())
        """
        captions = [e.caption for e in self.entries]
        embeddings = embed_fn(captions)  # (N, d)

        assert len(embeddings) == len(self.entries), "Embedding count mismatch."

        for entry, emb in zip(self.entries, embeddings):
            entry.embedding = emb

        self._embeddings_matrix = np.array(embeddings, dtype=np.float32)
        print(f"[Datastore] Embeddings built. Shape: {self._embeddings_matrix.shape}")

    def load_embeddings_from_file(self, npy_path: str) -> None:
        """Load precomputed embeddings from a .npy file."""
        matrix = np.load(npy_path)
        assert matrix.shape[0] == len(self.entries), (
            f"Embedding rows ({matrix.shape[0]}) != entry count ({len(self.entries)})"
        )
        self._embeddings_matrix = matrix.astype(np.float32)
        for entry, emb in zip(self.entries, matrix):
            entry.embedding = emb
        print(f"[Datastore] Loaded embeddings from {npy_path}. Shape: {matrix.shape}")

    def save_embeddings_to_file(self, npy_path: str) -> None:
        """Save current embeddings matrix to a .npy file."""
        assert self._embeddings_matrix is not None, "No embeddings to save."
        np.save(npy_path, self._embeddings_matrix)
        print(f"[Datastore] Embeddings saved to {npy_path}")

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def embeddings_matrix(self) -> Optional[np.ndarray]:
        """Returns (N, d) matrix of all embeddings, or None if not built yet."""
        return self._embeddings_matrix

    @property
    def captions(self) -> list[str]:
        return [e.caption for e in self.entries]

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        emb_status = (
            f"embeddings shape={self._embeddings_matrix.shape}"
            if self._embeddings_matrix is not None
            else "embeddings=None (not built yet)"
        )
        return f"Datastore(entries={len(self.entries)}, {emb_status})"


# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Adjust path as needed
    BASE = os.path.join(os.path.dirname(__file__), "..", "data", "audiocaps")

    ds = Datastore()
    ds.load_from_csv(os.path.join(BASE, "train.clean.csv"))
    print(ds)
    print("Sample entry:", ds.entries[0])

    # Simulate embeddings with random vectors (replace with CLAP later)
    print("\n[Test] Simulating embeddings with random vectors...")
    ds.build_embeddings(lambda texts: np.random.randn(len(texts), 512))
    print(ds)
