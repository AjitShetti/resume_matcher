"""
app/matching/embeddings.py — Sentence embedding helpers for resume-matcher.

The SentenceTransformer model is loaded once at module level so the expensive
model load only happens on first import, not on every call.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app import config

# ---------------------------------------------------------------------------
# Model loaded once at import time — do NOT reload inside functions
# ---------------------------------------------------------------------------
_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(text: str) -> np.ndarray:
    """
    Return the embedding vector for a single string.

    Args:
        text: The input string to embed.

    Returns:
        A 1-D numpy array of floats (shape: [embedding_dim]).
    """
    return _model.encode(text, convert_to_numpy=True)


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed multiple strings in a single forward pass — more efficient than
    calling embed() in a loop, especially for larger batches.

    Args:
        texts: List of strings to embed.

    Returns:
        A 2-D numpy array of shape [len(texts), embedding_dim].
    """
    return _model.encode(texts, convert_to_numpy=True)
