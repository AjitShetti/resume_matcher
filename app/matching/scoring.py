"""
app/matching/scoring.py — Semantic similarity scoring for resume-matcher.

Scores each repo against a job description using weighted cosine similarities
across the repo's description and README, plus a keyword bonus on the repo name.
"""

from __future__ import annotations

import re

import numpy as np

from app import config
from app.matching.embeddings import embed

# ---------------------------------------------------------------------------
# Common English stopwords to ignore when computing keyword bonus
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "it", "its", "as", "into",
    "not", "no", "so", "if", "we", "you", "i", "my", "your", "our",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Return the cosine similarity between two vectors as a float in [-1, 1].

    Returns 0.0 if either vector has zero norm (avoids division by zero).
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def score_repo(
    jd_text: str,
    repo: dict,
    jd_embedding: np.ndarray | None = None,
) -> dict:
    """
    Score a single repo against a job description.

    Args:
        jd_text:      The raw job description string.
        repo:         Dict with keys: "repo_name", "description", "cleaned_readme".
        jd_embedding: Optional pre-computed JD embedding. If supplied, jd_text
                      is only used for the keyword bonus — no re-embedding occurs.

    Returns:
        Dict with keys:
            repo_name       — str
            sim_description — float  (0.0 if description missing/empty)
            sim_readme      — float  (0.0 if cleaned_readme missing/empty)
            keyword_bonus   — float  (SCORING_WEIGHTS["name_keyword_bonus"] or 0.0)
            final_score     — float
    """
    weights = config.SCORING_WEIGHTS

    # Embed the JD once if not provided
    if jd_embedding is None:
        jd_embedding = embed(jd_text)

    # --- description similarity ---
    description = repo.get("description") or ""
    if description.strip():
        sim_description = cosine_similarity(jd_embedding, embed(description))
    else:
        sim_description = 0.0

    # --- README similarity ---
    cleaned_readme = repo.get("cleaned_readme") or ""
    if cleaned_readme.strip():
        sim_readme = cosine_similarity(jd_embedding, embed(cleaned_readme))
    else:
        sim_readme = 0.0

    # --- keyword bonus ---
    # Extract meaningful words from JD (lowercased, no stopwords, alpha-only)
    jd_words = {
        w for w in re.findall(r"[a-z]+", jd_text.lower())
        if w not in _STOPWORDS and len(w) > 2
    }
    repo_name_lower = repo.get("repo_name", "").lower()
    keyword_bonus = (
        weights["name_keyword_bonus"]
        if any(word in repo_name_lower for word in jd_words)
        else 0.0
    )

    # --- final weighted score ---
    final_score = (
        weights["description"] * sim_description
        + weights["readme"] * sim_readme
        + keyword_bonus
    )

    return {
        "repo_name": repo.get("repo_name", ""),
        "sim_description": round(sim_description, 4),
        "sim_readme": round(sim_readme, 4),
        "keyword_bonus": round(keyword_bonus, 4),
        "final_score": round(final_score, 4),
    }


def rank_repos(jd_text: str, repos: list[dict]) -> list[dict]:
    """
    Score all repos against the JD and return them sorted by final_score descending.

    The JD is embedded once and reused across all repos for efficiency.

    Args:
        jd_text: The raw job description string.
        repos:   List of dicts with keys "repo_name", "description", "cleaned_readme".

    Returns:
        The full list of scored dicts sorted by final_score descending.
        Callers slice to top-N themselves.
    """
    # Embed JD once — reuse for every repo
    jd_embedding = embed(jd_text)

    scores = [
        score_repo(jd_text, repo, jd_embedding=jd_embedding)
        for repo in repos
    ]

    return sorted(scores, key=lambda s: s["final_score"], reverse=True)
