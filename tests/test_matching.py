"""
tests/test_matching.py — Tests for app.matching.embeddings and app.matching.scoring.

Run standalone:  python -m tests.test_matching

Note: First run downloads the sentence-transformers model (~90 MB). Subsequent
runs use the cached model and are fast.
"""

import os
os.environ.setdefault("GITHUB_USERNAME", "AjitShetti")

import numpy as np

from app.matching.embeddings import embed, embed_batch
from app.matching.scoring import cosine_similarity, score_repo, rank_repos


# ---------------------------------------------------------------------------
# embeddings.py tests
# ---------------------------------------------------------------------------

def test_embed_returns_numpy_array() -> None:
    """embed() must return a 1-D numpy array."""
    vec = embed("Build scalable REST APIs with FastAPI and Python.")
    assert isinstance(vec, np.ndarray), f"Expected ndarray, got {type(vec)}"
    assert vec.ndim == 1, f"Expected 1-D array, got shape {vec.shape}"
    assert vec.shape[0] > 0, "Embedding vector must not be empty"
    print(f"PASS: embed() -> ndarray shape {vec.shape}")


def test_embed_batch_returns_2d_array() -> None:
    """embed_batch() must return a 2-D array with one row per input string."""
    texts = [
        "Machine learning engineer with PyTorch experience.",
        "Backend developer specialising in REST APIs.",
        "Data scientist with pandas and scikit-learn skills.",
    ]
    vecs = embed_batch(texts)
    assert isinstance(vecs, np.ndarray), f"Expected ndarray, got {type(vecs)}"
    assert vecs.ndim == 2, f"Expected 2-D array, got shape {vecs.shape}"
    assert vecs.shape[0] == len(texts), (
        f"Expected {len(texts)} rows, got {vecs.shape[0]}"
    )
    print(f"PASS: embed_batch() -> ndarray shape {vecs.shape}")


def test_embed_same_text_is_deterministic() -> None:
    """Embedding the same text twice must produce identical vectors."""
    text = "Resume matching with sentence transformers."
    vec1 = embed(text)
    vec2 = embed(text)
    assert np.allclose(vec1, vec2), "Same text must produce identical embeddings"
    print("PASS: embed() is deterministic")


def test_embed_different_texts_differ() -> None:
    """Different texts must produce clearly different vectors."""
    vec_a = embed("Python backend developer with FastAPI experience.")
    vec_b = embed("Graphic designer with Photoshop and Illustrator skills.")
    sim = cosine_similarity(vec_a, vec_b)
    assert sim < 0.95, f"Unrelated texts should not be nearly identical (sim={sim:.4f})"
    print(f"PASS: different texts produce different embeddings (sim={sim:.4f})")


# ---------------------------------------------------------------------------
# scoring.py — cosine_similarity tests
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors() -> None:
    """Identical vectors must return similarity of 1.0."""
    vec = embed("FastAPI Python REST API developer")
    sim = cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-5, f"Expected 1.0, got {sim}"
    print(f"PASS: cosine_similarity(identical) = {sim:.6f}")


def test_cosine_similarity_zero_vector() -> None:
    """Zero vector must return 0.0 without crashing."""
    vec = embed("some text")
    zero = np.zeros_like(vec)
    sim = cosine_similarity(vec, zero)
    assert sim == 0.0, f"Expected 0.0 for zero vector, got {sim}"
    print("PASS: cosine_similarity(zero vector) = 0.0")


def test_cosine_similarity_related_texts_score_higher() -> None:
    """Semantically related texts must score higher than unrelated texts."""
    jd = "Python backend engineer with REST API and FastAPI experience."
    related = "I built a REST API service using FastAPI and Python."
    unrelated = "Watercolour painting techniques for beginners."

    jd_vec = embed(jd)
    sim_related = cosine_similarity(jd_vec, embed(related))
    sim_unrelated = cosine_similarity(jd_vec, embed(unrelated))

    assert sim_related > sim_unrelated, (
        f"Related ({sim_related:.4f}) should score higher than "
        f"unrelated ({sim_unrelated:.4f})"
    )
    print(
        f"PASS: related sim={sim_related:.4f} > unrelated sim={sim_unrelated:.4f}"
    )


# ---------------------------------------------------------------------------
# scoring.py — score_repo tests
# ---------------------------------------------------------------------------

JD_TEXT = (
    "We are looking for a Python backend engineer with experience in "
    "FastAPI, REST APIs, and machine learning model deployment."
)

GOOD_REPO = {
    "repo_name": "fastapi-ml-service",
    "description": "A production-ready FastAPI service for ML model serving.",
    "cleaned_readme": (
        "Deploy machine learning models as REST APIs using FastAPI. "
        "Supports async inference, health checks, and Docker deployment. "
        "Built with Python, Pydantic, and uvicorn."
    ),
}

WEAK_REPO = {
    "repo_name": "photo-gallery",
    "description": "A simple photo gallery app built with jQuery.",
    "cleaned_readme": (
        "Browse and upload photos. Supports JPEG and PNG. "
        "Uses a flat-file storage backend. No database required."
    ),
}

EMPTY_REPO = {
    "repo_name": "my-project",
    "description": None,
    "cleaned_readme": "",
}


def test_score_repo_returns_required_keys() -> None:
    """score_repo must return all five required keys."""
    result = score_repo(JD_TEXT, GOOD_REPO)
    required = {"repo_name", "sim_description", "sim_readme", "keyword_bonus", "final_score"}
    missing = required - result.keys()
    assert not missing, f"Missing keys in score_repo output: {missing}"
    print(f"PASS: score_repo returns all required keys -> {result}")


def test_score_repo_empty_fields_dont_crash() -> None:
    """None description and empty readme must produce 0.0 sub-scores, not crash."""
    result = score_repo(JD_TEXT, EMPTY_REPO)
    assert result["sim_description"] == 0.0, "None description must give 0.0"
    assert result["sim_readme"] == 0.0, "Empty readme must give 0.0"
    print(f"PASS: empty fields handled gracefully -> {result}")


def test_score_repo_keyword_bonus_triggered() -> None:
    """Repo name containing a JD keyword must get the keyword bonus."""
    result = score_repo(JD_TEXT, GOOD_REPO)
    # 'fastapi' and 'ml' appear in repo_name 'fastapi-ml-service'
    assert result["keyword_bonus"] > 0.0, (
        f"Expected keyword bonus > 0, got {result['keyword_bonus']}"
    )
    print(f"PASS: keyword bonus triggered -> {result['keyword_bonus']}")


def test_score_repo_keyword_bonus_not_triggered() -> None:
    """Repo name with no JD keywords must get 0.0 keyword bonus."""
    result = score_repo(JD_TEXT, WEAK_REPO)
    # 'photo-gallery' shares no meaningful words with the JD
    assert result["keyword_bonus"] == 0.0, (
        f"Expected no keyword bonus, got {result['keyword_bonus']}"
    )
    print(f"PASS: keyword bonus correctly absent -> {result['keyword_bonus']}")


def test_score_repo_accepts_precomputed_jd_embedding() -> None:
    """Passing a precomputed JD embedding must give the same result as not passing one."""
    jd_emb = embed(JD_TEXT)
    result_with = score_repo(JD_TEXT, GOOD_REPO, jd_embedding=jd_emb)
    result_without = score_repo(JD_TEXT, GOOD_REPO)
    assert result_with["final_score"] == result_without["final_score"], (
        "Pre-computed embedding must produce the same final_score"
    )
    print("PASS: precomputed JD embedding produces same score")


# ---------------------------------------------------------------------------
# scoring.py — rank_repos tests
# ---------------------------------------------------------------------------

def test_rank_repos_sorted_descending() -> None:
    """rank_repos must return repos sorted by final_score descending."""
    repos = [WEAK_REPO, EMPTY_REPO, GOOD_REPO]
    ranked = rank_repos(JD_TEXT, repos)

    assert len(ranked) == 3, "Must return all repos"
    scores = [r["final_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True), (
        f"Expected descending order, got: {scores}"
    )
    # The relevant repo should rank first
    assert ranked[0]["repo_name"] == "fastapi-ml-service", (
        f"Expected fastapi-ml-service first, got: {ranked[0]['repo_name']}"
    )
    print(f"PASS: rank_repos sorted correctly -> {[r['repo_name'] for r in ranked]}")
    for r in ranked:
        print(f"  {r['repo_name']}: final={r['final_score']} "
              f"(desc={r['sim_description']}, readme={r['sim_readme']}, "
              f"bonus={r['keyword_bonus']})")


def test_rank_repos_returns_full_list() -> None:
    """rank_repos must return ALL repos, not just top-N (slicing is caller's job)."""
    repos = [GOOD_REPO, WEAK_REPO, EMPTY_REPO]
    ranked = rank_repos(JD_TEXT, repos)
    assert len(ranked) == len(repos), (
        f"Expected {len(repos)} results, got {len(ranked)}"
    )
    print("PASS: rank_repos returns full list (no slicing)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Stage 4 — embeddings + scoring tests")
    print("(First run may take a minute to download the model)")
    print("=" * 60 + "\n")

    print("--- embeddings.py ---")
    test_embed_returns_numpy_array()
    test_embed_batch_returns_2d_array()
    test_embed_same_text_is_deterministic()
    test_embed_different_texts_differ()

    print("\n--- scoring.py: cosine_similarity ---")
    test_cosine_similarity_identical_vectors()
    test_cosine_similarity_zero_vector()
    test_cosine_similarity_related_texts_score_higher()

    print("\n--- scoring.py: score_repo ---")
    test_score_repo_returns_required_keys()
    test_score_repo_empty_fields_dont_crash()
    test_score_repo_keyword_bonus_triggered()
    test_score_repo_keyword_bonus_not_triggered()
    test_score_repo_accepts_precomputed_jd_embedding()

    print("\n--- scoring.py: rank_repos ---")
    test_rank_repos_sorted_descending()
    test_rank_repos_returns_full_list()

    print("\nAll tests passed.")
