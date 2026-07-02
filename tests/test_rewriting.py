"""
tests/test_rewriting.py — Smoke tests for app.rewriting.bullet_rewriter (Stage 5).

Tests format_review() without any API call, and verifies that rewrite_bullets()
falls back to originals gracefully when the API key is invalid.

Run standalone:  python -m tests.test_rewriting
"""

import os
os.environ.setdefault("GITHUB_USERNAME", "AjitShetti")
os.environ["GROQ_API_KEY"] = "invalid-key-for-fallback-test"

from app.rewriting.bullet_rewriter import format_review, rewrite_bullets  # noqa: E402

ORIGINALS = [
    "Built a REST API using Flask and SQLite.",
    "Wrote unit tests with pytest covering 80% of the codebase.",
]


def test_format_review_contains_both_sides() -> None:
    rewritten = [
        "Developed a production REST API with Flask and SQLite, optimised for reliability.",
        "Achieved 80% test coverage using pytest, ensuring codebase stability.",
    ]
    output = format_review(ORIGINALS, rewritten)
    assert "ORIGINAL" in output and "REWRITTEN" in output
    assert "Flask" in output
    assert "[1]" in output and "[2]" in output
    print("PASS: format_review contains both sides")
    print(output)


def test_format_review_handles_length_mismatch() -> None:
    output = format_review(ORIGINALS, ["Only one rewritten bullet."])
    assert "(no rewrite)" in output
    print("PASS: format_review handles mismatched lengths")


def test_rewrite_bullets_falls_back_on_api_error() -> None:
    """With an invalid key the function must return original bullets, not crash."""
    result = rewrite_bullets("Some readme text.", "Some JD text.", ORIGINALS)
    assert result == ORIGINALS, f"Expected originals on failure, got: {result}"
    print("PASS: rewrite_bullets falls back to originals on API error")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Stage 5 — bullet_rewriter tests")
    print("=" * 60 + "\n")
    test_format_review_contains_both_sides()
    print()
    test_format_review_handles_length_mismatch()
    print()
    test_rewrite_bullets_falls_back_on_api_error()
    print("\nAll tests passed.")
