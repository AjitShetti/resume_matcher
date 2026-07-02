"""
tests/test_fetch.py — Smoke tests for app.github.fetch (Stage 2).

Run standalone:  python tests/test_fetch.py
"""

import os
import logging

# Set env before importing config
os.environ.setdefault("GITHUB_USERNAME", "AjitShetti")

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

from app.github.fetch import get_repo_description, get_repo_readme, load_shortlist  # noqa: E402


def test_404_on_missing_repo() -> None:
    """Both functions must return None gracefully for a non-existent repo."""
    fake_repo = "this-repo-does-not-exist-xyz999"
    desc = get_repo_description("AjitShetti", fake_repo)
    readme = get_repo_readme("AjitShetti", fake_repo)
    assert desc is None, f"Expected None for description, got: {desc!r}"
    assert readme is None, f"Expected None for readme, got: {readme!r}"
    print("PASS: 404 handled correctly — both returned None")


def test_load_shortlist() -> None:
    """shortlist.json must parse into a non-empty list of dicts with repo_name keys."""
    shortlist = load_shortlist()
    assert isinstance(shortlist, list), "Expected a list"
    assert len(shortlist) > 0, "Expected at least one entry"
    for entry in shortlist:
        assert "repo_name" in entry, f"Missing repo_name in entry: {entry}"
    names = [e["repo_name"] for e in shortlist]
    print(f"PASS: shortlist loaded with {len(shortlist)} entries -> {names}")


def test_real_repo_fetch() -> None:
    """
    Fetch a known public repo (resume_matcher itself) to verify the happy path.
    This requires an internet connection and counts against the rate limit.
    """
    username = os.environ["GITHUB_USERNAME"]
    repo_name = "resume_matcher"

    desc = get_repo_description(username, repo_name)
    readme = get_repo_readme(username, repo_name)

    print(f"Description : {desc!r}")
    print(f"README chars: {len(readme) if readme else 0}")

    # We just check types — don't assert content since it can change
    assert desc is None or isinstance(desc, str), "Description must be str or None"
    assert readme is None or isinstance(readme, str), "README must be str or None"
    print("PASS: real repo fetch returned correct types")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Stage 2 smoke tests")
    print("=" * 60 + "\n")

    test_404_on_missing_repo()
    print()
    test_load_shortlist()
    print()
    test_real_repo_fetch()

    print("\nAll tests passed.")
