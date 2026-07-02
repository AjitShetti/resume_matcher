"""
app/github/fetch.py — GitHub REST API helpers for resume-matcher.

All requests are unauthenticated (public repos only).
Unauthenticated GitHub API limit: 60 requests/hour.
Rate-limit remaining is logged after every call so you can track usage.
"""

from __future__ import annotations

import json
import logging

import requests

from app import config

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _log_rate_limit(response: requests.Response) -> None:
    """Log the X-RateLimit-Remaining header after every API call."""
    remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
    limit = response.headers.get("X-RateLimit-Limit", "unknown")
    logger.debug("GitHub rate limit: %s / %s requests remaining", remaining, limit)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_repo_description(username: str, repo_name: str) -> str | None:
    """
    Return the GitHub repo description string, or None if:
    - the repo doesn't exist (404)
    - the description field is absent or null
    - any other request error occurs
    """
    url = f"{_GITHUB_API}/repos/{username}/{repo_name}"
    try:
        response = requests.get(url, timeout=10)
        _log_rate_limit(response)

        if response.status_code == 404:
            logger.warning("Repo not found: %s/%s", username, repo_name)
            return None

        response.raise_for_status()
        return response.json().get("description") or None

    except requests.RequestException as exc:
        logger.error("Failed to fetch description for %s/%s: %s", username, repo_name, exc)
        return None


def get_repo_readme(username: str, repo_name: str) -> str | None:
    """
    Return the raw README text for a repo, or None if:
    - the repo has no README (404)
    - the repo doesn't exist (404)
    - any other request error occurs

    Uses Accept: application/vnd.github.raw+json so GitHub returns the
    raw file content directly — no manual base64 decoding needed.
    """
    url = f"{_GITHUB_API}/repos/{username}/{repo_name}/readme"
    headers = {"Accept": "application/vnd.github.raw+json"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        _log_rate_limit(response)

        if response.status_code == 404:
            logger.warning("README not found for: %s/%s", username, repo_name)
            return None

        response.raise_for_status()
        return response.text or None

    except requests.RequestException as exc:
        logger.error("Failed to fetch README for %s/%s: %s", username, repo_name, exc)
        return None


def load_shortlist() -> list[dict]:
    """
    Read and parse shortlist.json from the path defined in config.SHORTLIST_PATH.
    Returns a list of dicts, each expected to have a 'repo_name' key.
    """
    with open(config.SHORTLIST_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Standalone sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    username = config.GITHUB_USERNAME
    shortlist = load_shortlist()

    print(f"\nFetching data for {len(shortlist)} repos under '{username}'...\n")
    print("-" * 60)

    for entry in shortlist:
        repo_name = entry["repo_name"]
        description = get_repo_description(username, repo_name)
        readme = get_repo_readme(username, repo_name)
        readme_len = len(readme) if readme else 0

        print(f"Repo       : {repo_name}")
        print(f"Description: {description!r}")
        print(f"README len : {readme_len} chars")
        print("-" * 60)
