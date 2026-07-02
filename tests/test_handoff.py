"""
tests/test_handoff.py — Smoke tests for app.resume.docx_handoff (Stage 6).
Run: python -m tests.test_handoff
"""

import os
os.environ.setdefault("GITHUB_USERNAME", "AjitShetti")

from app.resume.docx_handoff import prepare_handoff  # noqa: E402

APPROVED = [
    {
        "repo_name": "fastapi-ml-service",
        "final_bullets": [
            "Deployed ML models as REST APIs using FastAPI.",
            "Achieved sub-100ms inference latency.",
        ],
    },
    {
        "repo_name": "resume-matcher",
        "final_bullets": ["Ranked GitHub projects against JDs using sentence-transformers."],
    },
]


def test_output_shape() -> None:
    result = prepare_handoff(APPROVED)
    assert "projects" in result
    assert len(result["projects"]) == 2
    p = result["projects"][0]
    assert p["title"] == "fastapi-ml-service"
    assert len(p["bullets"]) == 2
    print(f"PASS: output shape correct -> {result}")


def test_empty_repo_name_skipped() -> None:
    result = prepare_handoff([{"repo_name": "", "final_bullets": ["bullet"]}])
    assert result["projects"] == [], "Empty repo_name must be skipped"
    print("PASS: empty repo_name skipped")


def test_empty_input() -> None:
    result = prepare_handoff([])
    assert result == {"projects": []}
    print("PASS: empty input returns empty projects list")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Stage 6 — docx_handoff tests")
    print("=" * 60 + "\n")
    test_output_shape()
    test_empty_repo_name_skipped()
    test_empty_input()
    print("\nAll tests passed.")
