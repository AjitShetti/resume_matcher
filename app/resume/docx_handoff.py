"""
app/resume/docx_handoff.py — Data reshaper for the ats-resume-builder skill.

This module is intentionally thin. Its only job is to take approved project
bullets (post manual review) and reshape them into the structure that the
ats-resume-builder skill consumes for its Projects section.

It does NOT generate any docx itself — that responsibility belongs to the
ats-resume-builder skill, which is wired in once its expected input format
is confirmed (see TODO below).
"""

from __future__ import annotations


def prepare_handoff(approved_projects: list[dict]) -> dict:
    """
    Reshape approved project bullets into the input structure expected by
    the ats-resume-builder skill's Projects section.

    Args:
        approved_projects: List of dicts, each containing:
            - "repo_name"     (str)  — GitHub repo name used as project title
            - "final_bullets" (list[str]) — manually approved rewritten bullets

    Returns:
        A dict with a "projects" key containing a list of project dicts
        ready to pass to the ats-resume-builder skill.

    Example output shape:
        {
            "projects": [
                {
                    "title":   "fastapi-ml-service",
                    "bullets": [
                        "Deployed ML models as REST APIs using FastAPI and Docker.",
                        "Achieved sub-100ms inference latency with async endpoints.",
                    ]
                },
                ...
            ]
        }

    # -----------------------------------------------------------------------
    # TODO: Wire into ats-resume-builder
    # -----------------------------------------------------------------------
    # Once you confirm the ats-resume-builder skill's expected input format,
    # replace the stub call below with the real entry point. For example:
    #
    #   from skills.ats_resume_builder import build_resume
    #   docx_bytes = build_resume(handoff_payload)
    #
    # The `handoff_payload` returned by this function is designed to drop
    # directly into the skill's `projects` input key — adjust field names
    # (e.g. "title" vs "name", "bullets" vs "points") to match the skill's
    # actual schema once confirmed.
    # -----------------------------------------------------------------------
    """
    projects = []

    for entry in approved_projects:
        repo_name = entry.get("repo_name", "")
        final_bullets = entry.get("final_bullets", [])

        if not repo_name:
            continue  # skip malformed entries silently

        projects.append({
            "title": repo_name,
            "bullets": list(final_bullets),
        })

    handoff_payload = {"projects": projects}
    return handoff_payload
