"""
app/main.py — FastAPI orchestration layer for resume-matcher.

This file contains routing and orchestration ONLY.
All business logic lives in the sub-packages:
  app.github   → fetch
  app.cleaning → readme_cleaner
  app.matching → scoring
  app.rewriting → bullet_rewriter
  app.resume   → docx_handoff
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from app.cleaning.readme_cleaner import clean_readme
from app.github.fetch import get_repo_description, get_repo_readme, load_shortlist
from app.matching.scoring import rank_repos
from app.resume.docx_handoff import prepare_handoff
from app.rewriting.bullet_rewriter import format_review, rewrite_bullets
from app import config

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume Matcher",
    description="Rank GitHub projects against a JD and rewrite resume bullets.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    jd_text: str


class RewriteRequest(BaseModel):
    repo_name: str
    jd_text: str
    existing_bullets: list[str]


class FinalizeRequest(BaseModel):
    approved_projects: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/match")
def match(req: MatchRequest) -> dict:
    """
    Rank shortlisted repos against the JD.
    Returns top 3 + full ranked list, each with all sub-scores visible.
    """
    shortlist = load_shortlist()

    repos = []
    for entry in shortlist:
        name = entry["repo_name"]
        description = get_repo_description(config.GITHUB_USERNAME, name)
        raw_readme  = get_repo_readme(config.GITHUB_USERNAME, name)
        if description is None and raw_readme is None:
            logger.warning("Skipping %s — both description and README missing.", name)
            continue
        repos.append({
            "repo_name":      name,
            "description":    description,
            "cleaned_readme": clean_readme(raw_readme or ""),
        })

    ranked = rank_repos(req.jd_text, repos)
    return {"top_3": ranked[:3], "all_ranked": ranked}


@app.post("/rewrite")
def rewrite(req: RewriteRequest) -> dict:
    """
    Rewrite resume bullets for a single repo against the JD.
    Returns the side-by-side review string (for manual approval) and raw rewritten bullets.
    """
    raw_readme = get_repo_readme(config.GITHUB_USERNAME, req.repo_name) or ""
    rewritten  = rewrite_bullets(raw_readme, req.jd_text, req.existing_bullets)
    review     = format_review(req.existing_bullets, rewritten)
    return {"review": review, "rewritten_bullets": rewritten}


@app.post("/finalize")
def finalize(req: FinalizeRequest) -> dict:
    """
    Reshape manually approved bullets into the ats-resume-builder handoff structure.
    Actual docx generation is wired in once ats-resume-builder integration is confirmed.
    """
    handoff = prepare_handoff(req.approved_projects)
    return {"handoff": handoff}
