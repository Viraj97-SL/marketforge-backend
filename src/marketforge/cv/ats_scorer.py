"""
MarketForge AI — ATS Scorer.

Scores a parsed CV across five dimensions using traditional ML rules
(no LLM required — fast, deterministic, auditable).

Dimensions and weights:
  keyword_match  35%  — CV skills vs top market demand for target role
  structure      20%  — section presence, action verbs, quantified bullets
  readability    15%  — Flesch-Kincaid grade level (target 10–14)
  completeness   20%  — required fields, date ranges, contact info
  format_safety  10%  — ATS-hostile elements (tables, images, page count)

Grade scale:
  A+  90–100   A  80–89   B  70–79   C  60–69   D  <60
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from marketforge.cv.parser import ParsedCV

logger = structlog.get_logger(__name__)

# ── Scoring weights (must sum to 1.0) ─────────────────────────────────────────
_WEIGHTS: dict[str, float] = {
    "keyword_match": 0.35,
    "structure":     0.20,
    "readability":   0.15,
    "completeness":  0.20,
    "format_safety": 0.10,
}

# ── Grade thresholds (descending) ────────────────────────────────────────────
_GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (90.0, "A+"),
    (80.0, "A"),
    (70.0, "B"),
    (60.0, "C"),
    (0.0,  "D"),
]

# ── Strong action verbs ───────────────────────────────────────────────────────
_ACTION_VERBS: frozenset[str] = frozenset({
    "led", "built", "designed", "developed", "implemented", "deployed",
    "optimised", "optimized", "reduced", "improved", "increased", "managed",
    "delivered", "architected", "trained", "fine-tuned", "scaled", "automated",
    "launched", "researched", "published", "mentored", "collaborated", "created",
    "engineered", "integrated", "migrated", "refactored", "analyzed", "analysed",
    "established", "accelerated", "streamlined", "drove", "spearheaded",
})

_REQUIRED_SECTIONS: frozenset[str] = frozenset({"experience", "education", "skills"})
_BONUS_SECTIONS:    frozenset[str] = frozenset({"summary", "certifications", "projects"})

# Pattern: numbers/metrics in text
_METRIC_RE = re.compile(r"\d+\s*[%xX]|\d{4,}|\b\d+\s*(?:million|billion|k)\b", re.I)
_DATE_RE   = re.compile(r"\b(20[0-2]\d|19\d\d)\b")


@dataclass
class ATSScore:
    total:             float
    grade:             str
    breakdown:         dict[str, float]   # sub-score per dimension (0–100)
    issues:            list[str]          # human-readable improvement hints
    skills_found:      list[str]          # skills extracted from CV text
    keyword_match_pct: float              # % of target role's top skills present


def score_cv(
    cv:          ParsedCV,
    target_role: str,
    extra_skills: list[str] | None = None,
) -> ATSScore:
    """
    Score a ParsedCV.

    Args:
        cv:           Output of parser.parse_cv()
        target_role:  User's target role (e.g. "ML Engineer")
        extra_skills: Optional skills the user declared manually (supplements NLP)
    """
    issues: list[str] = []

    # Extract CV skills via existing NLP pipeline
    skills_found = _extract_skills(cv.raw_text, extra_skills)

    # Sub-scores
    kw_score, kw_pct = _score_keywords(skills_found, target_role, issues)
    struct_score     = _score_structure(cv, issues)
    read_score       = _score_readability(cv.raw_text, issues)
    complete_score   = _score_completeness(cv, issues)
    format_score     = _score_format(cv, issues)

    breakdown = {
        "keyword_match": round(kw_score, 1),
        "structure":     round(struct_score, 1),
        "readability":   round(read_score, 1),
        "completeness":  round(complete_score, 1),
        "format_safety": round(format_score, 1),
    }

    total = sum(_WEIGHTS[k] * v for k, v in breakdown.items())
    grade = next(g for threshold, g in _GRADE_THRESHOLDS if total >= threshold)

    return ATSScore(
        total             = round(total, 1),
        grade             = grade,
        breakdown         = breakdown,
        issues            = issues,
        skills_found      = skills_found,
        keyword_match_pct = round(kw_pct, 1),
    )


# ── Sub-score implementations ──────────────────────────────────────────────────

def _extract_skills(text: str, supplement: list[str] | None) -> list[str]:
    """Use existing 3-gate NLP pipeline; merge with any manually supplied skills."""
    extracted: list[str] = []
    try:
        from marketforge.nlp.taxonomy import extract_skills_flat
        extracted = [item[0] for item in extract_skills_flat(text)]
    except Exception as exc:
        logger.warning("ats.skill_extract.error", error=str(exc))

    if supplement:
        merged = set(extracted) | set(supplement)
        return sorted(merged)
    return extracted


def _score_keywords(
    skills_found: list[str],
    target_role:  str,
    issues:       list[str],
) -> tuple[float, float]:
    """
    Compare CV skills against the market's top-demanded skills for target_role.
    Falls back to role-agnostic top skills if no role-specific snapshot exists.
    Returns (score 0-100, raw_match_pct 0-100).
    """
    try:
        import json
        from marketforge.memory.postgres import get_sync_engine
        from sqlalchemy import text

        engine    = get_sync_engine()
        is_sqlite = engine.dialect.name == "sqlite"
        snap_t    = "weekly_snapshots" if is_sqlite else "market.weekly_snapshots"

        with engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT top_skills FROM {snap_t} ORDER BY week_start DESC LIMIT 1")
            ).fetchone()

        if not row or not row[0]:
            return 50.0, 50.0

        top_skills = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        market_top = list(top_skills.keys())[:30]

        found_lower  = {s.lower() for s in skills_found}
        matches      = sum(1 for s in market_top if s.lower() in found_lower)
        match_pct    = (matches / len(market_top)) * 100 if market_top else 0.0

        missing_top = [s for s in market_top[:10] if s.lower() not in found_lower]
        if missing_top:
            issues.append(
                f"Add these high-demand skills to your CV if you have them: "
                f"{', '.join(missing_top[:5])}"
            )

        # Slight scaling: 80% match → ~96 score
        score = min(match_pct * 1.2, 100.0)
        return score, match_pct

    except Exception as exc:
        logger.warning("ats.keywords.error", error=str(exc))
        return 50.0, 50.0


def _score_structure(cv: ParsedCV, issues: list[str]) -> float:
    """
    Section presence (60 pts), action verbs (10 pts), quantified bullets (10 pts),
    bonus sections (20 pts total, 5 each).
    """
    score = 0.0

    # Required sections (20 pts each)
    for section in _REQUIRED_SECTIONS:
        if cv.sections.get(section, "").strip():
            score += 20.0
        else:
            issues.append(f"Missing or empty '{section}' section")

    # Bonus sections (5 pts each, max 20)
    bonus = sum(5.0 for s in _BONUS_SECTIONS if cv.sections.get(s, "").strip())
    score += min(bonus, 20.0)

    # Action verbs in experience
    exp_text = cv.sections.get("experience", "")
    if exp_text:
        words      = set(re.findall(r"\b\w+\b", exp_text.lower()))
        verb_count = len(words & _ACTION_VERBS)
        score     += min(verb_count * 2.0, 10.0)

        # Quantified achievements
        metrics = _METRIC_RE.findall(exp_text)
        if len(metrics) >= 3:
            score += 10.0
        elif len(metrics) == 0:
            issues.append(
                "Add numbers/metrics to your experience bullets "
                "(e.g. 'reduced inference latency by 40%', 'trained on 2M samples')"
            )
        elif len(metrics) < 3:
            issues.append("Add more quantified achievements — aim for 3+ numbers in experience")

    return min(score, 100.0)


def _score_readability(text: str, issues: list[str]) -> float:
    """
    Flesch-Kincaid grade level. Professional CVs target grade 10–14.
    Graceful fallback if textstat is unavailable.
    """
    if not text.strip():
        return 0.0
    try:
        import textstat
        fk = textstat.flesch_kincaid_grade(text)
        if 10.0 <= fk <= 14.0:
            return 100.0
        if fk < 10.0:
            return max(60.0, 100.0 - (10.0 - fk) * 5.0)
        # fk > 14
        if fk > 16.0:
            issues.append(
                "Writing complexity is too high — use shorter, clearer sentences in bullet points"
            )
        return max(40.0, 100.0 - (fk - 14.0) * 6.0)
    except Exception:
        return 65.0   # neutral fallback


def _score_completeness(cv: ParsedCV, issues: list[str]) -> float:
    """
    Required contact info (40 pts), date ranges in experience (20 pts),
    skills content (20 pts), education content (20 pts).
    """
    score = 0.0

    # Contact info
    if cv.has_email:
        score += 20.0
    else:
        issues.append("No email address found — add your professional email")
    if cv.has_phone:
        score += 10.0
    if cv.has_linkedin:
        score += 10.0
    elif not cv.has_linkedin:
        issues.append("Add your LinkedIn URL to improve recruiter conversion")

    # Date ranges in experience
    exp_text = cv.sections.get("experience", "")
    if len(_DATE_RE.findall(exp_text)) >= 2:
        score += 20.0
    else:
        issues.append("Add start/end dates for each role in your experience section")

    # Skills section density
    skills_text = cv.sections.get("skills", "")
    if len(skills_text.split()) >= 6:
        score += 20.0
    else:
        issues.append("Skills section is sparse — list your key technical skills explicitly")

    # Education
    if cv.sections.get("education", "").strip():
        score += 20.0
    else:
        issues.append("Add an education section (degree, institution, year)")

    return min(score, 100.0)


def _score_format(cv: ParsedCV, issues: list[str]) -> float:
    """Penalise ATS-hostile formatting elements."""
    score = 100.0

    if cv.has_tables:
        score -= 40.0
        issues.append(
            "Tables detected — most ATS systems cannot parse table content; "
            "use plain bullet lists instead"
        )
    if cv.has_images:
        score -= 20.0
        issues.append(
            "Images/graphics found — remove them to prevent ATS parsing failures"
        )
    if cv.page_count > 3:
        score -= 15.0
        issues.append(
            f"CV is {cv.page_count} pages — aim for 1–2 pages (junior/mid) "
            f"or up to 3 pages (senior/principal)"
        )

    return max(score, 0.0)
