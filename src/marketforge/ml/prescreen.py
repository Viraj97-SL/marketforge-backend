"""
MarketForge AI — ML Pre-screen Gate

Three-signal ensemble that filters job descriptions before NLP/LLM processing.
Runs entirely on CPU in under 5ms per job.

Signal 1 — Dense (SBERT cosine)       weight 0.50
Signal 2 — Sparse (BM25 Okapi)        weight 0.30
Signal 3 — Exact skill overlap        weight 0.20

Combined score >= threshold (default 0.28) → process job
Combined score <  threshold             → skip (not an AI/ML role)
"""
from __future__ import annotations

import re
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from marketforge.models.job import RawJob

logger = structlog.get_logger(__name__)

_SBERT_MODEL = "all-MiniLM-L6-v2"

# Canonical AI/ML skill tokens for exact overlap scoring
_CORE_SKILLS = frozenset({
    "python", "pytorch", "tensorflow", "machine learning", "deep learning",
    "neural network", "llm", "langchain", "langgraph", "rag", "transformer",
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "nlp", "computer vision",
    "mlops", "airflow", "kubeflow", "mlflow", "sagemaker", "vertex",
    "data science", "reinforcement learning", "fine-tuning", "embeddings",
    "vector database", "fastapi", "huggingface", "bert", "gpt", "gemini",
    "claude", "llama", "mistral", "diffusion", "multimodal", "agent",
})


def _tokenise(text: str) -> list[str]:
    return re.findall(r"\b[a-z][a-z0-9+#.\-]{1,}\b", text.lower())


class MLPrescreen:
    """
    Hybrid dense + sparse + exact pre-screen gate.

    Usage:
        screen = MLPrescreen(threshold=0.28)
        screen.fit(all_jobs)        # fit BM25 once per batch
        ok, breakdown = screen.should_process(job)
    """

    W_EMBED = 0.50
    W_BM25  = 0.30
    W_SKILL = 0.20

    def __init__(self, threshold: float = 0.28) -> None:
        self.threshold = threshold
        self._profile_text = " ".join(_CORE_SKILLS)
        self._bm25: object | None = None
        self._bm25_job_ids: list[str] = []
        self._bm25_scores: np.ndarray | None = None

    @cached_property
    def _embedder(self):
        from sentence_transformers import SentenceTransformer
        logger.info("prescreen.embedder.loading", model=_SBERT_MODEL)
        m = SentenceTransformer(_SBERT_MODEL, device="cpu")
        logger.info("prescreen.embedder.ready")
        return m

    @cached_property
    def _profile_embedding(self) -> np.ndarray:
        return self._embedder.encode(self._profile_text, normalize_embeddings=True)

    def fit(self, jobs: list[RawJob]) -> None:
        """Fit BM25 index once over the full job batch."""
        try:
            from rank_bm25 import BM25Okapi
            corpus = [_tokenise(f"{j.title} {j.description}") for j in jobs]
            self._bm25         = BM25Okapi(corpus)
            self._bm25_job_ids = [j.job_id for j in jobs]
            query_tokens       = _tokenise(self._profile_text)
            raw_scores: np.ndarray = self._bm25.get_scores(query_tokens)
            max_s = raw_scores.max() if raw_scores.max() > 0 else 1.0
            self._bm25_scores  = raw_scores / max_s
            logger.info("prescreen.bm25.fitted", n=len(jobs))
        except ImportError:
            logger.warning("rank_bm25.not_installed — BM25 signal disabled")

    def score(self, job: RawJob) -> tuple[float, dict[str, float]]:
        text = f"{job.title} {job.description[:1500]}"

        # Dense signal
        try:
            jd_emb  = self._embedder.encode(text, normalize_embeddings=True)
            embed_s = float(np.dot(self._profile_embedding, jd_emb))
        except Exception:
            embed_s = 0.5

        # BM25 signal
        bm25_s = 0.5
        if self._bm25_scores is not None and job.job_id in self._bm25_job_ids:
            idx    = self._bm25_job_ids.index(job.job_id)
            bm25_s = float(self._bm25_scores[idx])

        # Exact skill overlap
        jd_tokens  = set(_tokenise(text))
        skill_s    = len(_CORE_SKILLS & jd_tokens) / max(len(_CORE_SKILLS), 1)

        combined = self.W_EMBED * embed_s + self.W_BM25 * bm25_s + self.W_SKILL * skill_s
        return combined, {
            "embed_sim":    round(embed_s,  3),
            "bm25_norm":    round(bm25_s,   3),
            "skill_overlap":round(skill_s,  3),
            "combined":     round(combined, 3),
        }

    def should_process(self, job: RawJob) -> tuple[bool, dict[str, float]]:
        combined, breakdown = self.score(job)
        return combined >= self.threshold, breakdown

    def batch_filter(self, jobs: list[RawJob]) -> tuple[list[RawJob], list[RawJob]]:
        """
        Returns (to_process, skipped).
        Fits BM25 once internally for efficiency.
        """
        self.fit(jobs)
        to_process, skipped = [], []
        for job in jobs:
            ok, _ = self.should_process(job)
            (to_process if ok else skipped).append(job)
        logger.info(
            "prescreen.batch_filter",
            total=len(jobs),
            passed=len(to_process),
            filtered=len(skipped),
            filter_rate=f"{len(skipped)/max(len(jobs),1):.0%}",
        )
        return to_process, skipped
