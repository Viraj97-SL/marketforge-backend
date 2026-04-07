"""
MarketForge AI — Department 1: DeduplicationCoordinatorAgent

Cross-source deduplication using three complementary signals:
  Signal 1 — Exact hash (title + company + location, normalised)
  Signal 2 — MinHash LSH near-duplicate detection (reworded titles)
  Signal 3 — SBERT cosine similarity (> 0.97 threshold, very high precision)

The agent generates a dedup report showing source overlap rates,
which feeds back into the DataCollectionLeadAgent's source prioritisation.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import structlog

from marketforge.agents.base import DeepAgent
from marketforge.models.job import RawJob

logger = structlog.get_logger(__name__)


class DeduplicationCoordinatorAgent(DeepAgent):
    """
    Deep Agent for cross-source deduplication.

    plan():    Reads prior run's overlap_report to identify which source
               pairs have the highest duplication rate. If source A shares
               > 60% of its jobs with source B, plan() recommends de-prioritising
               source A in the next ingestion run. Checks cross-run dedup store
               (Redis / PostgreSQL) to filter already-seen jobs.

    execute(): Applies three dedup signals in sequence:
               1. Exact hash: O(1) set lookup on normalised (title, company, location)
               2. MinHash LSH: approximates Jaccard similarity on character 4-grams
                  of the title string. Catches "Senior ML Engineer" ≈ "Sr. ML Engineer".
               3. SBERT cosine: for jobs that passed signals 1+2 but look suspicious
                  (same company, similar location), compute embedding cosine.
                  Only runs when two jobs share the same company to minimise cost.

    reflect(): Computes overlap_report: for each source pair, what fraction of
               source A's jobs were duplicates of source B's? Stores this in
               adaptive_params for the lead agent's planning phase.

    output():  Returns the deduplicated job list + overlap_report.
    """

    agent_id   = "dedup_coordinator_v1"
    department = "data_collection"

    # SBERT is lazy-loaded only when needed
    _embedder = None

    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        adaptive     = state.get("adaptive_params", {})
        overlap_rpt  = adaptive.get("overlap_report", {})

        # Identify over-lapping source pairs (> 60% overlap) to flag
        high_overlap_pairs = [
            pair for pair, rate in overlap_rpt.items() if rate > 0.6
        ]
        if high_overlap_pairs:
            logger.info(f"{self.agent_id}.plan.high_overlap", pairs=high_overlap_pairs)

        return {
            "adaptive":           adaptive,
            "high_overlap_pairs": high_overlap_pairs,
            "raw_jobs":           context.get("raw_jobs", []),
        }

    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        raw_jobs: list[RawJob] = plan.get("raw_jobs", [])
        if not raw_jobs:
            return {"deduplicated": [], "removed": 0, "overlap_data": {}}

        # ── Signal 1: Exact hash ──────────────────────────────────────────────
        seen_hashes: set[str] = set()
        after_exact: list[RawJob] = []
        for job in raw_jobs:
            if job.dedup_hash not in seen_hashes:
                seen_hashes.add(job.dedup_hash)
                after_exact.append(job)

        exact_removed = len(raw_jobs) - len(after_exact)
        logger.info(f"{self.agent_id}.signal1.exact", removed=exact_removed, remaining=len(after_exact))

        # ── Signal 2: MinHash LSH ────────────────────────────────────────────
        after_lsh, lsh_removed = self._minhash_dedup(after_exact)
        logger.info(f"{self.agent_id}.signal2.lsh", removed=lsh_removed, remaining=len(after_lsh))

        # ── Signal 3: SBERT cosine (same company only) ───────────────────────
        after_sbert, sbert_removed = self._sbert_dedup(after_lsh)
        logger.info(f"{self.agent_id}.signal3.sbert", removed=sbert_removed, remaining=len(after_sbert))

        # ── Cross-run dedup via store ─────────────────────────────────────────
        from marketforge.memory.postgres import DedupStore
        store = DedupStore()
        new_jobs = store.filter_new(after_sbert)

        # Overlap data for reflect()
        overlap_data = self._compute_overlap(raw_jobs, new_jobs)

        return {
            "deduplicated":  new_jobs,
            "removed":       len(raw_jobs) - len(new_jobs),
            "exact_removed": exact_removed,
            "lsh_removed":   lsh_removed,
            "sbert_removed": sbert_removed,
            "xrun_removed":  len(after_sbert) - len(new_jobs),
            "overlap_data":  overlap_data,
        }

    def _minhash_dedup(self, jobs: list[RawJob]) -> tuple[list[RawJob], int]:
        """
        MinHash LSH deduplication on title 4-grams.
        Uses a simple shingling + hash approach (no external dep).
        """
        def shingles(text: str, k: int = 4) -> set[int]:
            text = text.lower().replace(" ", "")
            return {int(hashlib.md5(text[i:i+k].encode()).hexdigest(), 16) % (2**32) for i in range(len(text) - k + 1)}

        def minhash_sig(shingle_set: set[int], n: int = 100) -> list[int]:
            sig = []
            for seed in range(n):
                min_val = min((((s * seed + seed * 7) % (2**31 - 1)) for s in shingle_set), default=0)
                sig.append(min_val)
            return sig

        def jaccard_approx(sig1: list[int], sig2: list[int]) -> float:
            return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)

        seen: list[tuple[str, list[int]]] = []
        deduped: list[RawJob] = []

        for job in jobs:
            title_sh = shingles(job.title)
            if not title_sh:
                deduped.append(job)
                continue
            sig = minhash_sig(title_sh)
            duplicate = False
            for prev_company, prev_sig in seen:
                if prev_company == job.company:
                    if jaccard_approx(sig, prev_sig) > 0.80:
                        duplicate = True
                        break
            if not duplicate:
                seen.append((job.company, sig))
                deduped.append(job)

        return deduped, len(jobs) - len(deduped)

    def _sbert_dedup(self, jobs: list[RawJob]) -> tuple[list[RawJob], int]:
        """
        SBERT cosine dedup for same-company jobs.
        Only loads the embedder model when there are candidates to check.
        """
        from collections import defaultdict
        company_groups: dict[str, list[tuple[int, RawJob]]] = defaultdict(list)
        for idx, job in enumerate(jobs):
            company_groups[job.company.lower()].append((idx, job))

        # Identify companies with multiple postings
        multi: list[tuple[int, RawJob]] = []
        for group in company_groups.values():
            if len(group) > 1:
                multi.extend(group)

        if not multi:
            return jobs, 0

        # Lazy-load embedder
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                DeduplicationCoordinatorAgent._embedder = SentenceTransformer(
                    "all-MiniLM-L6-v2", device="cpu"
                )
            except Exception as exc:
                logger.warning("sbert_dedup.load_failed", error=str(exc))
                return jobs, 0

        texts = [f"{job.title} {job.company}" for _, job in multi]
        try:
            embeddings = self._embedder.encode(texts, normalize_embeddings=True)
        except Exception:
            return jobs, 0

        import numpy as np
        remove_idx: set[int] = set()
        for i, (idx_i, _) in enumerate(multi):
            if idx_i in remove_idx:
                continue
            for j, (idx_j, _) in enumerate(multi):
                if i >= j or idx_j in remove_idx:
                    continue
                cosine = float(np.dot(embeddings[i], embeddings[j]))
                if cosine > 0.97:
                    remove_idx.add(idx_j)

        deduped = [j for i, j in enumerate(jobs) if i not in {idx for idx, _ in multi if idx in remove_idx}]
        return deduped, len(jobs) - len(deduped)

    def _compute_overlap(
        self, original: list[RawJob], final: list[RawJob]
    ) -> dict[str, float]:
        """
        For each source pair, compute what fraction of source A's jobs
        also appeared in source B (by hash).
        """
        source_hashes: dict[str, set[str]] = defaultdict(set)
        for job in original:
            source_hashes[job.source].add(job.dedup_hash)

        final_hashes = {j.dedup_hash for j in final}
        overlap: dict[str, float] = {}
        sources = list(source_hashes.keys())
        for i, src_a in enumerate(sources):
            for src_b in sources[i+1:]:
                common = source_hashes[src_a] & source_hashes[src_b]
                if common:
                    overlap[f"{src_a}|{src_b}"] = round(
                        len(common) / max(len(source_hashes[src_a]), 1), 3
                    )
        return overlap

    async def reflect(
        self,
        plan: dict[str, Any],
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        adaptive = plan.get("adaptive", {})
        adaptive["overlap_report"] = result.get("overlap_data", {})
        state["adaptive_params"]   = adaptive
        state["last_yield"]        = len(result.get("deduplicated", []))

        removed = result.get("removed", 0)
        total   = removed + len(result.get("deduplicated", []))
        dup_rate = removed / max(total, 1)

        quality = "good"
        if dup_rate > 0.7:
            quality = "warning"
            logger.warning(f"{self.agent_id}.high_dup_rate", rate=dup_rate)

        return {
            "quality":  quality,
            "dup_rate": round(dup_rate, 3),
            "yield":    len(result.get("deduplicated", [])),
            "notes":    f"removed={removed}, exact={result.get('exact_removed')}, lsh={result.get('lsh_removed')}, sbert={result.get('sbert_removed')}, xrun={result.get('xrun_removed')}",
        }

    async def output(self, result: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
        return {
            "deduped_jobs":  result.get("deduplicated", []),
            "dedup_report":  {
                "removed":    result.get("removed", 0),
                "dup_rate":   reflection.get("dup_rate", 0),
                "overlap":    result.get("overlap_data", {}),
            },
        }
