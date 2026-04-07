"""
MarketForge AI — LLM Cost Tracker.

Intercepts all LangChain LLM calls, accumulates spend per run/department,
and triggers a circuit-breaker if a single run exceeds the budget cap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from langchain_core.callbacks import BaseCallbackHandler

from marketforge.config.settings import settings

logger = structlog.get_logger(__name__)

# Gemini pricing (USD per token) — update when pricing changes
GEMINI_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.15  / 1_000_000,  "output": 0.60  / 1_000_000},
    "gemini-2.5-pro":   {"input": 1.25  / 1_000_000,  "output": 10.00 / 1_000_000},
}
DEFAULT_PRICING = GEMINI_PRICING["gemini-2.5-flash"]


@dataclass
class CostEntry:
    agent_name:    str
    department:    str
    model:         str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float


@dataclass
class CostTracker:
    """
    Thread-safe (within a single async event loop) cost accumulator.
    Injected as a LangChain callback into every LLM call.
    """
    run_id:        str
    cost_cap_usd:  float = field(default_factory=lambda: settings.llm.cost_cap_usd)

    _entries:      list[CostEntry] = field(default_factory=list, init=False)
    _total_usd:    float           = field(default=0.0, init=False)
    _tripped:      bool            = field(default=False, init=False)

    def record(self, agent_name: str, department: str, model: str,
               input_tokens: int, output_tokens: int) -> None:
        pricing = GEMINI_PRICING.get(model, DEFAULT_PRICING)
        cost    = input_tokens * pricing["input"] + output_tokens * pricing["output"]

        self._total_usd += cost
        self._entries.append(CostEntry(
            agent_name=agent_name, department=department, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost,
        ))

        if self._total_usd >= self.cost_cap_usd and not self._tripped:
            self._tripped = True
            logger.warning(
                "cost_tracker.circuit_breaker.tripped",
                total_usd=round(self._total_usd, 4),
                cap_usd=self.cost_cap_usd,
                run_id=self.run_id,
            )

        logger.debug(
            "cost_tracker.record",
            agent=agent_name, model=model,
            cost_usd=round(cost, 6), total_usd=round(self._total_usd, 4),
        )

    @property
    def is_over_budget(self) -> bool:
        return self._tripped

    @property
    def total_usd(self) -> float:
        return self._total_usd

    @property
    def summary(self) -> dict[str, Any]:
        by_dept: dict[str, float] = {}
        for e in self._entries:
            by_dept[e.department] = by_dept.get(e.department, 0.0) + e.cost_usd
        return {
            "run_id":       self.run_id,
            "total_usd":    round(self._total_usd, 4),
            "cap_usd":      self.cost_cap_usd,
            "tripped":      self._tripped,
            "calls":        len(self._entries),
            "by_department": {k: round(v, 4) for k, v in by_dept.items()},
        }

    def persist(self) -> None:
        """Write per-call cost entries to market.cost_log."""
        try:
            from datetime import datetime
            from sqlalchemy import text
            from marketforge.memory.postgres import get_sync_engine
            engine = get_sync_engine()
            is_sqlite = engine.dialect.name == "sqlite"
            table = "cost_log" if is_sqlite else "market.cost_log"
            now = datetime.utcnow().isoformat()
            with engine.connect() as conn:
                for e in self._entries:
                    conn.execute(text(f"""
                        INSERT INTO {table}
                            (run_id, agent_name, model, input_tokens, output_tokens, cost_usd, logged_at)
                        VALUES (:rid, :agent, :model, :it, :ot, :cost, :now)
                    """), {
                        "rid": self.run_id, "agent": e.agent_name, "model": e.model,
                        "it": e.input_tokens, "ot": e.output_tokens,
                        "cost": round(e.cost_usd, 8), "now": now,
                    })
                conn.commit()
        except Exception as exc:
            logger.warning("cost_tracker.persist.error", error=str(exc))


class CostTrackerCallback(BaseCallbackHandler):
    """
    LangChain callback that auto-records token usage into a CostTracker.
    Attach to any ChatGoogleGenerativeAI instance via the `callbacks` parameter.
    """

    def __init__(self, tracker: CostTracker, agent_name: str, department: str, model: str) -> None:
        super().__init__()
        self._tracker    = tracker
        self._agent_name = agent_name
        self._department = department
        self._model      = model

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            usage = getattr(response, "llm_output", {}) or {}
            token_usage = usage.get("token_usage", {}) or {}
            input_t  = token_usage.get("prompt_tokens", 0)
            output_t = token_usage.get("completion_tokens", 0)
            if input_t or output_t:
                self._tracker.record(
                    self._agent_name, self._department, self._model,
                    input_t, output_t,
                )
        except Exception:
            pass  # never crash the pipeline over telemetry
