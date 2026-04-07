"""
MarketForge AI — Deep Agent Abstract Base Class.

Every agent in the system — Lead Agents and sub-agents alike — must
inherit from DeepAgent and implement the four lifecycle methods.

CRITICAL: A sub-agent is NOT a function that calls an API.
It is a stateful reasoning entity with:
  - plan():    analyses inputs, reads prior state, forms a strategy
  - execute(): multi-step execution using tools, APIs, LLMs
  - reflect(): self-evaluates quality against quantified signals,
               updates adaptive params, logs outcome to persistent state
  - output():  packages typed results for the parent agent or state

Only components that reason about what to do, decide how to do it, and
evaluate whether they did it well qualify as sub-agents. Single-call
API wrappers are tools, not agents.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog

from marketforge.memory.postgres import AgentStateStore
from marketforge.utils.cost_tracker import CostTracker

logger = structlog.get_logger(__name__)


class DeepAgent(ABC):
    """
    Abstract base class for all Deep Agents.

    Lifecycle:
        PLAN → EXECUTE → REFLECT → OUTPUT

    Each phase is async to allow concurrent sub-agent fan-outs.
    Persistent state is managed via AgentStateStore.
    """

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """
        Unique stable identifier, e.g. 'adzuna_deep_scout_v1'.
        Used as the primary key in market.agent_state.
        """
        ...

    @property
    @abstractmethod
    def department(self) -> str:
        """Parent department name, e.g. 'data_collection'."""
        ...

    # ── Optional: override to enable LLM calls ────────────────────────────────
    @property
    def uses_llm(self) -> bool:
        return False

    # ── Lifecycle methods (must implement) ────────────────────────────────────

    @abstractmethod
    async def plan(self, context: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """
        Analyse inputs and prior state; produce an execution plan.

        Args:
            context:  upstream data from the parent agent or pipeline state
            state:    this agent's own persistent state (loaded from DB)

        Returns:
            A plan dict that execute() will consume.
        """
        ...

    @abstractmethod
    async def execute(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the plan using tools, APIs, or LLMs.

        Args:
            plan:   the dict returned by plan()
            state:  this agent's own persistent state

        Returns:
            A result dict passed to reflect() and output().
        """
        ...

    @abstractmethod
    async def reflect(
        self,
        plan: dict[str, Any],
        result: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Self-evaluate quality of the execution result.

        Must:
        - Check at least one quantified quality signal (yield, parse_rate, etc.)
        - Update state["adaptive_params"] based on what was learned
        - Record a structured entry for state["reflection_log"]
        - Flag quality as "good" / "warning" / "poor"

        Returns:
            A reflection dict with at minimum {"quality": str, "notes": str}
        """
        ...

    @abstractmethod
    async def output(
        self,
        result: dict[str, Any],
        reflection: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Package the final typed output for the parent agent or pipeline state.

        Returns:
            A dict of state-key → value pairs.
        """
        ...

    # ── Orchestrator (called by parent agents) ────────────────────────────────

    async def run(
        self,
        context: dict[str, Any],
        cost_tracker: CostTracker | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full PLAN → EXECUTE → REFLECT → OUTPUT lifecycle.
        Handles persistent state loading/saving and structured logging.
        """
        start_ts = time.monotonic()
        state_store = AgentStateStore()

        # ── Load persistent state ────────────────────────────────────────────
        state = state_store.load(self.agent_id, self.department)
        logger.info(
            f"{self.agent_id}.lifecycle.start",
            run_count=state.get("run_count", 0),
            last_run=state.get("last_run_at"),
        )

        plan: dict[str, Any]       = {}
        result: dict[str, Any]     = {}
        reflection: dict[str, Any] = {}

        try:
            # ── PLAN ─────────────────────────────────────────────────────────
            plan = await self.plan(context, state)
            logger.info(f"{self.agent_id}.plan.done", plan_keys=list(plan.keys()))

            # ── Circuit breaker: abort if cost cap already blown ─────────────
            if cost_tracker and cost_tracker.is_over_budget:
                logger.warning(f"{self.agent_id}.skipped.cost_cap_reached")
                return {"skipped": True, "reason": "cost_cap_reached"}

            # ── EXECUTE ──────────────────────────────────────────────────────
            result = await self.execute(plan, state)
            logger.info(f"{self.agent_id}.execute.done", result_keys=list(result.keys()))

        except Exception as exc:
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
            logger.error(
                f"{self.agent_id}.execute.error",
                error=str(exc),
                consecutive_failures=state["consecutive_failures"],
            )
            reflection = {"quality": "poor", "notes": f"execution_error: {exc}"}
            result     = {"error": str(exc)}
        else:
            state["consecutive_failures"] = 0

        # ── REFLECT (always runs, even on failure) ────────────────────────────
        try:
            reflection = await self.reflect(plan, result, state)
        except Exception as exc:
            reflection = {"quality": "poor", "notes": f"reflect_error: {exc}"}
            logger.warning(f"{self.agent_id}.reflect.error", error=str(exc))

        # ── Update persistent state ───────────────────────────────────────────
        elapsed_ms = int((time.monotonic() - start_ts) * 1000)
        state["run_count"]   = state.get("run_count", 0) + 1
        state["last_run_at"] = __import__("datetime").datetime.utcnow().isoformat()
        state.setdefault("reflection_log", [])
        state["reflection_log"].append({**reflection, "duration_ms": elapsed_ms})
        if len(state["reflection_log"]) > 10:
            state["reflection_log"] = state["reflection_log"][-10:]

        state_store.save(state)

        logger.info(
            f"{self.agent_id}.lifecycle.done",
            quality=reflection.get("quality", "unknown"),
            duration_ms=elapsed_ms,
        )

        # ── OUTPUT ────────────────────────────────────────────────────────────
        try:
            return await self.output(result, reflection)
        except Exception as exc:
            logger.error(f"{self.agent_id}.output.error", error=str(exc))
            return {"error": str(exc), "reflection": reflection}

    # ── Escalation helper ─────────────────────────────────────────────────────
    def should_escalate(self, state: dict[str, Any], threshold: int = 3) -> bool:
        """True if consecutive failure count exceeds the threshold."""
        return state.get("consecutive_failures", 0) >= threshold
