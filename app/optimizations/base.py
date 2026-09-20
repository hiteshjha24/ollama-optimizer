"""Optimization plugin interface.

Every optimization technique is a small class with a stable surface:

    id / name / category / description / parameters
    applicability(ctx)            -> Applicability
    generate_configurations(ctx)  -> list[RunConfig]     (coarse phase)
    refine(ctx, results)          -> list[RunConfig]     (fine phase, optional)
    evaluate(...)                 -> delegated to the evaluation engine

Adding a technique means writing one class and registering it; nothing else in
the application needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Applicability:
    applicable: bool
    reason: str
    status: str = "ok"   # ok | not_tested | informational

    @classmethod
    def ok(cls, reason: str = "") -> "Applicability":
        return cls(True, reason or "Supported for this model and runtime.")

    @classmethod
    def skip(cls, reason: str) -> "Applicability":
        return cls(False, reason, status="not_tested")

    @classmethod
    def info(cls, reason: str) -> "Applicability":
        return cls(False, reason, status="informational")


@dataclass
class RunConfig:
    """One benchmark-only configuration. Never written back to Ollama."""

    key: str
    label: str
    optimization_id: str
    category: str
    rationale: str
    phase: str = "coarse"            # baseline | coarse | fine | validation
    options: Dict[str, Any] = field(default_factory=dict)
    prompt: Optional[str] = None     # None -> the experiment's prompt
    system_prompt: Optional[str] = None
    model_override: Optional[str] = None
    stream: Optional[bool] = None    # None -> experiment default
    fmt: Optional[str] = None        # "json" to use Ollama's JSON mode
    keep_alive: Optional[str] = None
    is_baseline: bool = False
    tags: List[str] = field(default_factory=list)
    expectations_override: Optional[Dict[str, Any]] = None

    def to_db(self) -> Dict[str, Any]:
        return {
            "optimization_id": self.optimization_id,
            "category": self.category,
            "label": self.label,
            "phase": self.phase,
            "is_baseline": self.is_baseline,
            "options": self.persisted_options(),
            "prompt": self.prompt or "",
            "system_prompt": self.system_prompt,
            "rationale": self.rationale,
        }

    def persisted_options(self) -> Dict[str, Any]:
        """Options plus the non-option switches, so a run is reproducible."""
        data = dict(self.options)
        if self.stream is not None:
            data["_stream"] = self.stream
        if self.fmt:
            data["_format"] = self.fmt
        if self.keep_alive:
            data["_keep_alive"] = self.keep_alive
        if self.model_override:
            data["_model"] = self.model_override
        return data

    def signature(self) -> str:
        """Identity used to avoid running the same configuration twice."""
        parts = [
            self.model_override or "",
            self.prompt or "",
            self.system_prompt or "",
            self.fmt or "",
            str(self.stream),
            ";".join(f"{k}={self.options[k]}" for k in sorted(self.options)),
        ]
        return "|".join(parts)


@dataclass
class OptimizationContext:
    """Everything an optimization needs to propose configurations."""

    model: str
    prompt: str
    system_prompt: Optional[str]
    expectations: Dict[str, Any]
    capabilities: Dict[str, Any]
    model_meta: Dict[str, Any]
    available_models: List[Dict[str, Any]]
    max_tokens: int
    seed: Optional[int] = 42
    host_info: Dict[str, Any] = field(default_factory=dict)
    prompt_category: Optional[str] = None

    @property
    def baseline_options(self) -> Dict[str, Any]:
        """The model's own defaults, with only the output cap applied.

        We deliberately do not inject temperature/top_p/top_k here: leaving them
        out means Ollama uses the model's Modelfile defaults, which is the
        honest definition of "baseline".
        """
        return {"num_predict": int(self.max_tokens)}

    def default_param(self, name: str, fallback: Any) -> Any:
        """The model's documented default for a parameter, when /api/show has it."""
        defaults = self.capabilities.get("default_parameters") or {}
        value = defaults.get(name)
        return value if value is not None else fallback


@dataclass
class ConfigOutcome:
    """Aggregated result of one configuration, handed to ``refine()``."""

    config: RunConfig
    aggregate: Dict[str, Any]

    @property
    def quality(self) -> Optional[float]:
        return (self.aggregate.get("quality") or {}).get("mean")

    @property
    def latency(self) -> Optional[float]:
        return (self.aggregate.get("latency") or {}).get("median")

    @property
    def tokens_per_second(self) -> Optional[float]:
        return (self.aggregate.get("tokens_per_second") or {}).get("mean")

    @property
    def consistency(self) -> Optional[float]:
        return (self.aggregate.get("consistency") or {}).get("score")

    @property
    def failure_rate(self) -> float:
        return float(self.aggregate.get("failure_rate") or 0.0)


class Optimization:
    """Base class for all optimization techniques."""

    id: str = "optimization"
    name: str = "Optimization"
    category: str = "generic"
    description: str = ""
    parameters: List[str] = []
    #: how many of this optimization's coarse configs may be promoted to fine search
    fine_budget: int = 2

    def applicability(self, ctx: OptimizationContext) -> Applicability:
        return Applicability.ok()

    def generate_configurations(self, ctx: OptimizationContext) -> List[RunConfig]:
        raise NotImplementedError

    def refine(self, ctx: OptimizationContext,
               results: List[ConfigOutcome]) -> List[RunConfig]:
        return []

    def describe(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "parameters": self.parameters,
        }


def best_by_quality(results: List[ConfigOutcome]) -> Optional[ConfigOutcome]:
    scored = [r for r in results if r.quality is not None and r.failure_rate < 1.0]
    if not scored:
        return None
    return max(scored, key=lambda r: (r.quality, -(r.latency or 0)))
