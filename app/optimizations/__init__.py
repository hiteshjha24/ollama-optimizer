"""Optimization registry.

To add a technique: implement :class:`~app.optimizations.base.Optimization` and
append it to ``REGISTRY``. The engine, the UI strategy list and the report
sections all read from here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import (Applicability, ConfigOutcome, Optimization, OptimizationContext,
                   RunConfig)
from .context import ContextOptimization
from .generation import BaselineOptimization, GenerationParameterOptimization
from .prompting import PromptOptimization, SystemPromptOptimization
from .quantization import QuantizationOptimization, describe_quantization
from .runtime import RuntimeOptimization

BASELINE = BaselineOptimization()

REGISTRY: List[Optimization] = [
    GenerationParameterOptimization(),
    PromptOptimization(),
    SystemPromptOptimization(),
    ContextOptimization(),
    RuntimeOptimization(),
    QuantizationOptimization(),
]

#: strategy id -> optimization ids, used by the UI's optimization selector
STRATEGIES: Dict[str, List[str]] = {
    "baseline": [],
    "generation": ["generation_params"],
    "prompt": ["prompt_optimization"],
    "system_prompt": ["system_prompt"],
    "context": ["context"],
    "runtime": ["runtime"],
    "quantization": ["quantization"],
    "all": [opt.id for opt in REGISTRY],
}

STRATEGY_LABELS = {
    "baseline": "Baseline only",
    "generation": "Generation parameters",
    "prompt": "Prompt optimization",
    "system_prompt": "System prompt optimization",
    "context": "Context optimization",
    "runtime": "Runtime optimization",
    "quantization": "Quantization comparison",
    "all": "All optimizations",
}


def get_optimization(opt_id: str) -> Optional[Optimization]:
    if opt_id == BASELINE.id:
        return BASELINE
    for opt in REGISTRY:
        if opt.id == opt_id:
            return opt
    return None


def resolve_strategy(strategy: str, custom: Optional[List[str]] = None) -> List[Optimization]:
    """Map a strategy name (or an explicit list) to optimization instances."""
    if custom:
        chosen = [get_optimization(cid) for cid in custom]
        return [opt for opt in chosen if opt is not None and opt.id != BASELINE.id]
    ids = STRATEGIES.get(strategy, STRATEGIES["all"])
    return [opt for opt in REGISTRY if opt.id in ids]


def describe_all() -> List[Dict[str, object]]:
    return [BASELINE.describe()] + [opt.describe() for opt in REGISTRY]


__all__ = [
    "Applicability", "ConfigOutcome", "Optimization", "OptimizationContext",
    "RunConfig", "REGISTRY", "BASELINE", "STRATEGIES", "STRATEGY_LABELS",
    "get_optimization", "resolve_strategy", "describe_all", "describe_quantization",
]
