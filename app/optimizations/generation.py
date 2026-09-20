"""Baseline and generation-parameter optimizations.

Search strategy (documented in the report so the user can audit it):

    1. baseline           - model defaults, output cap only
    2. coarse, one factor at a time (OFAT) over temperature, top_p, top_k and
       repeat_penalty, so cost grows linearly rather than as a Cartesian product
    3. fine               - the best value of each factor is combined, and the
       winning factor is probed at neighbouring values
    4. validation         - handled by the engine on the top-ranked configs

With four factors this is 13 coarse configurations instead of the 180 a full
grid would require.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import (Applicability, ConfigOutcome, Optimization, OptimizationContext,
                   RunConfig, best_by_quality)

TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
TOP_P = [0.7, 0.9, 1.0]
TOP_K = [20, 40, 100]
REPEAT_PENALTY = [1.0, 1.1, 1.2]


class BaselineOptimization(Optimization):
    id = "baseline"
    name = "Baseline"
    category = "baseline"
    description = (
        "The model's own default generation settings with only an output-token cap "
        "applied. Every other configuration is compared against this."
    )
    parameters = ["num_predict"]

    def generate_configurations(self, ctx: OptimizationContext) -> List[RunConfig]:
        return [RunConfig(
            key="baseline",
            label="Baseline (model defaults)",
            optimization_id=self.id,
            category=self.category,
            phase="baseline",
            is_baseline=True,
            options=ctx.baseline_options,
            rationale=(
                "No sampling parameters are sent, so Ollama applies the values baked "
                "into the model's Modelfile. This is the reference point for every "
                "comparison in this report."
            ),
            tags=["reference"],
        )]


class GenerationParameterOptimization(Optimization):
    id = "generation_params"
    name = "Generation parameters"
    category = "generation"
    description = (
        "Sweeps the sampling parameters Ollama exposes (temperature, top_p, top_k, "
        "repeat_penalty) one factor at a time, then combines the best value of each."
    )
    parameters = ["temperature", "top_p", "top_k", "repeat_penalty", "seed"]
    fine_budget = 3

    def applicability(self, ctx: OptimizationContext) -> Applicability:
        return Applicability.ok(
            "All four sampling parameters are standard Ollama generation options."
        )

    def generate_configurations(self, ctx: OptimizationContext) -> List[RunConfig]:
        base = ctx.baseline_options
        configs: List[RunConfig] = []

        def add(param: str, value: Any, note: str) -> None:
            configs.append(RunConfig(
                key=f"gen_{param}_{value}",
                label=f"{param} = {value}",
                optimization_id=self.id,
                category=self.category,
                phase="coarse",
                options={**base, param: value},
                rationale=note,
                tags=["ofat", param],
            ))

        for value in TEMPERATURES:
            add("temperature", value,
                "Temperature controls how sharply the next-token distribution is "
                "sampled. Low values make output more deterministic and usually help "
                "extraction and format compliance; high values add variety that can "
                "help creative work. Tested because the right setting is task "
                "dependent, not universal.")
        for value in TOP_P:
            add("top_p", value,
                "Nucleus sampling keeps the smallest set of tokens whose cumulative "
                "probability reaches top_p. Tightening it trims the low-probability "
                "tail, which can reduce drift but also reduce variety.")
        for value in TOP_K:
            add("top_k", value,
                "top_k caps the candidate pool to the k most likely tokens. Smaller "
                "values constrain the model further; the interaction with top_p means "
                "the effect has to be measured rather than assumed.")
        for value in REPEAT_PENALTY:
            add("repeat_penalty", value,
                "Penalises tokens that already appeared. Higher values suppress "
                "looping but can distort required repetition such as JSON keys or "
                "list markers, so both directions are tested.")

        # Determinism check: a fixed seed at the model's default sampling settings.
        configs.append(RunConfig(
            key="gen_seed_fixed",
            label=f"seed = {ctx.seed} (fixed)",
            optimization_id=self.id,
            category=self.category,
            phase="coarse",
            options={**base, "seed": int(ctx.seed or 42)},
            rationale=(
                "A fixed seed makes sampling reproducible without changing the "
                "distribution. Tested to measure how much of the run-to-run variation "
                "comes from sampling randomness rather than from the configuration."
            ),
            tags=["determinism", "seed"],
        ))
        return configs

    def refine(self, ctx: OptimizationContext,
               results: List[ConfigOutcome]) -> List[RunConfig]:
        """Combine the best value of each swept factor, then probe around it."""
        base = ctx.baseline_options
        by_param: Dict[str, List[ConfigOutcome]] = {}
        for outcome in results:
            for param in ("temperature", "top_p", "top_k", "repeat_penalty"):
                if param in outcome.config.tags:
                    by_param.setdefault(param, []).append(outcome)

        winners: Dict[str, Any] = {}
        notes: List[str] = []
        for param, outcomes in by_param.items():
            best = best_by_quality(outcomes)
            if best is None:
                continue
            value = best.config.options.get(param)
            winners[param] = value
            notes.append(f"{param}={value} (quality {best.quality:.2f})")

        if not winners:
            return []

        configs: List[RunConfig] = [RunConfig(
            key="gen_combined_best",
            label="Best value of each parameter, combined",
            optimization_id=self.id,
            category=self.category,
            phase="fine",
            options={**base, **winners},
            rationale=(
                "Each parameter was swept independently; the highest-scoring value of "
                "each is combined here to test whether the gains add up. Selected from "
                "the coarse phase: " + ", ".join(notes) + "."
            ),
            tags=["fine", "combined"],
        )]

        # Probe either side of the winning temperature, the most influential factor.
        if "temperature" in winners:
            best_temp = float(winners["temperature"])
            for delta in (-0.15, 0.15):
                probe = round(best_temp + delta, 2)
                if probe < 0 or probe > 1.5:
                    continue
                configs.append(RunConfig(
                    key=f"gen_fine_temp_{probe}",
                    label=f"temperature = {probe} (fine probe)",
                    optimization_id=self.id,
                    category=self.category,
                    phase="fine",
                    options={**base, **winners, "temperature": probe},
                    rationale=(
                        f"Temperature {best_temp} won the coarse sweep, so the "
                        f"neighbouring value {probe} is tested to check whether the "
                        "coarse grid landed on a local peak or merely near one."
                    ),
                    tags=["fine", "temperature"],
                ))
        return configs
