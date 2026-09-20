"""Runtime / inference optimization.

Only settings that the installed Ollama build documents are tested, and only
when this machine gives us enough information to choose a sane value. Anything
that cannot be tested safely is reported as **Not tested** with the reason,
rather than being quietly skipped or, worse, described as if it had been run.

Explicitly *not* tested by default, and why:

* ``num_gpu`` - the right number of offloaded layers depends on the GPU, the
  model and what else is resident. Forcing a value would reload the model and
  could push it entirely onto the CPU, distorting every later measurement.
* ``keep_alive: 0`` - unloading the model between runs would make the following
  configuration pay the load cost, which would corrupt the comparison.
* Parallel request handling (``OLLAMA_NUM_PARALLEL``) - a server-level
  environment variable. Changing it means restarting the Ollama service, which
  this application will not do to a user's installation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Applicability, Optimization, OptimizationContext, RunConfig


class RuntimeOptimization(Optimization):
    id = "runtime"
    name = "Runtime / inference"
    category = "runtime"
    description = (
        "Tests runtime-level settings: streaming versus a single buffered response, "
        "CPU thread count, and batch size. Settings that would require reloading the "
        "model onto different hardware, or restarting the Ollama server, are reported "
        "as not tested."
    )
    parameters = ["num_thread", "num_batch", "streaming"]
    fine_budget = 1

    def applicability(self, ctx: OptimizationContext) -> Applicability:
        return Applicability.ok(
            "Streaming and batch settings are testable on any Ollama install; thread "
            "count is tested when the CPU topology is known."
        )

    def not_tested(self, ctx: OptimizationContext) -> List[Dict[str, str]]:
        """Runtime items deliberately left untested, surfaced in the report."""
        items = [
            {
                "item": "num_gpu (GPU layer offload)",
                "status": "Not tested",
                "reason": ("Hardware dependent. Choosing a layer count blind would force "
                           "a model reload and could move the model onto the CPU, "
                           "invalidating the rest of the benchmark."),
            },
            {
                "item": "keep_alive = 0 (unload between runs)",
                "status": "Not tested",
                "reason": ("Unloading the model after each request would charge the next "
                           "configuration with the load time, corrupting the comparison. "
                           f"All runs use keep_alive from settings so the model stays "
                           "resident."),
            },
            {
                "item": "OLLAMA_NUM_PARALLEL / OLLAMA_MAX_LOADED_MODELS",
                "status": "Not tested",
                "reason": ("Server-level environment variables. Applying them requires "
                           "restarting the Ollama service, which this application will "
                           "not do to your installation."),
            },
            {
                "item": "Concurrent request throughput",
                "status": "Not tested",
                "reason": ("Benchmark requests are issued with the configured concurrency "
                           "limit (default 1) so that latency measurements reflect the "
                           "model, not queueing against itself."),
            },
        ]
        if not ctx.host_info.get("cpu_count_physical"):
            items.append({
                "item": "num_thread",
                "status": "Not tested",
                "reason": ("CPU core count is unknown on this system (psutil unavailable), "
                           "so no sensible thread value could be chosen."),
            })
        return items

    def generate_configurations(self, ctx: OptimizationContext) -> List[RunConfig]:
        base = ctx.baseline_options
        configs: List[RunConfig] = []

        configs.append(RunConfig(
            key="rt_non_streaming",
            label="Non-streaming response",
            optimization_id=self.id,
            category=self.category,
            phase="coarse",
            options=dict(base),
            stream=False,
            rationale=(
                "Requests the whole response in one payload instead of a token stream. "
                "Streaming costs a little per-chunk overhead but gives the user output "
                "immediately; this measures whether the trade-off is visible in total "
                "wall time. Time-to-first-token cannot be measured without streaming, "
                "so it is reported as unavailable for this configuration."
            ),
            tags=["streaming"],
        ))

        physical = ctx.host_info.get("cpu_count_physical")
        logical = ctx.host_info.get("cpu_count_logical")
        thread_values = []
        if isinstance(physical, int) and physical >= 2:
            thread_values.append(physical)
        if isinstance(logical, int) and logical and logical != physical and logical >= 4:
            thread_values.append(max(2, logical // 2))
        for value in sorted(set(thread_values)):
            configs.append(RunConfig(
                key=f"rt_num_thread_{value}",
                label=f"num_thread = {value}",
                optimization_id=self.id,
                category=self.category,
                phase="coarse",
                options={**base, "num_thread": value},
                rationale=(
                    f"Pins inference to {value} CPU threads (this machine reports "
                    f"{physical} physical / {logical} logical cores). More threads are "
                    "not always faster: oversubscription causes contention, and on a "
                    "GPU-resident model the setting may have no effect at all. Changing "
                    "it reloads the model."
                ),
                tags=["num_thread", "reload"],
            ))

        for value in (256, 1024):
            configs.append(RunConfig(
                key=f"rt_num_batch_{value}",
                label=f"num_batch = {value}",
                optimization_id=self.id,
                category=self.category,
                phase="coarse",
                options={**base, "num_batch": value},
                rationale=(
                    f"Sets the prompt-processing batch size to {value} (Ollama's default "
                    "is 512). Larger batches can speed up prompt evaluation at the cost "
                    "of memory; smaller batches can help on constrained hardware. The "
                    "effect is mostly on prompt-eval time, so it matters more for long "
                    "prompts than for long outputs."
                ),
                tags=["num_batch", "reload"],
            ))
        return configs
