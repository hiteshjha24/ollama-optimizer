"""Experiment manager.

Runs the full pipeline on a background thread so the HTTP request returns
immediately:

    health check -> capability detection -> baseline -> coarse search ->
    fine search -> validation -> aggregation -> recommendation -> report -> PDF

Design notes:

* Concurrency is deliberately conservative (default 1). Local inference is
  resource bound; running configurations in parallel would make every latency
  measurement a measurement of queueing.
* One warm-up generation is issued before the baseline so the first measured
  run is not paying the model load cost. It is excluded from all statistics.
* A failed run is recorded and the experiment continues. An experiment only
  aborts when Ollama itself becomes unreachable or the model disappears.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import db
from .config import get_settings
from .evaluation import consistency_across_runs, evaluate_output, infer_expectations, llm_judge
from .logging_setup import get_logger
from .metrics import ResourceSampler, host_info, merge_resource_samples, summarize
from .ollama import (Cancelled, OllamaError, OllamaModelMissing, OllamaService,
                     OllamaUnavailable)
from .optimizations import (BASELINE, ConfigOutcome, OptimizationContext, RunConfig,
                            resolve_strategy)
from .optimizations.runtime import RuntimeOptimization
from .prompts import get_prompt
from .recommend import recommend
from .version import APP_VERSION

log = get_logger("engine")

MAX_CONSECUTIVE_FATAL = 3
VALIDATION_TOP_N = 2
VALIDATION_EXTRA_RUNS = 2


class ExperimentCancelled(RuntimeError):
    pass


# --------------------------------------------------------------------------
# progress tracking
# --------------------------------------------------------------------------

class Progress:
    """Ordered checklist mirrored to the database and pushed over SSE."""

    def __init__(self, emit: Callable[[Dict[str, Any]], None]):
        self.steps: List[Dict[str, Any]] = []
        self._emit = emit
        self.started = time.time()
        self.total_runs = 0
        self.completed_runs = 0
        self.current = ""
        self._lock = threading.Lock()

    def add(self, step_id: str, label: str) -> None:
        with self._lock:
            self.steps.append({"id": step_id, "label": label, "status": "pending",
                               "detail": "", "started_at": None, "finished_at": None})

    def ensure(self, step_id: str, label: str) -> None:
        if not any(s["id"] == step_id for s in self.steps):
            self.add(step_id, label)

    def set(self, step_id: str, status: str, detail: str = "") -> None:
        with self._lock:
            for step in self.steps:
                if step["id"] == step_id:
                    step["status"] = status
                    if detail:
                        step["detail"] = detail
                    if status == "running" and not step["started_at"]:
                        step["started_at"] = time.time()
                    if status in ("done", "failed", "skipped"):
                        step["finished_at"] = time.time()
                    break
        self.push()

    def run_done(self, current: str = "") -> None:
        with self._lock:
            self.completed_runs += 1
            if current:
                self.current = current
        self.push()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self.started
            fraction = (self.completed_runs / self.total_runs) if self.total_runs else 0.0
            eta = None
            if self.completed_runs >= 2 and fraction > 0:
                eta = round((elapsed / fraction) - elapsed, 1)
            return {
                "steps": [dict(s) for s in self.steps],
                "total_runs": self.total_runs,
                "completed_runs": self.completed_runs,
                "remaining_runs": max(0, self.total_runs - self.completed_runs),
                "fraction": round(fraction, 4),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": eta,
                "current": self.current,
            }

    def push(self) -> None:
        self._emit({"type": "progress", "progress": self.snapshot()})


# --------------------------------------------------------------------------
# the manager
# --------------------------------------------------------------------------

class ExperimentManager:
    """Owns running experiments, their cancel flags and their event subscribers."""

    def __init__(self) -> None:
        self._threads: Dict[str, threading.Thread] = {}
        self._cancels: Dict[str, threading.Event] = {}
        self._subscribers: Dict[str, List["queue.Queue[Dict[str, Any]]"]] = {}
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # -- subscriptions ------------------------------------------------
    def subscribe(self, exp_id: str) -> "queue.Queue[Dict[str, Any]]":
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=500)
        with self._lock:
            self._subscribers.setdefault(exp_id, []).append(q)
            latest = self._latest.get(exp_id)
        if latest:
            q.put(latest)
        return q

    def unsubscribe(self, exp_id: str, q: "queue.Queue[Dict[str, Any]]") -> None:
        with self._lock:
            subs = self._subscribers.get(exp_id, [])
            if q in subs:
                subs.remove(q)

    def emit(self, exp_id: str, event: Dict[str, Any]) -> None:
        event = {**event, "experiment_id": exp_id, "timestamp": time.time()}
        with self._lock:
            if event.get("type") == "progress":
                self._latest[exp_id] = event
            subs = list(self._subscribers.get(exp_id, []))
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    # -- lifecycle ----------------------------------------------------
    def start(self, params: Dict[str, Any]) -> str:
        settings = get_settings()
        service = OllamaService()

        health = service.check_health()
        if not health["connected"]:
            raise OllamaUnavailable(
                health["error"]["message"] if health.get("error") else OllamaUnavailable.user_message,
                detail=(health.get("error") or {}).get("detail", ""),
            )

        model = params["model"]
        available = service.list_models()
        if not any(m["name"] == model for m in available):
            raise OllamaModelMissing(
                f"Model '{model}' is not in Ollama's model list. Pull it with "
                f"`ollama pull {model}` or pick another model."
            )

        preset = get_prompt(params["prompt_id"]) if params.get("prompt_id") else None
        prompt = (params.get("prompt") or (preset or {}).get("prompt") or "").strip()
        if not prompt:
            raise ValueError("A prompt is required to start an experiment.")

        exp_id = db.create_experiment({
            "model": model,
            "model_meta": next((m for m in available if m["name"] == model), {}),
            "prompt": prompt,
            "prompt_label": (preset or {}).get("label") or params.get("prompt_label"),
            "prompt_category": (preset or {}).get("category"),
            "system_prompt": params.get("system_prompt") or None,
            "strategy": params.get("strategy", settings.default_strategy),
            "objective": params.get("objective", settings.default_objective),
            "weights": params.get("custom_weights") or {},
            "runs_per_config": int(params.get("runs_per_config", settings.default_runs_per_config)),
            "max_tokens": int(params.get("max_tokens", settings.default_max_tokens)),
            "timeout_seconds": float(params.get("timeout_seconds", settings.default_timeout_seconds)),
            "stream": bool(params.get("stream", settings.default_stream)),
            "concurrency": max(1, int(params.get("concurrency", settings.max_concurrency))),
            "evaluator_mode": params.get("evaluator_mode", settings.evaluator_mode),
            "seed": int(params.get("seed", 42)),
            "app_version": APP_VERSION,
            "ollama_version": health.get("version"),
            "host_info": host_info(),
            "status": "queued",
        })

        cancel = threading.Event()
        runner = _ExperimentRunner(self, exp_id, params, preset, cancel)
        thread = threading.Thread(target=runner.run, name=f"exp-{exp_id}", daemon=True)
        with self._lock:
            self._threads[exp_id] = thread
            self._cancels[exp_id] = cancel
        thread.start()
        return exp_id

    def cancel(self, exp_id: str) -> bool:
        with self._lock:
            event = self._cancels.get(exp_id)
            thread = self._threads.get(exp_id)
        if not event or not thread or not thread.is_alive():
            # already finished, cancelled or never started: nothing to cancel
            return False
        event.set()
        self.emit(exp_id, {"type": "cancelling",
                           "message": "Cancelling after the current generation finishes."})
        return True

    def is_running(self, exp_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(exp_id)
        return bool(thread and thread.is_alive())

    def running_ids(self) -> List[str]:
        with self._lock:
            return [eid for eid, t in self._threads.items() if t.is_alive()]


MANAGER = ExperimentManager()


# --------------------------------------------------------------------------
# the worker
# --------------------------------------------------------------------------

class _ExperimentRunner:
    def __init__(self, manager: ExperimentManager, exp_id: str, params: Dict[str, Any],
                 preset: Optional[Dict[str, Any]], cancel: threading.Event):
        self.manager = manager
        self.exp_id = exp_id
        self.params = params
        self.preset = preset
        self.cancel = cancel
        self.settings = get_settings()
        self.service = OllamaService()
        self.progress = Progress(lambda ev: manager.emit(exp_id, ev))
        self.consecutive_fatal = 0
        self.seen_signatures: Dict[str, str] = {}
        self.outcomes: Dict[str, List[ConfigOutcome]] = {}
        self.config_rows: List[Dict[str, Any]] = []
        self.seq = 0
        self.not_tested: List[Dict[str, str]] = []

    # -- helpers ------------------------------------------------------
    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise ExperimentCancelled()

    def emit(self, event: Dict[str, Any]) -> None:
        self.manager.emit(self.exp_id, event)

    def _save_progress(self) -> None:
        db.update_experiment(self.exp_id, progress=self.progress.snapshot())

    # -- main ---------------------------------------------------------
    def run(self) -> None:
        experiment = db.get_experiment(self.exp_id)
        started = time.time()
        db.update_experiment(self.exp_id, status="running", started_at=started)
        self.emit({"type": "started", "experiment_id": self.exp_id})

        try:
            self._run_pipeline(experiment)
        except ExperimentCancelled:
            db.update_experiment(self.exp_id, status="cancelled", finished_at=time.time(),
                                 error="Cancelled by the user.")
            self.emit({"type": "cancelled", "message": "Experiment cancelled."})
            log.info("Experiment %s cancelled", self.exp_id)
        except OllamaError as exc:
            db.update_experiment(self.exp_id, status="failed", finished_at=time.time(),
                                 error=f"{exc} {exc.detail}".strip())
            self.emit({"type": "failed", "message": str(exc), "detail": exc.detail})
            log.error("Experiment %s failed: %s", self.exp_id, exc)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Experiment %s crashed: %s\n%s", self.exp_id, exc, traceback.format_exc())
            db.update_experiment(self.exp_id, status="failed", finished_at=time.time(),
                                 error=f"Unexpected error: {exc}")
            self.emit({"type": "failed", "message": f"Unexpected error: {exc}"})
        finally:
            self._save_progress()

    def _run_pipeline(self, experiment: Dict[str, Any]) -> None:
        settings = self.settings
        model = experiment["model"]
        prompt = experiment["prompt"]
        runs_per_config = experiment["runs_per_config"]

        # ---- step 1: discovery & capabilities -----------------------
        self.progress.add("discover", "Model discovered")
        self.progress.add("plan", "Optimization plan built")
        self.progress.add("baseline", "Baseline test completed")
        self.progress.set("discover", "running")

        available = self.service.list_models()
        capabilities = self.service.capabilities(model)
        expectations = infer_expectations(prompt, (self.preset or {}).get("expectations"))
        self.progress.set("discover", "done",
                          f"{capabilities.get('parameter_size') or 'unknown size'}, "
                          f"{capabilities.get('quantization_level') or 'unknown quantization'}, "
                          f"Ollama {capabilities.get('version')}")

        ctx = OptimizationContext(
            model=model,
            prompt=prompt,
            system_prompt=experiment.get("system_prompt"),
            expectations=expectations,
            capabilities=capabilities,
            model_meta=experiment.get("model_meta", {}),
            available_models=available,
            max_tokens=experiment["max_tokens"],
            seed=experiment.get("seed") or 42,
            host_info=experiment.get("host_info", {}),
            prompt_category=experiment.get("prompt_category"),
        )

        # ---- step 2: plan -------------------------------------------
        self.progress.set("plan", "running")
        optimizations = resolve_strategy(experiment["strategy"],
                                         self.params.get("optimizations"))
        plan: List[tuple[Any, List[RunConfig]]] = []
        applicability: Dict[str, Dict[str, str]] = {}

        for opt in optimizations:
            verdict = opt.applicability(ctx)
            applicability[opt.id] = {"applicable": verdict.applicable,
                                     "reason": verdict.reason, "status": verdict.status}
            if isinstance(opt, RuntimeOptimization):
                self.not_tested.extend(opt.not_tested(ctx))
            if not verdict.applicable:
                if verdict.status == "informational" and opt.id == "quantization":
                    self.not_tested.append({
                        "item": "Quantization comparison",
                        "status": "Informational / not directly tested",
                        "reason": verdict.reason,
                    })
                else:
                    self.not_tested.append({
                        "item": opt.name,
                        "status": "Not tested",
                        "reason": verdict.reason,
                    })
                self.progress.ensure(f"opt_{opt.id}", f"{opt.name} benchmark")
                self.progress.set(f"opt_{opt.id}", "skipped", verdict.reason)
                continue
            configs = opt.generate_configurations(ctx)
            configs = self._dedupe(configs)
            if not configs:
                self.progress.ensure(f"opt_{opt.id}", f"{opt.name} benchmark")
                self.progress.set(f"opt_{opt.id}", "skipped",
                                  "No applicable configurations were generated.")
                continue
            plan.append((opt, configs))
            self.progress.ensure(f"opt_{opt.id}", f"{opt.name} benchmark")

        self.progress.ensure("validation", "Final validation")
        self.progress.ensure("analysis", "Final analysis")
        self.progress.ensure("report", "Report generated")
        self.progress.ensure("pdf", "PDF generated")

        coarse_runs = sum(len(cfgs) for _, cfgs in plan) * runs_per_config
        self.progress.total_runs = runs_per_config + coarse_runs
        self.progress.set("plan", "done",
                          f"{len(plan)} optimization categories, "
                          f"{sum(len(c) for _, c in plan)} configurations before refinement")
        self._save_progress()

        # ---- step 3: warm-up + baseline -----------------------------
        self._warm_up(model, experiment)
        self._check_cancel()

        self.progress.set("baseline", "running")
        baseline_cfg = BASELINE.generate_configurations(ctx)[0]
        baseline_outcome = self._execute_config(baseline_cfg, ctx, experiment, runs_per_config)
        if baseline_outcome.aggregate.get("successful_runs", 0) == 0:
            raise OllamaError(
                "The baseline configuration produced no successful runs, so there is "
                "nothing to compare optimizations against. See the run errors for detail."
            )
        self.progress.set("baseline", "done",
                          _fmt_outcome(baseline_outcome))

        # ---- step 4: coarse search ----------------------------------
        for opt, configs in plan:
            self._check_cancel()
            self.progress.set(f"opt_{opt.id}", "running", f"{len(configs)} configurations")
            results: List[ConfigOutcome] = []
            for cfg in configs:
                self._check_cancel()
                results.append(self._execute_config(cfg, ctx, experiment, runs_per_config))
            self.outcomes[opt.id] = results

            # ---- step 5: fine search for this optimization ----------
            if get_settings().coarse_fine_search and opt.fine_budget > 0:
                refined = self._dedupe(opt.refine(ctx, results)[: opt.fine_budget + 1])
                if refined:
                    self.progress.total_runs += len(refined) * runs_per_config
                    self.progress.set(f"opt_{opt.id}", "running",
                                      f"refining: {len(refined)} follow-up configurations")
                    for cfg in refined:
                        self._check_cancel()
                        results.append(self._execute_config(cfg, ctx, experiment, runs_per_config))
            best = max((r for r in results if r.quality is not None),
                       key=lambda r: r.quality, default=None)
            self.progress.set(f"opt_{opt.id}", "done",
                              (f"best: {best.config.label} ({_fmt_outcome(best)})"
                               if best else "no scorable result"))

        # ---- step 6: validation -------------------------------------
        self.progress.set("validation", "running")
        self._validation_phase(ctx, experiment)
        self.progress.set("validation", "done",
                          f"top {VALIDATION_TOP_N} configurations re-sampled with "
                          f"{VALIDATION_EXTRA_RUNS} extra runs each")

        # ---- step 7: analysis ---------------------------------------
        self.progress.set("analysis", "running")
        summary = self._analyse(experiment)
        self.progress.set("analysis", "done", summary.get("headline", ""))

        # The analysis is stored immediately so the UI can show results while the
        # report renders, but the experiment is only marked "completed" once its
        # report files exist - otherwise a download could 404 straight after the
        # status flips.
        db.update_experiment(self.exp_id, summary=summary)
        self.emit({"type": "analysis", "summary": summary})

        # ---- step 8: report + pdf -----------------------------------
        self._generate_outputs(experiment)

        db.update_experiment(self.exp_id, status="completed", finished_at=time.time())
        self._save_progress()
        self.emit({"type": "completed", "experiment_id": self.exp_id})
        log.info("Experiment %s completed", self.exp_id)

    # -- phases -------------------------------------------------------
    def _warm_up(self, model: str, experiment: Dict[str, Any]) -> None:
        """Load the model so the first measured run does not pay the load cost."""
        self.progress.ensure("warmup", "Model loaded (warm-up)")
        self.progress.set("warmup", "running")
        try:
            result = self.service.generate(
                model, "Reply with the single word: ready.",
                options={"num_predict": 8, "temperature": 0.0},
                stream=False, timeout=min(120.0, experiment["timeout_seconds"]),
                cancel_event=self.cancel,
            )
            detail = "model resident"
            if result.load_duration is not None:
                detail = f"load time {result.load_duration:.2f}s (excluded from results)"
            self.progress.set("warmup", "done", detail)
        except Cancelled:
            raise ExperimentCancelled()
        except OllamaError as exc:
            self.progress.set("warmup", "failed", f"{exc}")
            raise

    def _validation_phase(self, ctx: OptimizationContext, experiment: Dict[str, Any]) -> None:
        """Re-sample the leading configurations to strengthen the comparison."""
        rows = self._collect_config_rows()
        result = recommend(rows, experiment["objective"], experiment.get("weights"))
        ranking = [r for r in result["ranking"] if not r["is_baseline"]][:VALIDATION_TOP_N]
        if not ranking:
            return
        self.progress.total_runs += len(ranking) * VALIDATION_EXTRA_RUNS
        for row in ranking:
            self._check_cancel()
            cfg_row = db.get_configuration(row["configuration_id"])
            if not cfg_row:
                continue
            cfg = self._rehydrate(cfg_row, ctx)
            existing = db.list_runs(self.exp_id, cfg_row["id"])
            start_index = len(existing)
            for offset in range(VALIDATION_EXTRA_RUNS):
                self._check_cancel()
                self._execute_single_run(cfg, cfg_row["id"], ctx, experiment,
                                         start_index + offset)
            self._reaggregate(cfg_row["id"], cfg, ctx, experiment)
            db.update_configuration(cfg_row["id"], phase="validated")

    def _analyse(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        rows = self._collect_config_rows()
        result = recommend(rows, experiment["objective"], experiment.get("weights"))
        winner = result.get("recommended")
        baseline = result.get("baseline_row")
        headline = "No configuration could be scored."
        if winner:
            if winner["is_baseline"]:
                headline = ("Baseline ranked first: no tested optimization beat the model "
                            "defaults under these weights.")
            else:
                comparison = result.get("recommended_comparison", {})
                quality_delta = comparison.get("quality_percent")
                headline = (f"Recommended: {winner['label']}"
                            + (f" (quality {quality_delta:+.1f}% vs baseline)"
                               if quality_delta is not None else ""))
        return {
            "headline": headline,
            "objective": result["objective"],
            "weights": result["weights"],
            "recommended": winner,
            "recommended_comparison": result.get("recommended_comparison"),
            "alternatives": result.get("alternatives", {}),
            "tradeoffs": result.get("tradeoffs", []),
            "explanation": result.get("explanation"),
            "caveat": result.get("caveat"),
            "ranking_ids": [r["configuration_id"] for r in result["ranking"]],
            "configurations_tested": len(rows),
            "not_tested": self.not_tested,
        }

    def _generate_outputs(self, experiment: Dict[str, Any]) -> None:
        from .report import build_report, write_markdown  # local import: avoids cycles
        self.progress.set("report", "running")
        try:
            report = build_report(self.exp_id)
            md_path = write_markdown(report)
            db.create_report(self.exp_id, "markdown", str(md_path),
                             pages=report["meta"]["estimated_pages"])
            self.progress.set("report", "done",
                              f"~{report['meta']['estimated_pages']} pages")
        except Exception as exc:  # report failure must not lose the experiment
            log.error("Report generation failed for %s: %s", self.exp_id, exc)
            db.create_report(self.exp_id, "markdown", None, status="failed", error=str(exc))
            self.progress.set("report", "failed", str(exc))
            report = None

        if not get_settings().auto_generate_pdf:
            self.progress.set("pdf", "skipped",
                              "Automatic PDF generation is switched off in settings.")
            return
        if report is None:
            self.progress.set("pdf", "skipped", "No report data to render.")
            return

        self.progress.set("pdf", "running")
        try:
            from .pdf import generate_pdf
            pdf_path, pages = generate_pdf(report)
            db.create_report(self.exp_id, "pdf", str(pdf_path), pages=pages)
            self.progress.set("pdf", "done", f"{pages} pages")
            self.emit({"type": "report_ready", "pdf": True})
        except Exception as exc:
            log.error("PDF generation failed for %s: %s\n%s", self.exp_id, exc,
                      traceback.format_exc())
            db.create_report(self.exp_id, "pdf", None, status="failed", error=str(exc))
            self.progress.set("pdf", "failed",
                              f"PDF generation failed: {exc}. The Markdown report is "
                              "still available.")

    # -- execution ----------------------------------------------------
    def _dedupe(self, configs: Sequence[RunConfig]) -> List[RunConfig]:
        """Drop configurations identical to one already scheduled."""
        out = []
        for cfg in configs:
            signature = cfg.signature()
            if signature in self.seen_signatures:
                log.debug("Skipping duplicate configuration %s (same as %s)",
                          cfg.key, self.seen_signatures[signature])
                continue
            self.seen_signatures[signature] = cfg.key
            out.append(cfg)
        return out

    def _execute_config(self, cfg: RunConfig, ctx: OptimizationContext,
                        experiment: Dict[str, Any], runs: int) -> ConfigOutcome:
        self.seq += 1
        record = cfg.to_db()
        record["prompt"] = cfg.prompt or ctx.prompt
        cfg_id = db.create_configuration(self.exp_id, self.seq, record)
        db.update_configuration(cfg_id, status="running")
        self.progress.current = cfg.label
        self.emit({"type": "configuration_started", "configuration_id": cfg_id,
                   "label": cfg.label, "category": cfg.category})

        concurrency = max(1, min(int(experiment["concurrency"]), runs))
        if concurrency == 1:
            for index in range(runs):
                self._check_cancel()
                self._execute_single_run(cfg, cfg_id, ctx, experiment, index)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(self._execute_single_run, cfg, cfg_id, ctx,
                                       experiment, index) for index in range(runs)]
                for future in as_completed(futures):
                    exc = future.exception()
                    if isinstance(exc, ExperimentCancelled):
                        self.cancel.set()
            self._check_cancel()

        outcome = self._reaggregate(cfg_id, cfg, ctx, experiment)
        self.emit({"type": "configuration_finished", "configuration_id": cfg_id,
                   "label": cfg.label, "aggregate": outcome.aggregate})
        self._save_progress()
        return outcome

    def _execute_single_run(self, cfg: RunConfig, cfg_id: str, ctx: OptimizationContext,
                            experiment: Dict[str, Any], index: int) -> None:
        self._check_cancel()
        run_id = db.create_run(self.exp_id, cfg_id, index)
        prompt = cfg.prompt or ctx.prompt
        model = cfg.model_override or ctx.model
        stream = cfg.stream if cfg.stream is not None else experiment["stream"]
        expectations = cfg.expectations_override or ctx.expectations

        try:
            with ResourceSampler() as sampler:
                result = self.service.generate(
                    model, prompt,
                    system=cfg.system_prompt or experiment.get("system_prompt"),
                    options=cfg.options,
                    stream=stream,
                    timeout=experiment["timeout_seconds"],
                    keep_alive=cfg.keep_alive,
                    fmt=cfg.fmt,
                    cancel_event=self.cancel,
                )
            resources = sampler.result()
            metrics = result.to_metrics()
            evaluation = evaluate_output(prompt, result.text, expectations, metrics)

            if experiment["evaluator_mode"] == "heuristic+llm":
                judge_model = (get_settings().evaluator_model or model)
                evaluation["llm_judge"] = llm_judge(
                    self.service, judge_model, prompt, result.text,
                    timeout=get_settings().evaluator_timeout)

            db.finish_run(run_id, status="ok", output=result.text, metrics=metrics,
                          resources=resources, evaluation=evaluation)
            self.consecutive_fatal = 0
            self.progress.run_done(f"{cfg.label} - run {index + 1}")
            self.emit({"type": "run_finished", "configuration_id": cfg_id,
                       "run_index": index, "status": "ok",
                       "quality": evaluation.get("quality_score"),
                       "latency": metrics.get("wall_seconds")})

        except Cancelled:
            db.finish_run(run_id, status="cancelled", error="Cancelled by the user.")
            raise ExperimentCancelled()
        except OllamaModelMissing as exc:
            db.finish_run(run_id, status="failed", error=f"{exc} {exc.detail}".strip())
            self.consecutive_fatal += 1
            self.progress.run_done(f"{cfg.label} - run {index + 1} (model missing)")
            if self.consecutive_fatal >= MAX_CONSECUTIVE_FATAL:
                raise OllamaModelMissing(
                    f"The model disappeared during the experiment: {exc}. "
                    "Aborting after repeated failures."
                ) from exc
        except OllamaUnavailable as exc:
            db.finish_run(run_id, status="failed", error=f"{exc} {exc.detail}".strip())
            self.consecutive_fatal += 1
            self.progress.run_done(f"{cfg.label} - run {index + 1} (Ollama unreachable)")
            if self.consecutive_fatal >= MAX_CONSECUTIVE_FATAL:
                raise
        except OllamaError as exc:
            # A single failed run is recorded; the experiment continues.
            db.finish_run(run_id, status="failed", error=f"{exc} {exc.detail}".strip())
            self.progress.run_done(f"{cfg.label} - run {index + 1} (failed)")
            self.emit({"type": "run_finished", "configuration_id": cfg_id,
                       "run_index": index, "status": "failed", "error": str(exc)})

    # -- aggregation --------------------------------------------------
    def _reaggregate(self, cfg_id: str, cfg: RunConfig, ctx: OptimizationContext,
                     experiment: Dict[str, Any]) -> ConfigOutcome:
        runs = db.list_runs(self.exp_id, cfg_id)
        aggregate = aggregate_runs(runs)
        status = ("completed" if aggregate["successful_runs"] else "failed")
        db.update_configuration(cfg_id, status=status, aggregate=aggregate)
        outcome = ConfigOutcome(cfg, aggregate)
        return outcome

    def _collect_config_rows(self) -> List[Dict[str, Any]]:
        rows = db.list_configurations(self.exp_id)
        for row in rows:
            row["prompt_modified"] = row.get("prompt") != db.get_experiment(self.exp_id)["prompt"]
        return rows

    def _rehydrate(self, row: Dict[str, Any], ctx: OptimizationContext) -> RunConfig:
        """Rebuild a RunConfig from its stored row (used by the validation phase)."""
        options = {k: v for k, v in row["options"].items() if not k.startswith("_")}
        return RunConfig(
            key=row["id"],
            label=row["label"],
            optimization_id=row["optimization_id"],
            category=row["category"],
            rationale=row.get("rationale") or "",
            phase="validation",
            options=options,
            prompt=row.get("prompt") or None,
            system_prompt=row.get("system_prompt"),
            model_override=row["options"].get("_model"),
            stream=row["options"].get("_stream"),
            fmt=row["options"].get("_format"),
            keep_alive=row["options"].get("_keep_alive"),
            is_baseline=row.get("is_baseline", False),
        )


# --------------------------------------------------------------------------
# aggregation (module level so tests can call it directly)
# --------------------------------------------------------------------------

def aggregate_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn a configuration's runs into statistics.

    Metrics that Ollama did not report simply produce an empty summary (``n: 0``),
    which the UI and report render as "not available".
    """
    ok = [r for r in runs if r.get("status") == "ok"]
    failed = [r for r in runs if r.get("status") == "failed"]

    def metric(name: str) -> List[float]:
        return [r["metrics"].get(name) for r in ok
                if isinstance(r.get("metrics", {}).get(name), (int, float))]

    quality_scores = [r["evaluation"].get("quality_score") for r in ok
                      if isinstance(r.get("evaluation", {}).get("quality_score"), (int, float))]
    outputs = [r.get("output") or "" for r in ok]

    criteria_summary: Dict[str, Any] = {}
    criteria_names = set()
    for r in ok:
        criteria_names.update((r.get("evaluation", {}).get("criteria") or {}).keys())
    for name in sorted(criteria_names):
        values = []
        for r in ok:
            crit = (r.get("evaluation", {}).get("criteria") or {}).get(name) or {}
            if isinstance(crit.get("score"), (int, float)):
                values.append(crit["score"])
        criteria_summary[name] = summarize(values, "0-10")

    judge_scores = [((r.get("evaluation", {}).get("llm_judge") or {}).get("mean"))
                    for r in ok]
    judge_scores = [s for s in judge_scores if isinstance(s, (int, float))]

    total = len(runs)
    aggregate: Dict[str, Any] = {
        "total_runs": total,
        "successful_runs": len(ok),
        "failed_runs": len(failed),
        "failure_rate": round(len(failed) / total, 4) if total else 0.0,
        "latency": summarize(metric("wall_seconds"), "s"),
        "time_to_first_token": summarize(metric("time_to_first_token"), "s"),
        "total_duration": summarize(metric("total_duration"), "s"),
        "load_duration": summarize(metric("load_duration"), "s"),
        "prompt_eval_duration": summarize(metric("prompt_eval_duration"), "s"),
        "eval_duration": summarize(metric("eval_duration"), "s"),
        "tokens_per_second": summarize(metric("tokens_per_second"), "tok/s"),
        "output_tokens": summarize(metric("output_tokens"), "tokens"),
        "prompt_tokens": summarize(metric("prompt_tokens"), "tokens"),
        "output_chars": summarize(metric("output_chars"), "chars"),
        "quality": summarize(quality_scores, "0-10"),
        "criteria": criteria_summary,
        "consistency": consistency_across_runs(outputs, quality_scores),
        "resources": merge_resource_samples([r.get("resources") or {} for r in ok]),
        "truncated_runs": sum(1 for r in ok if r.get("metrics", {}).get("truncated")),
        "samples": {
            "quality": quality_scores,
            "latency": metric("wall_seconds"),
            "tokens_per_second": metric("tokens_per_second"),
        },
        "errors": [r.get("error") for r in failed if r.get("error")][:5],
        "llm_judge": (summarize(judge_scores, "0-10 (model estimate)")
                      if judge_scores else None),
    }
    return aggregate


def _fmt_outcome(outcome: Optional[ConfigOutcome]) -> str:
    if outcome is None:
        return ""
    agg = outcome.aggregate
    quality = (agg.get("quality") or {}).get("mean")
    latency = (agg.get("latency") or {}).get("median")
    tps = (agg.get("tokens_per_second") or {}).get("mean")
    bits = []
    if quality is not None:
        bits.append(f"quality {quality:.2f}/10")
    if latency is not None:
        bits.append(f"median {latency:.2f}s")
    if tps is not None:
        bits.append(f"{tps:.1f} tok/s")
    return ", ".join(bits)
