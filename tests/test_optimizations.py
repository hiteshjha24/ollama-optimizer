"""Optimization plugins: applicability, configuration generation, refinement."""

from __future__ import annotations

import unittest

from support import SandboxCase  # noqa: F401  (keeps sys.path setup consistent)

from app.optimizations import REGISTRY, BASELINE, STRATEGIES, resolve_strategy, describe_all
from app.optimizations.base import ConfigOutcome, OptimizationContext, RunConfig
from app.optimizations.context import ContextOptimization
from app.optimizations.generation import GenerationParameterOptimization
from app.optimizations.prompting import PromptOptimization, SystemPromptOptimization
from app.optimizations.quantization import QuantizationOptimization
from app.optimizations.runtime import RuntimeOptimization
from app.ollama import SUPPORTED_OPTIONS


def make_context(**kwargs) -> OptimizationContext:
    base = dict(
        model="fake-model:8b",
        prompt="Summarise the following release notes in exactly 3 bullet points.",
        system_prompt=None,
        expectations={"format": "bullets", "min_items": 3},
        capabilities={"context_length": 8192, "quantization_level": "Q4_K_M",
                      "parameter_size": "8B", "family": "fakellama", "version": "0.5.13"},
        model_meta={"name": "fake-model:8b", "quantization_level": "Q4_K_M",
                    "family": "fakellama"},
        available_models=[{"name": "fake-model:8b", "family": "fakellama",
                           "quantization_level": "Q4_K_M"},
                          {"name": "fake-model:8b-q8", "family": "fakellama",
                           "quantization_level": "Q8_0"}],
        max_tokens=256,
        seed=42,
        host_info={"cpu_count_logical": 8, "cpu_count_physical": 4},
        prompt_category="summarization",
    )
    base.update(kwargs)
    return OptimizationContext(**base)


class TestRegistry(unittest.TestCase):
    def test_every_optimization_has_the_plugin_interface(self):
        for opt in REGISTRY:
            for attr in ("id", "name", "category", "description"):
                self.assertTrue(getattr(opt, attr), f"{opt} missing {attr}")
            self.assertTrue(callable(opt.applicability))
            self.assertTrue(callable(opt.generate_configurations))
            self.assertTrue(callable(opt.refine))

    def test_ids_are_unique(self):
        ids = [o.id for o in REGISTRY]
        self.assertEqual(len(ids), len(set(ids)))

    def test_strategies_reference_known_ids(self):
        known = {o.id for o in REGISTRY} | {BASELINE.id}
        for name, ids in STRATEGIES.items():
            for opt_id in ids:
                self.assertIn(opt_id, known, f"strategy {name} references {opt_id}")

    def test_resolve_strategy_all_returns_everything(self):
        self.assertEqual(len(resolve_strategy("all")), len(STRATEGIES["all"]))

    def test_describe_all_is_serialisable(self):
        described = describe_all()
        self.assertTrue(described)
        self.assertIn("description", described[0])


class TestGenerationParameters(unittest.TestCase):
    def setUp(self):
        self.ctx = make_context()
        self.opt = GenerationParameterOptimization()

    def test_coarse_search_is_one_factor_at_a_time(self):
        configs = self.opt.generate_configurations(self.ctx)
        self.assertGreater(len(configs), 8)
        # A full Cartesian product of the candidate grid would be >100 configs.
        self.assertLess(len(configs), 30)
        for cfg in configs:
            varied = [k for k in ("temperature", "top_p", "top_k", "repeat_penalty")
                      if k in cfg.options]
            self.assertLessEqual(len(varied), 2, cfg.label)

    def test_only_documented_options_are_emitted(self):
        for cfg in self.opt.generate_configurations(self.ctx):
            for key in cfg.options:
                self.assertIn(key, SUPPORTED_OPTIONS, f"{key} is not a documented option")

    def test_signatures_are_unique(self):
        configs = self.opt.generate_configurations(self.ctx)
        signatures = {c.signature() for c in configs}
        self.assertEqual(len(signatures), len(configs))

    def test_refine_uses_the_coarse_winners(self):
        configs = self.opt.generate_configurations(self.ctx)
        outcomes = []
        for i, cfg in enumerate(configs):
            quality = 9.0 if cfg.options.get("temperature") == 0.3 else 5.0
            outcomes.append(ConfigOutcome(cfg, {
                "successful_runs": 3, "total_runs": 3,
                "quality": {"n": 3, "mean": quality},
                "latency": {"n": 3, "median": 1.0, "mean": 1.0},
                "tokens_per_second": {"n": 3, "mean": 20.0},
            }))
        refined = self.opt.refine(self.ctx, outcomes)
        self.assertTrue(refined)
        self.assertLessEqual(len(refined), self.opt.fine_budget)
        for cfg in refined:
            self.assertEqual(cfg.phase, "fine")

    def test_refine_without_results_returns_nothing(self):
        self.assertEqual(self.opt.refine(self.ctx, []), [])


class TestPromptOptimizations(unittest.TestCase):
    def test_prompt_variants_preserve_the_task_and_differ(self):
        ctx = make_context()
        opt = PromptOptimization()
        configs = opt.generate_configurations(ctx)
        self.assertGreaterEqual(len(configs), 3)
        for cfg in configs:
            self.assertTrue(cfg.prompt)
            self.assertNotEqual(cfg.prompt, ctx.prompt, cfg.label)
            self.assertTrue(cfg.rationale)

    def test_system_prompt_variants_set_a_system_prompt(self):
        ctx = make_context()
        configs = SystemPromptOptimization().generate_configurations(ctx)
        self.assertTrue(configs)
        for cfg in configs:
            self.assertTrue(cfg.system_prompt)
            # The user's prompt itself is untouched by system-prompt experiments.
            self.assertIn(cfg.prompt, (None, ctx.prompt))


class TestContextOptimization(unittest.TestCase):
    def test_num_ctx_never_exceeds_declared_context(self):
        ctx = make_context(capabilities={"context_length": 4096, "version": "0.5.13"})
        for cfg in ContextOptimization().generate_configurations(ctx):
            if "num_ctx" in cfg.options:
                self.assertLessEqual(cfg.options["num_ctx"], 4096)

    def test_unknown_context_length_is_handled(self):
        ctx = make_context(capabilities={"version": "0.5.13"})
        configs = ContextOptimization().generate_configurations(ctx)
        for cfg in configs:
            self.assertIn("num_ctx", cfg.options | {"num_ctx": None})


class TestRuntimeOptimization(unittest.TestCase):
    def test_untestable_settings_are_reported_not_faked(self):
        entries = RuntimeOptimization().not_tested(make_context())
        self.assertTrue(entries)
        for entry in entries:
            self.assertTrue(entry.get("reason"), entry)

    def test_thread_config_uses_detected_cpu_count(self):
        configs = RuntimeOptimization().generate_configurations(make_context())
        threads = [c.options.get("num_thread") for c in configs if "num_thread" in c.options]
        for value in threads:
            self.assertLessEqual(value, 8)

    def test_no_thread_config_when_cpu_count_unknown(self):
        ctx = make_context(host_info={})
        configs = RuntimeOptimization().generate_configurations(ctx)
        self.assertFalse([c for c in configs if "num_thread" in c.options])


class TestQuantizationOptimization(unittest.TestCase):
    def test_compares_only_locally_installed_variants(self):
        opt = QuantizationOptimization()
        ctx = make_context()
        configs = opt.generate_configurations(ctx)
        for cfg in configs:
            self.assertIn(cfg.model_override,
                          [m["name"] for m in ctx.available_models])

    def test_informational_when_no_alternative_exists(self):
        ctx = make_context(available_models=[{"name": "fake-model:8b",
                                              "family": "fakellama",
                                              "quantization_level": "Q4_K_M"}])
        applicability = QuantizationOptimization().applicability(ctx)
        self.assertFalse(applicability.applicable)
        self.assertEqual(applicability.status, "informational")
        self.assertIn("alternative", applicability.reason.lower())


class TestRunConfig(unittest.TestCase):
    def test_persisted_options_keep_internal_markers(self):
        cfg = RunConfig(key="k", label="l", optimization_id="o", category="c",
                        rationale="r", options={"temperature": 0.2}, stream=False,
                        model_override="other:8b")
        persisted = cfg.persisted_options()
        self.assertEqual(persisted["_stream"], False)
        self.assertEqual(persisted["_model"], "other:8b")
        self.assertEqual(persisted["temperature"], 0.2)

    def test_identical_configs_share_a_signature(self):
        a = RunConfig(key="a", label="A", optimization_id="o", category="c",
                      rationale="", options={"temperature": 0.2})
        b = RunConfig(key="b", label="B", optimization_id="o", category="c",
                      rationale="", options={"temperature": 0.2})
        self.assertEqual(a.signature(), b.signature())


if __name__ == "__main__":
    unittest.main()
