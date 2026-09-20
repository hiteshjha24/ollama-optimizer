"""End-to-end pipeline: plan -> baseline -> optimize -> aggregate -> report -> PDF.

Every generation in these tests is served by tests/fake_ollama.py. The numbers
are synthetic by construction and are only used to prove that the pipeline
carries measurements through to the report intact.
"""

from __future__ import annotations

import time
import unittest

from support import SandboxCase
from fake_ollama import FakeOllama

from app import db
from app.engine import MANAGER
from app.ollama import OllamaModelMissing, OllamaUnavailable
from app.report import build_report, render_markdown, write_markdown

MODEL = "fake-model:8b"


def wait_for(exp_id: str, timeout: float = 240.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        exp = db.get_experiment(exp_id)
        if exp and exp["status"] in {"completed", "failed", "cancelled"}:
            return exp
        time.sleep(0.15)
    raise AssertionError(f"experiment {exp_id} did not finish within {timeout}s")


class PipelineCase(SandboxCase):
    fake_config: dict = {}

    def setUp(self):
        super().setUp()
        self.fake = FakeOllama(**self.fake_config).start()
        self.use_ollama(self.fake.url)

    def tearDown(self):
        self.fake.stop()
        super().tearDown()

    def start(self, **overrides) -> str:
        params = {
            "model": MODEL,
            "prompt_id": "reasoning_trains",
            "strategy": "generation",
            "objective": "balanced",
            "runs_per_config": 2,
            "max_tokens": 128,
            "timeout_seconds": 30,
            "concurrency": 1,
            "stream": True,
            "evaluator_mode": "heuristic",
            "seed": 42,
        }
        params.update(overrides)
        return MANAGER.start(params)


class TestFullRun(PipelineCase):
    def test_pipeline_produces_measurements_and_a_recommendation(self):
        exp_id = self.start()
        exp = wait_for(exp_id)
        self.assertEqual(exp["status"], "completed", exp.get("error"))

        configs = db.list_configurations(exp_id)
        self.assertGreater(len(configs), 5)

        baselines = [c for c in configs if c["is_baseline"]]
        self.assertEqual(len(baselines), 1)
        self.assertGreater(baselines[0]["aggregate"]["successful_runs"], 0)

        # every completed configuration carries real per-run measurements
        completed = [c for c in configs if c["status"] == "completed"]
        self.assertTrue(completed)
        for cfg in completed:
            agg = cfg["aggregate"]
            self.assertGreater(agg["latency"]["n"], 0, cfg["label"])
            self.assertIsNotNone(agg["quality"]["mean"], cfg["label"])
            self.assertGreater(agg["tokens_per_second"]["mean"], 0, cfg["label"])

        runs = db.list_runs(exp_id)
        self.assertGreaterEqual(len(runs), len(completed) * 2)
        self.assertTrue(all(r["output"] for r in runs if r["status"] == "ok"))

        summary = exp["summary"]
        self.assertTrue(summary["headline"])
        self.assertIsNotNone(summary["recommended"])
        self.assertIn("weights", summary)

        # the fake server answers better at low temperature, so an optimization
        # should win over the 0.8-temperature baseline
        self.assertFalse(summary["recommended"]["is_baseline"])

    def test_fine_phase_and_validation_run(self):
        exp_id = self.start()
        wait_for(exp_id)
        phases = {c["phase"] for c in db.list_configurations(exp_id)}
        self.assertIn("baseline", phases)
        self.assertIn("coarse", phases)
        self.assertIn("fine", phases)

    def test_report_and_pdf_are_generated(self):
        exp_id = self.start()
        wait_for(exp_id)

        report = build_report(exp_id)
        for section in ("meta", "executive_summary", "methodology", "baseline",
                        "optimization_results", "comparison", "recommended",
                        "not_tested", "limitations", "raw_outputs"):
            self.assertIn(section, report)

        markdown = render_markdown(report)
        self.assertGreater(len(markdown), 4000)
        self.assertIn("Recommended configuration", markdown)
        self.assertIn("Limitations", markdown)
        self.assertIn("Not tested", markdown)
        path = write_markdown(report)
        self.assertTrue(path.exists())

        from app.pdf import generate_pdf
        pdf_path, pages = generate_pdf(report)
        self.assertTrue(pdf_path.exists())
        self.assertGreaterEqual(pages, 4)
        self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF"))

    def test_report_marks_untested_items_explicitly(self):
        exp_id = self.start(strategy="all")
        wait_for(exp_id)
        report = build_report(exp_id)
        self.assertTrue(report["not_tested"])
        for entry in report["not_tested"]:
            self.assertTrue(entry.get("reason"), entry)

    def test_custom_prompt_without_preset(self):
        exp_id = self.start(prompt_id=None, strategy="baseline",
                            prompt="Reply with exactly one short sentence.")
        exp = wait_for(exp_id)
        self.assertEqual(exp["status"], "completed", exp.get("error"))
        runs = db.list_runs(exp_id)
        self.assertTrue(runs)


class TestCancellation(PipelineCase):
    fake_config = {"delay": 0.25}

    def test_cancel_stops_the_experiment(self):
        exp_id = self.start(strategy="generation", runs_per_config=5)
        time.sleep(1.2)
        self.assertTrue(MANAGER.cancel(exp_id))
        exp = wait_for(exp_id, timeout=90)
        self.assertEqual(exp["status"], "cancelled")
        # work already completed is kept
        self.assertTrue(db.list_runs(exp_id))


class TestFailureHandling(PipelineCase):
    # the warm-up generation succeeds, every measured run then fails
    fake_config = {"fail_generate_after": 1}

    def test_failed_runs_are_recorded_with_their_error(self):
        exp_id = self.start(strategy="baseline")
        exp = wait_for(exp_id, timeout=120)
        self.assertIn(exp["status"], {"completed", "failed"})
        runs = db.list_runs(exp_id)
        self.assertTrue(runs)
        self.assertTrue(all(r["status"] == "failed" for r in runs))
        self.assertTrue(all(r["error"] for r in runs))
        self.assertTrue(all(r["output"] in (None, "") for r in runs))


class TestTotalOutage(PipelineCase):
    fake_config = {"fail_generate": True}

    def test_experiment_fails_cleanly_when_nothing_can_be_generated(self):
        exp_id = self.start(strategy="baseline")
        exp = wait_for(exp_id, timeout=120)
        self.assertEqual(exp["status"], "failed")
        self.assertTrue(exp["error"])


class TestPreflight(SandboxCase):
    def test_start_fails_clearly_when_ollama_is_down(self):
        with self.assertRaises(OllamaUnavailable):
            MANAGER.start({"model": MODEL, "prompt": "hi"})

    def test_start_rejects_an_unknown_model(self):
        fake = FakeOllama().start()
        try:
            self.use_ollama(fake.url)
            with self.assertRaises(OllamaModelMissing):
                MANAGER.start({"model": "not-installed:70b", "prompt": "hi"})
        finally:
            fake.stop()

    def test_start_rejects_an_empty_prompt(self):
        fake = FakeOllama().start()
        try:
            self.use_ollama(fake.url)
            with self.assertRaises(ValueError):
                MANAGER.start({"model": MODEL, "prompt": "   "})
        finally:
            fake.stop()


if __name__ == "__main__":
    unittest.main()
