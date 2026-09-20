"""Deterministic quality evaluation."""

from __future__ import annotations

import unittest

from support import SandboxCase  # noqa: F401

from app.evaluation import (EVALUATOR_ID, consistency_across_runs, evaluate_output,
                            infer_expectations)
from app.prompts import get_prompt


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.preset = get_prompt("reasoning_trains")
        self.expectations = infer_expectations(self.preset["prompt"],
                                               self.preset["expectations"])

    def test_evaluation_is_deterministic(self):
        text = "Step 1: combine speeds.\nANSWER: 12:24"
        a = evaluate_output(self.preset["prompt"], text, self.expectations, {})
        b = evaluate_output(self.preset["prompt"], text, self.expectations, {})
        self.assertEqual(a["quality_score"], b["quality_score"])
        self.assertEqual(a["evaluator"], EVALUATOR_ID)

    def test_correct_answer_outscores_a_vague_one(self):
        good = evaluate_output(
            self.preset["prompt"],
            "Step 1: 100 km/h combined.\nStep 2: 2.4 hours.\nANSWER: 12:24",
            self.expectations, {})
        bad = evaluate_output(
            self.preset["prompt"],
            "They probably meet around lunchtime somewhere in the middle, I think.",
            self.expectations, {})
        self.assertGreater(good["quality_score"], bad["quality_score"])
        self.assertGreater(good["criteria"]["correctness"]["score"],
                           bad["criteria"]["correctness"]["score"])

    def test_scores_are_bounded_and_explained(self):
        result = evaluate_output(self.preset["prompt"], "ANSWER: 12:24",
                                 self.expectations, {})
        for name, crit in result["criteria"].items():
            if crit["scorable"]:
                self.assertGreaterEqual(crit["score"], 0, name)
                self.assertLessEqual(crit["score"], 10, name)
                self.assertTrue(crit["basis"], name)

    def test_unscorable_criteria_are_marked_not_scored(self):
        """A free-text prompt with no checkable ground truth must not invent one."""
        expectations = infer_expectations("Write a haiku about the sea.", None)
        result = evaluate_output("Write a haiku about the sea.",
                                 "Waves fold over stone", expectations, {})
        self.assertIn("correctness", result["criteria"])
        self.assertFalse(result["criteria"]["correctness"]["scorable"])

    def test_empty_output_scores_zero(self):
        result = evaluate_output(self.preset["prompt"], "", self.expectations, {})
        self.assertEqual(result["quality_score"], 0.0)

    def test_json_expectations_are_inferred(self):
        expectations = infer_expectations(
            "Return only a JSON object with keys id and total. No prose.", None)
        self.assertEqual(expectations.get("format"), "json")
        good = evaluate_output("Return only JSON.", '{"id": 1, "total": 2}',
                               expectations, {})
        bad = evaluate_output("Return only JSON.", "Sure! Here you go: id=1",
                              expectations, {})
        self.assertGreater(good["criteria"]["format_compliance"]["score"],
                           bad["criteria"]["format_compliance"]["score"])

    def test_word_limit_is_detected_and_enforced(self):
        expectations = infer_expectations("Summarise this in at most 10 words.", None)
        # the evaluator applies a small tolerance above the stated limit
        self.assertGreaterEqual(expectations.get("max_words"), 10)
        self.assertLessEqual(expectations.get("max_words"), 15)
        short = evaluate_output("Summarise this in at most 10 words.",
                                "A short compliant summary of the text.", expectations, {})
        long = evaluate_output("Summarise this in at most 10 words.",
                               " ".join(["word"] * 80), expectations, {})
        self.assertGreater(short["criteria"]["instruction_following"]["score"],
                           long["criteria"]["instruction_following"]["score"])

    def test_forbidden_word_detected(self):
        preset = get_prompt("creative_lighthouse")
        expectations = infer_expectations(preset["prompt"], preset["expectations"])
        clean = evaluate_output(preset["prompt"],
                                "The keeper watched the grey swell rise and fall all night, "
                                "counting the beats of the lamp above him.", expectations, {})
        dirty = evaluate_output(preset["prompt"],
                                "The storm rolled in and the storm did not stop.",
                                expectations, {})
        self.assertGreater(clean["quality_score"], dirty["quality_score"])

    def test_truncation_lowers_completeness(self):
        expectations = infer_expectations("Explain gravity in a paragraph.", None)
        full = evaluate_output("Explain gravity in a paragraph.",
                               "Gravity is the mutual attraction between masses. " * 4,
                               expectations, {"truncated": False})
        cut = evaluate_output("Explain gravity in a paragraph.",
                              "Gravity is the mutual attraction between masses. " * 4,
                              expectations, {"truncated": True})
        self.assertGreater(full["criteria"]["completeness"]["score"],
                           cut["criteria"]["completeness"]["score"])


class TestConsistency(unittest.TestCase):
    def test_identical_outputs_are_perfectly_consistent(self):
        result = consistency_across_runs(["same text here"] * 4, [8.0, 8.0, 8.0, 8.0])
        self.assertAlmostEqual(result["score"], 10.0, places=4)

    def test_divergent_outputs_score_lower(self):
        same = consistency_across_runs(["alpha beta gamma"] * 3, [7, 7, 7])
        different = consistency_across_runs(
            ["alpha beta gamma", "completely unrelated words", "third distinct answer"],
            [3, 7, 9])
        self.assertGreater(same["score"], different["score"])

    def test_single_run_cannot_be_scored(self):
        result = consistency_across_runs(["only one"], [7.0])
        self.assertIsNone(result["score"])
        self.assertIn("basis", result)


if __name__ == "__main__":
    unittest.main()
