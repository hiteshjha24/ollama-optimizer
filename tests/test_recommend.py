"""Recommendation engine: normalisation, weighting, transparency."""

from __future__ import annotations

import unittest

from support import SandboxCase  # noqa: F401

from app.recommend import (OBJECTIVES, build_scoreboard, compare_to_baseline,
                           recommend, resolve_weights, score_configurations)


def cfg(cid, label, quality, tps, latency, consistency=8.0, baseline=False, runs=5):
    return {
        "id": cid,
        "label": label,
        "is_baseline": baseline,
        "category": "generation",
        "optimization_id": "generation",
        "options": {},
        "aggregate": {
            "successful_runs": runs, "total_runs": runs, "failure_rate": 0.0,
            "quality": {"n": runs, "mean": quality, "median": quality,
                        "samples": [quality] * runs},
            "tokens_per_second": {"n": runs, "mean": tps},
            "latency": {"n": runs, "mean": latency, "median": latency},
            "output_tokens": {"n": runs, "mean": 120},
            "consistency": {"score": consistency},
            "samples": {"quality": [quality] * runs, "latency": [latency] * runs},
        },
    }


BASE = cfg("c0", "Baseline", 7.0, 28.0, 4.2, 7.4, baseline=True)
QUALITY = cfg("c1", "Temperature 0.3", 8.6, 27.0, 4.3, 8.1)
SPEED = cfg("c2", "Non-streaming", 7.9, 38.0, 3.1, 7.9)
CONFIGS = [BASE, QUALITY, SPEED]


class TestWeights(unittest.TestCase):
    def test_builtin_objectives_sum_to_one(self):
        for name, weights in OBJECTIVES.items():
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=6, msg=name)

    def test_custom_weights_are_normalised(self):
        weights = resolve_weights("custom", {"quality": 50, "speed": 30, "consistency": 20})
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertGreater(weights["quality"], weights["speed"])

    def test_unknown_objective_falls_back_to_balanced(self):
        self.assertEqual(resolve_weights("nonsense"), OBJECTIVES["balanced"])


class TestScoreboard(unittest.TestCase):
    def test_normalisation_spans_zero_to_ten(self):
        board = build_scoreboard(CONFIGS)
        quality = board["components"]["quality"]["scores"]
        self.assertAlmostEqual(max(quality.values()), 10.0)
        self.assertAlmostEqual(min(quality.values()), 0.0)

    def test_configurations_without_successful_runs_are_excluded(self):
        broken = cfg("c3", "Broken", 0, 0, 0, runs=0)
        broken["aggregate"]["successful_runs"] = 0
        board = build_scoreboard(CONFIGS + [broken])
        self.assertNotIn("c3", board["config_ids"])

    def test_every_component_documents_its_source(self):
        for name, comp in build_scoreboard(CONFIGS)["components"].items():
            self.assertTrue(comp["source"], name)
            self.assertTrue(comp["direction"], name)


class TestRanking(unittest.TestCase):
    def test_quality_first_prefers_the_quality_winner(self):
        ranked = score_configurations(CONFIGS, resolve_weights("quality_first"))
        self.assertEqual(ranked[0]["configuration_id"], "c1")

    def test_speed_first_prefers_the_speed_winner(self):
        ranked = score_configurations(CONFIGS, resolve_weights("speed_first"))
        self.assertEqual(ranked[0]["configuration_id"], "c2")

    def test_breakdown_shows_the_arithmetic(self):
        ranked = score_configurations(CONFIGS, resolve_weights("balanced"))
        row = ranked[0]
        self.assertIn("breakdown", row)
        parts = [p for p in row["breakdown"] if p["contribution"] is not None]
        self.assertTrue(parts)
        total = sum(p["contribution"] for p in parts) / row["used_weight"]
        self.assertAlmostEqual(total, row["composite_before_penalty"], places=2)
        for part in row["breakdown"]:
            self.assertIn("weight", part)

    def test_failures_are_penalised(self):
        flaky = cfg("c4", "Flaky", 9.9, 40.0, 2.0, 9.0)
        flaky["aggregate"]["failure_rate"] = 0.6
        flaky["aggregate"]["successful_runs"] = 2
        flaky["aggregate"]["total_runs"] = 5
        ranked = score_configurations(CONFIGS + [flaky], resolve_weights("balanced"))
        row = next(r for r in ranked if r["configuration_id"] == "c4")
        self.assertLess(row["composite_score"], row["composite_before_penalty"])
        self.assertAlmostEqual(row["failure_penalty"], 0.4, places=3)


class TestRecommendation(unittest.TestCase):
    def test_recommendation_is_explained_and_conditional(self):
        result = recommend(CONFIGS, "balanced")
        self.assertIsNotNone(result["recommended"])
        self.assertIn("weights", result)
        self.assertTrue(result["explanation"])
        self.assertIn("this benchmark", result["explanation"].lower() + (result.get("caveat") or "").lower())

    def test_alternatives_cover_quality_and_speed(self):
        result = recommend(CONFIGS, "balanced")
        self.assertTrue(set(result["alternatives"]) <= {"maximum_quality", "maximum_speed",
                                                        "most_consistent"})

    def test_comparison_to_baseline_is_percentage_based(self):
        result = recommend(CONFIGS, "quality_first")
        comparison = result["recommended_comparison"]
        self.assertAlmostEqual(comparison["quality_percent"],
                               (8.6 - 7.0) / 7.0 * 100, places=1)

    def test_no_usable_configurations(self):
        empty = cfg("x", "Nothing", 0, 0, 0, runs=0)
        empty["aggregate"]["successful_runs"] = 0
        result = recommend([empty], "balanced")
        self.assertIsNone(result["recommended"])
        self.assertIn("reason", result)

    def test_baseline_can_win(self):
        strong_baseline = cfg("b", "Baseline", 9.5, 40.0, 2.0, 9.5, baseline=True)
        weak = cfg("w", "Temperature 1.0", 5.0, 20.0, 6.0, 5.0)
        result = recommend([strong_baseline, weak], "balanced")
        self.assertTrue(result["recommended"]["is_baseline"])

    def test_compare_to_baseline_without_baseline(self):
        row = score_configurations(CONFIGS, resolve_weights("balanced"))[0]
        result = compare_to_baseline(row, None)
        self.assertFalse(result["available"])
        self.assertTrue(result["reason"])


if __name__ == "__main__":
    unittest.main()
