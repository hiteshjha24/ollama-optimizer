"""Statistics, aggregation and host/resource probing."""

from __future__ import annotations

import unittest

from support import SandboxCase  # noqa: F401

from app.engine import aggregate_runs
from app.metrics import host_info, percent_change, summarize, welch_t


def run(status="ok", latency=1.0, tps=20.0, quality=8.0, output="answer text"):
    return {
        "status": status,
        "output": output,
        "metrics": {"wall_seconds": latency, "tokens_per_second": tps,
                    "output_tokens": 40, "prompt_tokens": 12,
                    "total_duration": latency, "eval_duration": latency * 0.9,
                    "time_to_first_token": latency * 0.2},
        "evaluation": {"quality_score": quality,
                       "criteria": {"correctness": {"score": quality, "scorable": True}}},
        "resources": {},
        "error": None,
    }


class TestSummarize(unittest.TestCase):
    def test_basic_statistics(self):
        s = summarize([1.0, 2.0, 3.0, 4.0], "s")
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["mean"], 2.5)
        self.assertEqual(s["median"], 2.5)
        self.assertEqual(s["min"], 1.0)
        self.assertEqual(s["max"], 4.0)
        self.assertGreater(s["stdev"], 0)
        self.assertEqual(s["unit"], "s")

    def test_empty_sample_is_marked_unavailable(self):
        s = summarize([], "s")
        self.assertEqual(s["n"], 0)
        self.assertIsNone(s["mean"])

    def test_confidence_interval_requires_three_samples(self):
        self.assertIsNone(summarize([1.0, 2.0], "s").get("ci95"))
        self.assertIsNotNone(summarize([1.0, 2.0, 3.0], "s").get("ci95"))

    def test_single_sample_has_no_stdev(self):
        s = summarize([2.0], "s")
        self.assertEqual(s["n"], 1)
        self.assertEqual(s["mean"], 2.0)
        self.assertIsNone(s.get("ci95"))

    def test_percent_change(self):
        self.assertAlmostEqual(percent_change(110, 100), 10.0)
        self.assertIsNone(percent_change(10, 0))

    def test_welch_t_detects_a_clear_difference(self):
        result = welch_t([10, 10.1, 9.9, 10.05], [20, 20.1, 19.9, 20.05])
        self.assertTrue(result["significant_95"])
        result2 = welch_t([10, 10.1, 9.9], [10.02, 10.05, 9.95])
        self.assertFalse(result2["significant_95"])

    def test_welch_t_needs_samples(self):
        result = welch_t([1.0], [2.0])
        self.assertIsNone(result["significant_95"])
        self.assertIn("note", result)


class TestAggregation(unittest.TestCase):
    def test_aggregate_computes_all_blocks(self):
        runs = [run(latency=1.0), run(latency=1.2), run(latency=1.1)]
        agg = aggregate_runs(runs)
        self.assertEqual(agg["successful_runs"], 3)
        self.assertEqual(agg["failed_runs"], 0)
        self.assertEqual(agg["failure_rate"], 0.0)
        self.assertAlmostEqual(agg["latency"]["median"], 1.1)
        self.assertEqual(agg["quality"]["n"], 3)
        self.assertIn("correctness", agg["criteria"])
        self.assertIsNotNone(agg["consistency"]["score"])
        self.assertEqual(len(agg["samples"]["latency"]), 3)

    def test_failed_runs_are_counted_but_excluded_from_statistics(self):
        runs = [run(), run(status="failed"), run()]
        runs[1]["error"] = "timeout"
        agg = aggregate_runs(runs)
        self.assertEqual(agg["total_runs"], 3)
        self.assertEqual(agg["successful_runs"], 2)
        self.assertAlmostEqual(agg["failure_rate"], 1 / 3, places=3)
        self.assertEqual(agg["latency"]["n"], 2)
        self.assertIn("timeout", agg["errors"])

    def test_missing_metric_is_not_invented(self):
        r = run()
        del r["metrics"]["tokens_per_second"]
        agg = aggregate_runs([r])
        self.assertEqual(agg["tokens_per_second"]["n"], 0)
        self.assertIsNone(agg["tokens_per_second"]["mean"])

    def test_no_runs(self):
        agg = aggregate_runs([])
        self.assertEqual(agg["successful_runs"], 0)
        self.assertEqual(agg["failure_rate"], 0.0)


class TestHostInfo(unittest.TestCase):
    def test_host_info_reports_what_it_can(self):
        info = host_info()
        self.assertIn("platform", info)
        self.assertIn("python", info)
        # GPU info may legitimately be absent; it must never be fabricated.
        if "gpu" in info:
            self.assertTrue(info["gpu"] is None or isinstance(info["gpu"], (list, dict, str)))


if __name__ == "__main__":
    unittest.main()
