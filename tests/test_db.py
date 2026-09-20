"""SQLite persistence: schema, CRUD, JSON round-trips, cascade delete."""

from __future__ import annotations

import unittest

from support import SandboxCase

from app import db


class TestPersistence(SandboxCase):
    def make_experiment(self, **overrides):
        data = {
            "model": "fake-model:8b",
            "model_meta": {"parameter_size": "8B"},
            "prompt": "Explain gravity.",
            "strategy": "all",
            "objective": "balanced",
            "runs_per_config": 3,
            "max_tokens": 256,
            "timeout_seconds": 60.0,
            "stream": True,
            "concurrency": 1,
            "evaluator_mode": "heuristic",
            "seed": 42,
            "host_info": {"platform": "test"},
        }
        data.update(overrides)
        return db.create_experiment(data)

    def test_schema_is_created_on_first_use(self):
        tables = {r[0] for r in db.connect().execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        for expected in ("experiments", "configurations", "runs", "reports", "meta"):
            self.assertIn(expected, tables)

    def test_experiment_round_trip_preserves_json_fields(self):
        exp_id = self.make_experiment()
        exp = db.get_experiment(exp_id)
        self.assertEqual(exp["model"], "fake-model:8b")
        self.assertEqual(exp["model_meta"]["parameter_size"], "8B")
        self.assertEqual(exp["host_info"]["platform"], "test")
        self.assertIs(exp["stream"], True)
        self.assertEqual(exp["status"], "queued")

    def test_update_experiment(self):
        exp_id = self.make_experiment()
        db.update_experiment(exp_id, status="completed",
                             summary={"headline": "done"}, error=None)
        exp = db.get_experiment(exp_id)
        self.assertEqual(exp["status"], "completed")
        self.assertEqual(exp["summary"]["headline"], "done")

    def test_missing_experiment_returns_none(self):
        self.assertIsNone(db.get_experiment("exp_does_not_exist"))

    def test_configurations_and_runs(self):
        exp_id = self.make_experiment()
        cfg_id = db.create_configuration(exp_id, 1, {
            "optimization_id": "generation", "category": "generation",
            "label": "Temperature 0.3", "phase": "coarse",
            "options": {"temperature": 0.3}, "prompt": "Explain gravity.",
            "rationale": "lower randomness",
        })
        run_id = db.create_run(exp_id, cfg_id, 0)
        db.finish_run(run_id, status="ok", output="text",
                      metrics={"wall_seconds": 1.5},
                      evaluation={"quality_score": 8.0}, resources={})
        runs = db.list_runs(exp_id, cfg_id)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "ok")
        self.assertEqual(runs[0]["metrics"]["wall_seconds"], 1.5)
        self.assertEqual(runs[0]["evaluation"]["quality_score"], 8.0)

        db.update_configuration(cfg_id, status="completed",
                                aggregate={"successful_runs": 1})
        cfg = db.get_configuration(cfg_id)
        self.assertEqual(cfg["status"], "completed")
        self.assertEqual(cfg["aggregate"]["successful_runs"], 1)
        self.assertEqual(cfg["options"]["temperature"], 0.3)

    def test_configurations_are_ordered_by_sequence(self):
        exp_id = self.make_experiment()
        for seq, label in ((2, "second"), (1, "first"), (3, "third")):
            db.create_configuration(exp_id, seq, {
                "optimization_id": "generation", "category": "generation",
                "label": label, "prompt": "p"})
        labels = [c["label"] for c in db.list_configurations(exp_id)]
        self.assertEqual(labels, ["first", "second", "third"])

    def test_failed_run_records_its_error(self):
        exp_id = self.make_experiment()
        cfg_id = db.create_configuration(exp_id, 1, {
            "optimization_id": "generation", "category": "generation",
            "label": "x", "prompt": "p"})
        run_id = db.create_run(exp_id, cfg_id, 0)
        db.finish_run(run_id, status="failed", error="timed out after 60s")
        run = db.list_runs(exp_id, cfg_id)[0]
        self.assertEqual(run["status"], "failed")
        self.assertIn("timed out", run["error"])

    def test_reports_latest_wins(self):
        exp_id = self.make_experiment()
        db.create_report(exp_id, "pdf", "/tmp/a.pdf", pages=4)
        db.create_report(exp_id, "pdf", "/tmp/b.pdf", pages=6)
        latest = db.latest_report(exp_id, "pdf")
        self.assertEqual(latest["path"], "/tmp/b.pdf")
        self.assertEqual(latest["pages"], 6)

    def test_failed_report_is_recorded(self):
        exp_id = self.make_experiment()
        db.create_report(exp_id, "pdf", None, status="failed", error="reportlab missing")
        row = db.list_reports(exp_id)[0]
        self.assertEqual(row["status"], "failed")
        self.assertIn("reportlab", row["error"])

    def test_delete_cascades_to_children(self):
        exp_id = self.make_experiment()
        cfg_id = db.create_configuration(exp_id, 1, {
            "optimization_id": "generation", "category": "generation",
            "label": "x", "prompt": "p"})
        db.create_run(exp_id, cfg_id, 0)
        db.create_report(exp_id, "markdown", "/tmp/x.md")
        self.assertTrue(db.delete_experiment(exp_id))
        self.assertIsNone(db.get_experiment(exp_id))
        self.assertEqual(db.list_configurations(exp_id), [])
        self.assertEqual(db.list_runs(exp_id), [])
        self.assertEqual(db.list_reports(exp_id), [])

    def test_delete_missing_experiment_is_false(self):
        self.assertFalse(db.delete_experiment("exp_nope"))

    def test_counts_and_listing(self):
        for _ in range(3):
            self.make_experiment()
        counts = db.counts()
        self.assertEqual(counts["experiments"], 3)
        self.assertEqual(len(db.list_experiments(limit=2)), 2)

    def test_ids_are_prefixed_and_unique(self):
        ids = {db.new_id("run") for _ in range(50)}
        self.assertEqual(len(ids), 50)
        self.assertTrue(all(i.startswith("run_") for i in ids))


if __name__ == "__main__":
    unittest.main()
