"""HTTP layer: routing, JSON contracts, static files, SSE, error handling."""

from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from support import SandboxCase
from fake_ollama import FakeOllama

from app import db
from app.server import create_server


def request(url, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            parsed = json.loads(body) if "json" in ctype else body
            return resp.status, parsed
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


class ApiCase(SandboxCase):
    def setUp(self):
        super().setUp()
        self.fake = FakeOllama().start()
        self.use_ollama(self.fake.url)
        self.httpd = create_server(host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.httpd.server_address[0], self.httpd.server_address[1]
        self.base = f"http://{host}:{port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.fake.stop()
        super().tearDown()


class TestReadEndpoints(ApiCase):
    def test_health(self):
        status, body = request(self.base + "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ollama"]["connected"])
        self.assertIn("version", body["app"])

    def test_models_listing(self):
        status, body = request(self.base + "/api/models")
        self.assertEqual(status, 200)
        self.assertTrue(body["connected"])
        self.assertEqual(len(body["models"]), 2)

    def test_model_info(self):
        status, body = request(self.base + "/api/models/fake-model%3A8b/info")
        self.assertEqual(status, 200)
        self.assertEqual(body["quantization_level"], "Q4_K_M")

    def test_prompts_and_optimizations(self):
        _, prompts = request(self.base + "/api/prompts")
        self.assertGreaterEqual(len(prompts["prompts"]), 8)
        _, opts = request(self.base + "/api/optimizations")
        self.assertTrue(opts["optimizations"])
        self.assertTrue(opts["strategies"])
        self.assertTrue(opts["objectives"])

    def test_dashboard(self):
        status, body = request(self.base + "/api/dashboard")
        self.assertEqual(status, 200)
        self.assertEqual(body["models_available"], 2)
        self.assertIn("counts", body)

    def test_static_index_is_served(self):
        status, body = request(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Ollama Optimizer", body)

    def test_unknown_api_route_is_json_404(self):
        status, body = request(self.base + "/api/nope")
        self.assertEqual(status, 404)
        self.assertIn("message", body["error"])

    def test_unknown_page_falls_back_to_the_spa(self):
        status, body = request(self.base + "/experiments/anything")
        self.assertEqual(status, 200)
        self.assertIn(b"<html", body.lower())


class TestSettings(ApiCase):
    def test_get_and_update(self):
        status, body = request(self.base + "/api/settings")
        self.assertEqual(status, 200)
        self.assertIn("default_runs_per_config", body["settings"])

        status, body = request(self.base + "/api/settings", "PUT",
                               {"default_runs_per_config": 7})
        self.assertEqual(status, 200)
        self.assertEqual(body["settings"]["default_runs_per_config"], 7)

    def test_non_editable_key_is_rejected(self):
        status, body = request(self.base + "/api/settings", "PUT", {"port": 1234})
        self.assertEqual(status, 400)
        self.assertIn("not editable", body["error"]["message"])


class TestExperimentEndpoints(ApiCase):
    def start_experiment(self, **overrides):
        payload = {"model": "fake-model:8b", "prompt_id": "reasoning_trains",
                   "strategy": "baseline", "runs_per_config": 2,
                   "max_tokens": 64, "timeout_seconds": 20}
        payload.update(overrides)
        return request(self.base + "/api/experiments", "POST", payload)

    def wait(self, exp_id, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, body = request(f"{self.base}/api/experiments/{exp_id}/progress")
            if body["status"] in {"completed", "failed", "cancelled"}:
                return body
            time.sleep(0.15)
        raise AssertionError("experiment did not finish")

    def test_validation_errors(self):
        status, body = request(self.base + "/api/experiments", "POST", {"prompt": "hi"})
        self.assertEqual(status, 400)
        status, body = request(self.base + "/api/experiments", "POST",
                               {"model": "fake-model:8b"})
        self.assertEqual(status, 400)

    def test_unknown_model_returns_502(self):
        status, body = self.start_experiment(model="ghost:70b")
        self.assertEqual(status, 502)
        self.assertIn("ghost:70b", body["error"]["message"])

    def test_full_lifecycle(self):
        status, body = self.start_experiment()
        self.assertEqual(status, 201)
        exp_id = body["experiment_id"]

        progress = self.wait(exp_id)
        self.assertEqual(progress["status"], "completed")
        self.assertGreater(progress["progress"]["completed_runs"], 0)

        status, detail = request(f"{self.base}/api/experiments/{exp_id}")
        self.assertEqual(status, 200)
        self.assertTrue(detail["configurations"])

        cfg_id = detail["configurations"][0]["id"]
        status, runs = request(
            f"{self.base}/api/experiments/{exp_id}/runs?configuration_id={cfg_id}")
        self.assertEqual(len(runs["runs"]), 2)

        status, listing = request(self.base + "/api/experiments")
        self.assertTrue(any(e["id"] == exp_id for e in listing["experiments"]))

        # report generation + download
        status, report = request(f"{self.base}/api/experiments/{exp_id}/report",
                                 "POST", {"pdf": True})
        self.assertEqual(status, 200)
        self.assertTrue(report["markdown"])
        self.assertIsNone(report["pdf_error"], report["pdf_error"])

        status, pdf = request(f"{self.base}/api/experiments/{exp_id}/report.pdf")
        self.assertEqual(status, 200)
        self.assertTrue(pdf.startswith(b"%PDF"))

        status, md = request(f"{self.base}/api/experiments/{exp_id}/report.md")
        self.assertEqual(status, 200)
        self.assertIn(b"Executive summary", md)

        status, reports = request(self.base + "/api/reports")
        self.assertTrue(any(r["experiment_id"] == exp_id for r in reports["reports"]))

        status, body = request(f"{self.base}/api/experiments/{exp_id}", "DELETE")
        self.assertEqual(status, 200)
        self.assertIsNone(db.get_experiment(exp_id))

    def test_missing_experiment_is_404(self):
        status, body = request(self.base + "/api/experiments/exp_missing")
        self.assertEqual(status, 404)

    def test_cancel_when_not_running_is_409(self):
        _, body = self.start_experiment()
        exp_id = body["experiment_id"]
        self.wait(exp_id)
        status, _ = request(f"{self.base}/api/experiments/{exp_id}/cancel", "POST")
        self.assertEqual(status, 409)

    def test_report_download_before_generation_is_404(self):
        _, body = self.start_experiment()
        exp_id = body["experiment_id"]
        self.wait(exp_id)
        status, payload = request(f"{self.base}/api/experiments/{exp_id}/report.pdf")
        # AUTO_PDF is disabled in the test sandbox, so no PDF exists yet
        self.assertEqual(status, 404)
        self.assertIn("report", payload["error"]["message"].lower())

    def test_event_stream_emits_progress(self):
        _, body = self.start_experiment(strategy="generation", runs_per_config=2)
        exp_id = body["experiment_id"]
        seen = []
        with urllib.request.urlopen(
                f"{self.base}/api/experiments/{exp_id}/events", timeout=60) as stream:
            deadline = time.time() + 45
            for raw in stream:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    seen.append(json.loads(line[6:]))
                if any(e.get("type") in {"closed", "finished", "cancelled", "failed"}
                       for e in seen) or time.time() > deadline:
                    break
        self.assertTrue(seen)
        self.assertTrue(any(e.get("progress") for e in seen))
        self.wait(exp_id)


class TestOllamaDown(SandboxCase):
    """The API must degrade gracefully, never invent data."""

    def setUp(self):
        super().setUp()
        self.httpd = create_server(host="127.0.0.1", port=0)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        super().tearDown()

    def test_models_endpoint_reports_the_outage(self):
        status, body = request(self.base + "/api/models")
        self.assertEqual(status, 200)
        self.assertFalse(body["connected"])
        self.assertEqual(body["models"], [])
        self.assertIn("hint", body)

    def test_starting_an_experiment_returns_503(self):
        status, body = request(self.base + "/api/experiments", "POST",
                               {"model": "any:8b", "prompt": "hello"})
        self.assertEqual(status, 503)
        self.assertTrue(body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
