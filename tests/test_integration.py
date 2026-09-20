"""Optional integration checks against a REAL Ollama instance.

These are skipped unless Ollama is actually reachable, and they are excluded
from the default suite (run them with ``python3 tests/run_tests.py --integration``
or ``python3 -m unittest test_integration``). They perform a small number of real
generations, so they take as long as your model needs.

    OLLAMA_URL=http://127.0.0.1:11434 INTEGRATION_MODEL=llama3.1:8b \
        python3 tests/run_tests.py --integration
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import db  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.engine import MANAGER  # noqa: E402
from app.ollama import OllamaError, OllamaService  # noqa: E402


def live_model():
    """Return a model name if a real Ollama with at least one model is reachable."""
    try:
        service = OllamaService()
        if not service.check_health()["connected"]:
            return None
        models = service.list_models()
    except OllamaError:
        return None
    if not models:
        return None
    wanted = os.environ.get("INTEGRATION_MODEL")
    if wanted:
        return wanted if any(m["name"] == wanted for m in models) else None
    return models[0]["name"]


MODEL = live_model()
REASON = ("no reachable Ollama instance with an installed model "
          f"at {get_settings().ollama_url}")


@unittest.skipUnless(MODEL, REASON)
class TestRealOllama(unittest.TestCase):
    def test_discovery_and_single_generation(self):
        service = OllamaService()
        self.assertTrue(service.get_version())
        caps = service.capabilities(MODEL)
        self.assertEqual(caps["model"], MODEL)

        result = service.generate(MODEL, "Reply with the single word: ready.",
                                  options={"temperature": 0.0, "num_predict": 32},
                                  stream=True, timeout=180)
        self.assertTrue(result.text.strip())
        self.assertIsNotNone(result.time_to_first_token)
        metrics = result.to_metrics()
        self.assertGreater(metrics["output_tokens"], 0)
        self.assertGreater(metrics["tokens_per_second"], 0)

    def test_short_baseline_experiment(self):
        exp_id = MANAGER.start({
            "model": MODEL,
            "prompt_id": "factual_qa_units",
            "strategy": "baseline",
            "runs_per_config": 2,
            "max_tokens": 128,
            "timeout_seconds": 300,
        })
        deadline = time.time() + 1800
        while time.time() < deadline:
            exp = db.get_experiment(exp_id)
            if exp["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(1.0)
        else:
            self.fail("the integration experiment did not finish in 30 minutes")

        self.assertEqual(exp["status"], "completed", exp.get("error"))
        runs = db.list_runs(exp_id)
        self.assertTrue(runs)
        self.assertTrue(any(r["status"] == "ok" for r in runs))


if __name__ == "__main__":
    unittest.main()
