"""Ollama service layer: connection handling, discovery, generation, errors."""

from __future__ import annotations

import unittest

from support import SandboxCase
from fake_ollama import FakeOllama

from app.ollama import (OllamaService, OllamaModelMissing, OllamaUnavailable,
                        sanitize_options, human_size, parse_parameters_block,
                        extract_modelfile_system)


class TestConnectionFailures(SandboxCase):
    """Nothing is fabricated when Ollama is missing: errors are structured."""

    def test_health_reports_disconnected(self):
        health = OllamaService().check_health()
        self.assertFalse(health["connected"])
        self.assertIn("message", health["error"])

    def test_list_models_raises_unavailable(self):
        with self.assertRaises(OllamaUnavailable):
            OllamaService().list_models()


class TestAgainstFakeOllama(SandboxCase):
    def setUp(self):
        super().setUp()
        self.fake = FakeOllama().start()
        self.use_ollama(self.fake.url)
        self.service = OllamaService()

    def tearDown(self):
        self.fake.stop()
        super().tearDown()

    def test_version_and_health(self):
        self.assertEqual(self.service.get_version(), "0.5.13")
        health = self.service.check_health()
        self.assertTrue(health["connected"])
        self.assertEqual(health["version"], "0.5.13")

    def test_discovery_returns_metadata(self):
        models = self.service.list_models()
        self.assertEqual(len(models), 2)
        first = models[0]
        for key in ("name", "size_bytes", "parameter_size", "quantization_level", "family",
                    "format", "modified_at", "digest", "size_human"):
            self.assertIn(key, first)
        self.assertEqual(first["parameter_size"], "8B")
        self.assertEqual(first["quantization_level"], "Q4_K_M")

    def test_capabilities_include_context_length(self):
        caps = self.service.capabilities("fake-model:8b")
        self.assertEqual(caps["quantization_level"], "Q4_K_M")
        self.assertEqual(caps.get("context_length"), 8192)
        self.assertTrue(caps.get("version"))

    def test_model_info_missing_model(self):
        with self.assertRaises(OllamaModelMissing):
            self.service.get_model_info("does-not-exist:1b")

    def test_generate_streaming_measures_ttft(self):
        result = self.service.generate("fake-model:8b", "Say hello.",
                                       options={"temperature": 0.0}, stream=True)
        self.assertTrue(result.text)
        self.assertIsNotNone(result.time_to_first_token)
        self.assertGreater(result.stream_chunks, 1)
        metrics = result.to_metrics()
        self.assertGreater(metrics["output_tokens"], 0)
        self.assertGreater(metrics["tokens_per_second"], 0)
        self.assertGreater(metrics["wall_seconds"], 0)

    def test_generate_non_streaming_has_no_ttft(self):
        """TTFT must be absent rather than estimated when not streaming."""
        result = self.service.generate("fake-model:8b", "Say hello.", stream=False)
        self.assertTrue(result.text)
        self.assertIsNone(result.time_to_first_token)
        self.assertIsNone(result.to_metrics().get("time_to_first_token"))

    def test_generate_missing_model_raises(self):
        with self.assertRaises(OllamaModelMissing):
            self.service.generate("ghost:1b", "hi", stream=False)

    def test_chat_endpoint(self):
        result = self.service.chat("fake-model:8b",
                                   [{"role": "user", "content": "score this"}])
        self.assertIn("correctness", result.text)


class TestHelpers(unittest.TestCase):
    def test_sanitize_options_drops_unknown_keys(self):
        clean, dropped = sanitize_options({"temperature": 0.5, "made_up_param": 1})
        self.assertEqual(clean, {"temperature": 0.5})
        self.assertEqual(dropped, ["made_up_param"])

    def test_sanitize_options_strips_internal_keys(self):
        clean, _ = sanitize_options({"_stream": False, "top_k": 40})
        self.assertEqual(clean, {"top_k": 40})

    def test_human_size(self):
        self.assertTrue(human_size(4_700_000_000).endswith("GB"))
        self.assertIsNone(human_size(None))

    def test_parse_parameters_block(self):
        parsed = parse_parameters_block('temperature 0.8\nstop "<|end|>"')
        self.assertEqual(parsed["temperature"], 0.8)

    def test_extract_modelfile_system(self):
        system = extract_modelfile_system('FROM x\nSYSTEM """You are careful."""')
        self.assertEqual(system, "You are careful.")


if __name__ == "__main__":
    unittest.main()
