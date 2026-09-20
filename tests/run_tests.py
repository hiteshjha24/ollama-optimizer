#!/usr/bin/env python3
"""Run the whole test-suite with the standard library's unittest runner.

    python3 tests/run_tests.py            # everything (uses the mock Ollama)
    python3 tests/run_tests.py -v         # verbose
    python3 tests/run_tests.py test_api   # one module
    python3 tests/run_tests.py --integration   # additionally hit a real Ollama

The default suite never touches a real Ollama instance or your real database:
every case runs against tests/fake_ollama.py with a temporary data directory.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama Optimizer test-suite")
    parser.add_argument("modules", nargs="*", help="module names, e.g. test_db")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--integration", action="store_true",
                        help="also run tests against a real Ollama instance")
    args = parser.parse_args()

    loader = unittest.TestLoader()
    if args.modules:
        suite = loader.loadTestsFromNames(args.modules)
    else:
        suite = loader.discover(start_dir=str(HERE), top_level_dir=str(HERE),
                                pattern="test_*.py")
        if not args.integration:
            suite = _without_integration(suite)

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def _without_integration(suite: unittest.TestSuite) -> unittest.TestSuite:
    """Drop the live-Ollama integration module unless it was asked for."""
    kept = unittest.TestSuite()
    for test in suite:
        name = getattr(test, "_testMethodName", "")
        module = test.__class__.__module__
        if isinstance(test, unittest.TestSuite):
            kept.addTest(_without_integration(test))
        elif "integration" not in module and "integration" not in name:
            kept.addTest(test)
    return kept


if __name__ == "__main__":
    raise SystemExit(main())
