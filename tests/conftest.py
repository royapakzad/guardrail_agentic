"""
tests/conftest.py
-----------------
Add agentic_guardrails/ to sys.path so tests can import directly
(the package uses flat imports — no package structure).
pyproject.toml also sets pythonpath = ["agentic_guardrails"] for pytest 7+.
This conftest is kept as a fallback for editors and direct pytest invocations.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic_guardrails"))
