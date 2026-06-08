"""Shared pytest configuration.

The toolkit is a flat-layout package: modules import each other by bare name
(``from tools import ...``) relying on ``agentic_guardrails/`` being on sys.path
at runtime (run_agentic_comparison.py inserts it). Mirror that here so the same
imports resolve under pytest.
"""

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent / "agentic_guardrails"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
