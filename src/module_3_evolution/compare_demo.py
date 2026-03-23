"""Alias for the Module 3 full demo (Module 2 + GA + selection + reason).

Prefer:
    python -m src.module_3_evolution.compare_demo

Also works when run as a script (double-click / IDE / path to file) because we
insert the repo root on ``sys.path`` before importing ``src``.

Run:
    python -m src.module_3_evolution.compare_demo
"""

from __future__ import annotations

import pathlib
import sys

# Repo root must be on sys.path before ``from src...`` (needed for direct execution).
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.module_3_strategy_selection.demo_selection import main

if __name__ == "__main__":
    main()
