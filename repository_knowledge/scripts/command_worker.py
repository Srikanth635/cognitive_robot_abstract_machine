"""
Run a selected module after the parent attaches its memory-limited job.
"""

from __future__ import annotations

import runpy
import sys

# %% memory gate

if __name__ == "__main__":
    if sys.stdin.readline().strip() == "run":
        sys.argv = sys.argv[1:]
        runpy.run_module(sys.argv[0], run_name="__main__")
