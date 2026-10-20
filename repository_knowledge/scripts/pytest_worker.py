"""
Run pytest after the parent attaches the process to its memory-limited job.
"""

import runpy
import sys

# %% memory gate

if __name__ == "__main__":
    if sys.stdin.readline().strip() == "run":
        sys.argv = sys.argv[1:]
        runpy.run_module("pytest", run_name="__main__")
