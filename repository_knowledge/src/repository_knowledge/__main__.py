"""
Run source-navigation commands with the selected interpreter.
"""

from __future__ import annotations

import sys

from repository_knowledge.composition import main

# %% module entry point

if __name__ == "__main__":
    sys.exit(main())
