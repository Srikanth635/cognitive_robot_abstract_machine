"""
Run source-navigation commands with the selected interpreter.
"""

from __future__ import annotations

import sys

from repository_knowledge.composition import main, publication_main
from repository_knowledge.interfaces.publishing import PublicationEntry

# %% module entry point

if __name__ == "__main__":
    if sys.argv[1:2] == [PublicationEntry.PUBLICATION]:
        sys.argv.pop(1)
        sys.exit(publication_main())
    sys.exit(main())
