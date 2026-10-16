"""
``python -m experiments.frame_comparison <directory>``: summarise the records a
comparison kept in a directory, writing the summary beside them and printing it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import asdict

from typing_extensions import List, Optional

from experiments.frame_comparison.output import ComparisonOutput
from experiments.frame_comparison.records import summarise


def main(argv: Optional[List[str]] = None) -> None:
    """
    Summarise a comparison's records and print one line per measurement, case and arm.
    """
    parser = argparse.ArgumentParser(prog="python -m experiments.frame_comparison")
    parser.add_argument("directory", help="where the comparison kept its records")
    output = ComparisonOutput(
        directory=Path(
            parser.parse_args(sys.argv[1:] if argv is None else argv).directory
        )
    )
    output.write_summary()
    for summary in summarise(output.records()):
        print(asdict(summary))


if __name__ == "__main__":
    main()
