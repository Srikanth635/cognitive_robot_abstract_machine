"""
The evidence format must not initialize unrelated runtime subsystems.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# %% serializer dependency boundary


def test_nested_evidence_round_trip_does_not_import_eql() -> None:
    """
    Basic evidence encoding remains available without the EQL runtime.
    """
    script = Path(__file__).parent / "dataset/serialization_without_eql.py"
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
