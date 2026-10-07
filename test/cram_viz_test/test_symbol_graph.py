"""
Tests that the viewer's process can read and query hypotheses outside pytest, where no
fixture builds krrood's symbol graph after every vocabulary class is imported.
"""

import subprocess
import sys
from pathlib import Path

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from cram_viz import hypothesis_knowledge  # noqa: E402

QUERY_SCRIPT = Path(__file__).parent / "scripts" / "query_sample_outside_pytest.py"
"""
A script querying the sample hypothesis in a process of its own.
"""


def test_sample_is_queryable_in_the_servers_own_process():
    result = subprocess.run(
        [sys.executable, str(QUERY_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().splitlines()[-1] == str(
        len(hypothesis_knowledge.sample_view().root.defines_phase)
    )
