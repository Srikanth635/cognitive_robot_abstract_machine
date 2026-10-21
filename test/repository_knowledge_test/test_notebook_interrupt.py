"""
The Windows memory guard preserves normal Jupyter interrupts.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import nbformat
import pytest
from jupyter_client import AsyncKernelManager
from nbclient import NotebookClient

from .dataset.execute_replies import ExecutionRecorder, ExecutionReply, ExecutionStatus

# %% guarded Windows kernel


@pytest.mark.skipif(os.name != "nt", reason="The launcher uses Windows job objects.")
def test_memory_guard_preserves_kernel_interrupts() -> None:
    """
    Interrupt running notebook work without terminating its resource guard.
    """
    shell = shutil.which("pwsh.exe")
    if shell is None:
        pytest.skip("PowerShell is required by the Windows memory guard.")
    root = Path(__file__).parents[2]
    source = (Path(__file__).parent / "dataset/interruptible.py").read_text(
        encoding="utf-8"
    )
    document = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    manager = AsyncKernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        shell,
        "-NoProfile",
        "-File",
        str(root / "repository_knowledge/scripts/run_kernel.ps1"),
        "-ConnectionFile",
        "{connection_file}",
    ]
    manager.kernel_spec.interrupt_mode = "signal"
    recorder = ExecutionRecorder()
    NotebookClient(
        document,
        km=manager,
        timeout=2,
        interrupt_on_timeout=True,
        allow_errors=True,
        startup_timeout=90,
        on_cell_executed=recorder.record,
    ).execute()
    assert recorder.replies == [
        ExecutionReply(ExecutionStatus.ERROR, KeyboardInterrupt.__name__)
    ]
