"""
Tests for the CLI entry point.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typesafe_sdk import TypeSafeAPIError

from doc_evaluator import cli

from .conftest import StubScoreAnswer, StubSystemOneResponse, StubTypeSafeClient

# %% main


def test_main_reports_an_error_and_returns_1_when_directory_has_no_documents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main([str(tmp_path)])

    assert exit_code == 1
    assert str(tmp_path) in capsys.readouterr().err


def test_main_prints_a_table_and_returns_0_when_every_document_evaluates(
    monkeypatch: pytest.MonkeyPatch,
    document_directory: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    response = StubSystemOneResponse(
        scores={
            "Clarity / readability": StubScoreAnswer(score=2.0, confidence=0.9),
            "Factual accuracy / rigor": StubScoreAnswer(score=1.0, confidence=0.7),
        }
    )
    monkeypatch.setattr(
        cli, "TypeSafeClient", lambda: StubTypeSafeClient(response=response)
    )

    exit_code = cli.main([str(document_directory)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "first.txt" in out
    assert "second.md" in out
    assert "Clarity / readability" in out


def test_main_returns_1_and_lists_failures_when_the_api_call_fails(
    monkeypatch: pytest.MonkeyPatch,
    document_directory: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = TypeSafeAPIError(
        status=500, body=None, headers={}, endpoint="POST /system_one"
    )
    monkeypatch.setattr(cli, "TypeSafeClient", lambda: StubTypeSafeClient(error=error))

    exit_code = cli.main([str(document_directory)])

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Failed to evaluate:" in out
    assert "first.txt" in out
    assert "second.md" in out
