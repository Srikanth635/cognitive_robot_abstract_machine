"""Tests for workflow utility functions."""

import json
from pathlib import Path

import pytest

from llmr.workflows.utils import read_json_from_file, remove_think_tags


class TestReadJsonFromFile:
    def test_reads_valid_json(self, tmp_path: Path) -> None:
        data = {"key": "value", "number": 42}
        file = tmp_path / "test.json"
        file.write_text(json.dumps(data))
        assert read_json_from_file(str(file)) == data

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_json_from_file(str(tmp_path / "nonexistent.json"))

    def test_raises_value_error_on_invalid_json(self, tmp_path: Path) -> None:
        file = tmp_path / "bad.json"
        file.write_text("not valid json {{{")
        with pytest.raises(ValueError):
            read_json_from_file(str(file))

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        file = tmp_path / "data.json"
        file.write_text('{"a": 1}')
        assert read_json_from_file(file) == {"a": 1}


class TestRemoveThinkTags:
    def test_removes_think_block(self) -> None:
        text = "<think>some internal reasoning</think>actual output"
        assert remove_think_tags(text) == "actual output"

    def test_removes_multiline_think_block(self) -> None:
        text = "<think>\nline one\nline two\n</think>result"
        assert remove_think_tags(text) == "result"

    def test_no_think_block_returns_stripped(self) -> None:
        assert remove_think_tags("  hello  ") == "hello"

    def test_empty_string(self) -> None:
        assert remove_think_tags("") == ""

    def test_think_remover_alias(self) -> None:
        from llmr.workflows.utils import think_remover

        assert think_remover("<think>x</think>y") == "y"
