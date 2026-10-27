"""
Tests of EQL source text: its setup statements run, its last expression is the query,
and the names it reads are known before it is evaluated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cramera.resolution.query_source import EmptyQuery, QueryName, QuerySource

SOURCES = Path(__file__).parent / "query_sources"
"""
The query sources the tests read.
"""

# %% evaluating


class TestEvaluation:
    def test_the_last_expression_is_the_value_after_the_setup_ran(self):
        source = QuerySource.read(SOURCES / "setup_then_query.py")
        assert source.value_in({}) == 3

    def test_without_a_last_expression_the_result_variable_is_the_value(self):
        source = QuerySource.read(SOURCES / "result_assigned.py")
        namespace = {}
        assert source.value_in(namespace) == namespace[QueryName.RESULT]

    def test_an_empty_source_is_refused(self):
        with pytest.raises(EmptyQuery):
            QuerySource(text="").value_in({})


# %% reading


class TestOneLine:
    def test_a_source_on_one_line_keeps_every_token_and_no_line_break(self):
        source = QuerySource.read(SOURCES / "reads_and_assigns.py")
        assert "\n" not in source.on_one_line
        assert source.on_one_line.split() == source.text.split()


class TestNamesRead:
    def test_the_names_read_leave_out_those_the_source_assigns(self):
        source = QuerySource.read(SOURCES / "reads_and_assigns.py")
        assert source.names_read() == {"the", "PouringAction", "arm"}
