"""
EQL source text: setup statements followed by the query as the last expression, evaluated
in a namespace that binds the names it reads.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from typing_extensions import Any, Dict, List, Set


class QueryName(StrEnum):
    """
    Names with a meaning of their own in query source.
    """

    RESULT = "result"
    """
    The variable holding the query's value when its last statement is not an
    expression.
    """

    UNNAMED = "<eql>"
    """
    Where source typed rather than read from a file comes from, as tracebacks name it.
    """


@dataclass
class EmptyQuery(ValueError):
    """
    The query source holds no statement.
    """

    origin: str
    """
    Where the source came from.
    """

    def __post_init__(self):
        super().__init__(f"empty query in {self.origin}")


@dataclass(frozen=True)
class QuerySource:
    """
    The source of one EQL query: setup statements, then the query itself as the last
    expression.
    """

    text: str
    """
    The source as written.
    """

    origin: str = QueryName.UNNAMED
    """
    Where the source came from, as tracebacks name it.
    """

    @classmethod
    def read(cls, path: Path) -> QuerySource:
        """
        :return: The source written in a file.
        """
        return cls(text=path.read_text(), origin=str(path))

    @property
    def on_one_line(self) -> str:
        """
        :return: The source with its line breaks and indentation folded into single
            spaces.
        """
        return " ".join(self.text.split())

    @cached_property
    def statements(self) -> List[ast.stmt]:
        """
        :return: The source's statements.
        :raises EmptyQuery: When it holds none.
        """
        statements = ast.parse(self.text, filename=self.origin, mode="exec").body
        if not statements:
            raise EmptyQuery(origin=self.origin)
        return statements

    def value_in(self, namespace: Dict[str, Any]) -> Any:
        """
        Run the source in a namespace, which it may extend with names of its own.

        :return: The value of the last expression, or of the variable
            :attr:`QueryName.RESULT` when the last statement is not an expression.
        """
        *setup, last = self.statements
        if not isinstance(last, ast.Expr):
            self.run(self.statements, namespace)
            return namespace.get(QueryName.RESULT)
        self.run(setup, namespace)
        return eval(  # noqa: S307
            compile(ast.Expression(last.value), self.origin, "eval"), namespace
        )

    def run(self, statements: List[ast.stmt], namespace: Dict[str, Any]) -> None:
        """
        Run statements of the source in a namespace.
        """
        exec(  # noqa: S102
            compile(ast.Module(body=statements, type_ignores=[]), self.origin, "exec"),
            namespace,
        )

    def names_read(self) -> Set[str]:
        """
        :return: The names the source reads without assigning them itself, which the
            namespace it runs in has to bind.
        """
        names = [
            node
            for statement in self.statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
        ]
        assigned = {name.id for name in names if isinstance(name.ctx, ast.Store)}
        return {
            name.id
            for name in names
            if isinstance(name.ctx, ast.Load) and name.id not in assigned
        }
