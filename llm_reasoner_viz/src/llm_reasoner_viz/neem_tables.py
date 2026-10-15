"""
How the tables of the NEEM database extend one another.

The ORM keeps a class in a table for each level of its inheritance, each row sharing its
identifier with the row of the class it extends, so what a row holds is spread over a
chain of tables that the foreign keys of the identifier give.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from typing_extensions import Dict, List, Optional, Set

from llm_reasoner_viz.neem_schema import DAO_SUFFIX, Column

REFERENCE_PREFIX = "_"
"""
What begins the name of a column that refers to a row of another table.
"""

REFERENCE_SUFFIX = "_id"
"""
What ends the name of a column that refers to a row of another table.
"""

MAPPING_SUFFIX = "Mapping"
"""
What the ORM adds to the name of a class it keeps through a mapping, such as a pose.
"""

BINARY_TYPE = "BLOB"
"""
The declared type of a column of bytes.
"""

BOOLEAN_TYPE = "BOOLEAN"
"""
The declared type of a column that holds true or false, which SQLite stores as 1 or 0.
"""

STRUCTURE_COLUMNS = (Column.IDENTIFIER, Column.POLYMORPHIC_TYPE)
"""
The columns that tie the rows of a family together instead of holding something.
"""


@dataclass(frozen=True)
class TableColumn:
    """
    A column of a table of the database.
    """

    name: str
    """
    The name of the column.
    """

    declared_type: str
    """
    The type the table declares for the column.
    """

    references: Optional[str]
    """
    The table whose row the column refers to; none for a column that holds a value.
    """

    @property
    def is_boolean(self) -> bool:
        """
        Whether the column holds true or false.
        """
        return self.declared_type.upper() == BOOLEAN_TYPE

    @property
    def is_binary(self) -> bool:
        """
        Whether the column holds bytes, which the viewer leaves out.
        """
        return self.declared_type.upper() == BINARY_TYPE

    @property
    def attribute(self) -> str:
        """
        What the column holds, as the class it keeps calls it: the name of a column that
        refers to a row without the marks of a reference.
        """
        if self.references is None:
            return self.name
        return self.name.removeprefix(REFERENCE_PREFIX).removesuffix(REFERENCE_SUFFIX)


@dataclass
class TableSchema:
    """
    The tables of an open database, and how each extends another.
    """

    connection: sqlite3.Connection
    """
    The connection to read the definitions of the tables from.
    """

    known_columns: Dict[str, List[TableColumn]] = field(default_factory=dict)
    """
    The columns of each table already read.
    """

    known_tables: Optional[Set[str]] = None
    """
    The names of the tables of the database, once read.
    """

    @staticmethod
    def class_name(table: str) -> str:
        """
        :return: The name of the class a table keeps, which is the name of the table
            without what the ORM adds to it.
        """
        return table.removesuffix(DAO_SUFFIX).removesuffix(MAPPING_SUFFIX)

    def has_table(self, table: str) -> bool:
        """
        :return: Whether the database has a table of the name.
        """
        if self.known_tables is None:
            self.known_tables = {
                name
                for (name,) in self.connection.execute(
                    "select name from sqlite_master where type = 'table'"
                )
            }
        return table in self.known_tables

    def columns(self, table: str) -> List[TableColumn]:
        """
        :return: The columns of a table, in the order the table declares them.
        """
        if table not in self.known_columns:
            references = {
                row[3]: row[2]
                for row in self.connection.execute(
                    f'pragma foreign_key_list("{table}")'
                )
                if row[3] != Column.IDENTIFIER
            }
            self.known_columns[table] = [
                TableColumn(
                    name=name,
                    declared_type=declared_type,
                    references=references.get(name),
                )
                for _, name, declared_type, *_ in self.connection.execute(
                    f'pragma table_info("{table}")'
                )
            ]
        return self.known_columns[table]

    def parent_of(self, table: str) -> Optional[str]:
        """
        :return: The table of the class that the class of a table extends, none for a table
            at the root of a family.
        """
        return next(
            (
                row[2]
                for row in self.connection.execute(
                    f'pragma foreign_key_list("{table}")'
                )
                if row[3] == Column.IDENTIFIER
            ),
            None,
        )

    def chain(self, table: str) -> List[str]:
        """
        :return: The tables a row of the table is spread over, from the root of its family
            to the table.
        """
        parent = self.parent_of(table)
        return [table] if parent is None else [*self.chain(parent), table]

    def parameter_columns(self, table: str) -> List[TableColumn]:
        """
        :return: The columns of a table that hold a value, leaving out those that tie the
            rows of a family together, those that refer to other rows and those of bytes.
        """
        return [
            column
            for column in self.columns(table)
            if column.references is None
            and column.name not in STRUCTURE_COLUMNS
            and not column.is_binary
        ]

    def reference_columns(self, table: str) -> List[TableColumn]:
        """
        :return: The columns of a table that refer to a row of another table.
        """
        return [column for column in self.columns(table) if column.references]
