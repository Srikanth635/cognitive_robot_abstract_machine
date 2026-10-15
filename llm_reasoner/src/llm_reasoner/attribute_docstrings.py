"""
The docstrings of the fields of a class: the string written below each field, whole, as
the project's convention documents a field (PEP 257 attribute docstrings).
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap

from krrood.patterns.caching import weak_key_cache
from typing_extensions import Dict

MODULE_FILE = "__file__"
"""
The module attribute naming the file a module was read from, which a module built in
memory lacks.
"""


@weak_key_cache
def attribute_docstrings(described: type) -> Dict[str, str]:
    """
    :param described: A class.
    :return: The whole docstring below each field its own body declares, by field name;
        nothing for a class whose source is not in a file. Read once per class, which is
        held weakly.
    """
    module = sys.modules.get(described.__module__)
    if module is None or vars(module).get(MODULE_FILE) is None:
        return {}
    body = ast.parse(textwrap.dedent(inspect.getsource(described))).body[0].body
    return {
        declaration.target.id: inspect.cleandoc(following.value.value)
        for declaration, following in zip(body, body[1:])
        if isinstance(declaration, ast.AnnAssign)
        and isinstance(declaration.target, ast.Name)
        and isinstance(following, ast.Expr)
        and isinstance(following.value, ast.Constant)
        and isinstance(following.value.value, str)
    }
