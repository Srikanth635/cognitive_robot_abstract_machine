"""PyCRAM integration — the single boundary layer between llmr and pycram.

All pycram imports for the entire llmr package are funnelled through
the adapter module in this package. Nothing outside pycram imports
pycram directly.

For field introspection (``FieldKind``, ``ActionFieldIntrospector``, etc.) use
:mod:`llmr.bridge.introspect` — the canonical home.

Public surface:
  PycramContext       — structural protocol matching pycram Context
  PycramPlanNode      — structural protocol matching pycram PlanNode
  execute_single      — wraps pycram.plans.factories.execute_single
  discover_action_classes — scans pycram action package tree
"""

from llmr.pycram.adapter import (
    PycramContext,
    PycramPlanNode,
    discover_action_classes,
    execute_single,
)

__all__ = [
    "PycramContext",
    "PycramPlanNode",
    "discover_action_classes",
    "execute_single",
]
