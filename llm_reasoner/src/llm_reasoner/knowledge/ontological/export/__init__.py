"""Ontological model exporters."""

"""Exporting a declared model to other formats.

One-way, and deliberately so. The Python model is the source of truth; an export
is a review aid and a conformance target, never something read back in.

``owl``
    Emits OWL/Turtle from the ontological metadata, so a microworld can be
    opened in Protege or run through a reasoner, and -- for Flanagan -- diffed
    against the hand-authored ontology it was transcribed from.
"""

from llm_reasoner.knowledge.ontological.export.owl import (
    ExportDiff,
    KNOWN_DIVERGENCES,
    OwlExport,
    diff_against,
    export_module,
)

__all__ = [
    "ExportDiff",
    "KNOWN_DIVERGENCES",
    "OwlExport",
    "diff_against",
    "export_module",
]