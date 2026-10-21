"""
The names of the fields in the payloads the add-on's routes answer with.
"""

from __future__ import annotations

from enum import StrEnum


class PayloadKey(StrEnum):
    """
    A field of an answer, by the name the panels read it by.
    """

    OK = "ok"
    ERROR = "error"
    ADDON = "addon"
    CRUMB = "crumb"
    ROOT = "root"
    NODES = "nodes"
    EDGES = "edges"
    DETAILS = "details"
    EMPTY = "empty"
    ID = "id"
    LABEL = "label"
    KIND = "kind"
    FROM = "from"
    TO = "to"
    GROUP = "group"
    LINES = "lines"
    INSTRUCTION = "instruction"
    CATEGORY = "category"
    COUNT = "count"
    SOURCE = "source"
    TARGET = "target"
    IRI = "iri"
    PREFIXED = "prefixed"
    DESCRIPTION = "description"
    EVIDENCE = "evidence"
    CLAIMS = "claims"
    TEXT = "text"
    REQUIREMENTS = "requirements"
    DEFINITION = "definition"
    INSTANCES = "instances"
    DESIGNATOR = "designator"
    SCENE = "scene"
    DESIGNATORS = "designators"
    PHASE_STRUCTURE = "phase_structure"
    GEOMETRY_ADJUSTMENT = "geometry_adjustment"
    PERSPECTIVES = "perspectives"
    RESOLUTION = "resolution"
    HYPOTHESIS = "hypothesis"
    VIEWER = "viewer"
