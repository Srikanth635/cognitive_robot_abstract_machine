"""
A mimic of a viewer extension contributing knowledge of its own to the EQL panel:
records titled by a label rather than a name, an answer rendering for them, and a preset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List

from cramera.knowledge.presets import Preset
from cramera.knowledge.query_contribution import QueryContribution
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.query_runner import AnswerRendering

# %% the records


@dataclass
class LabelledRecord:
    """
    A record titled by its label, which the viewer's own renderer does not know.
    """

    label: str
    """
    What the record is called.
    """

    notes: List[str] = field(default_factory=list)
    """
    Further text, which an answer about the record leaves out.
    """


# %% how answers about them read


class LabelledRecordRendering(AnswerRendering):
    """
    Writes a labelled record as its label, and lights up the node named after it.
    """

    def renders(self, value: Any) -> bool:
        return isinstance(value, LabelledRecord)

    def rendered(self, value: Any) -> Any:
        return value.label

    def highlights_of(self, value: Any) -> List[str]:
        return [self.node_of(value)]

    @staticmethod
    def node_of(record: LabelledRecord) -> str:
        """
        :return: The graph node the record stands for.
        """
        return "node:" + record.label


# %% the contribution


@dataclass
class LabelledRecordContribution(QueryContribution):
    """
    Labelled records, offered to the EQL panel as the variable ``labelled_record``.
    """

    records: List[LabelledRecord]
    """
    The records contributed.
    """

    def domains(self) -> List[QueryDomain]:
        return [QueryDomain("labelled_record", LabelledRecord, self.records)]

    def names(self) -> Dict[str, Any]:
        return {"FIRST_LABEL": self.records[0].label}

    def presets(self) -> List[Preset]:
        return [
            Preset(text="which records are there?", code="an(entity(labelled_record))")
        ]

    def answers(self) -> AnswerRendering:
        return LabelledRecordRendering()
