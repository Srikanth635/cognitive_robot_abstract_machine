"""
Where the knowledge of each perspective on the action core is rooted: the class whose
one fragment describes a frame of the core, and the field naming that frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.factories import the
from krrood.entity_query_language.query.match import Match
from typing_extensions import Any

from llm_reasoner.knowledge.ontologies import ActionPerspectiveFile
from llm_reasoner.knowledge.ontologies.action_perspectives import action_frame
from llm_reasoner.knowledge.ontologies.terms import DescriptionEntity


@dataclass(frozen=True)
class PerspectiveRoot:
    """
    The class a perspective's knowledge is rooted in, and the field of it naming the
    frame of the action core it describes.
    """

    root: type[DescriptionEntity]
    """
    The class of the fragment the perspective's knowledge is rooted in.
    """

    frame_field: str
    """
    The field of that fragment naming the frame it describes.
    """

    def query(self, frame: Any) -> Match:
        """
        :param frame: The frame described, such as the root of a stored core hypothesis.
        :return: A query for the one fragment of this perspective describing the frame.
        """
        return the(self.root)(**{self.frame_field: frame})


PERSPECTIVE_ROOTS: dict[ActionPerspectiveFile, PerspectiveRoot] = {
    ActionPerspectiveFile.ACTION_FRAME: PerspectiveRoot(
        root=action_frame.FrameStereotype, frame_field="stereotype_of"
    ),
}
"""
The root of each perspective on the action core, by its module.
"""
