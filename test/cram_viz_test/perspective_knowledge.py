"""
The knowledge a resolution consults when perspectives are offered: the phase structure
of pouring, the action core's frame and one hypothesis for each perspective, all stored
together, with a resolution that rests on fragments of every one of them.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.access import (
    ActionPerspectiveFile,
    DescribedActions,
    HypothesisBackend,
    HypothesisStore,
    KnowledgeService,
    PERSPECTIVE_ROOTS,
    StoredRoot,
)
from llm_reasoner.knowledge.admission.consistency import ConsistencyCheck
from coraplex.robot_plans.actions.composite.tool_based import PouringAction
from llm_reasoner.resolver.grounds import GroundKey
from llm_reasoner.world.classes import identifier_of
from rdflib import Graph
from typing_extensions import List

from cram_viz import hypothesis_knowledge, paths
from cram_viz.resolution.resolver import ResolvedDesignator, ShownGround

from .resolutions import objects_given

CASES = Path(__file__).parent / "perspective_cases"
"""
The recorded answers of a model describing pouring in each action perspective module.
"""

ANSWER_FILES = {
    ActionPerspectiveFile.ACTION_CORE: "pouring_core.json",
    ActionPerspectiveFile.ACTION_FRAME: "pouring_frame.json",
    ActionPerspectiveFile.IMAGE_SCHEMAS: "pouring_image_schemas.json",
    ActionPerspectiveFile.FAILURES: "pouring_failures.json",
}
"""
The recorded answer describing pouring, by the module it is written in.
"""


class AlwaysConsistent(ConsistencyCheck):
    """
    A consistency check that finds every hypothesis consistent.
    """

    def is_consistent(self, graph: Graph) -> bool:
        return True


@dataclass
class ConsultedKnowledge:
    """
    Everything a resolution consulted, stored in one knowledge component.
    """

    service: KnowledgeService
    """
    The knowledge component holding all of it.
    """

    core_record_id: str
    """
    The record of the action core's frame of pouring.
    """

    perspective_record_ids: List[str]
    """
    The record of each perspective on that frame, in the order of the modules.
    """

    @property
    def consulted_record_ids(self) -> List[str]:
        """
        The core first, then each perspective, as a resolution lists them.
        """
        return [self.core_record_id, *self.perspective_record_ids]

    def resolution(self) -> ResolvedDesignator:
        """
        :return: A resolution of pouring that consulted the phase structure, the core
            and every perspective, resting on a fragment of each core and perspective.
        """
        recorded = objects_given()
        cited = [
            self.service.view(record_id).root for record_id in self.consulted_record_ids
        ]
        grounds = [
            ShownGround(
                GroundKey.KNOWLEDGE,
                name=self.service.view(record_id).citation_of(root).fragment_name,
                identifier=self.service.view(record_id).citation_of(root).iri,
            )
            for record_id, root in zip(self.consulted_record_ids, cited)
        ]
        (action,) = recorded.actions
        return replace(
            recorded,
            actions=[replace(action, grounds=[*action.grounds, *grounds])],
            record_id=hypothesis_knowledge.SAMPLE_RECORD_ID,
            perspective_record_ids=self.consulted_record_ids,
        )


def consulted_knowledge(directory: Path) -> ConsultedKnowledge:
    """
    :param directory: Where the knowledge is stored.
    :return: The sample phase structure of pouring stored with the recorded core and
        perspective hypotheses of pouring, served as the resolver serves them.
    """
    shutil.copytree(paths.SAMPLE_HYPOTHESES_PATH, directory, dirs_exist_ok=True)
    perspectives = [
        perspective
        for perspective in ActionPerspectiveFile
        if perspective in PERSPECTIVE_ROOTS
    ]
    service = KnowledgeService(
        backend=HypothesisBackend(
            client=ScriptedClient(
                responses=[
                    (CASES / ANSWER_FILES[module]).read_text()
                    for module in (ActionPerspectiveFile.ACTION_CORE, *perspectives)
                ]
            ),
            store=HypothesisStore(directory=directory),
            consistency=AlwaysConsistent(),
        )
    )
    described = DescribedActions().describing(identifier_of(PouringAction))
    core_record = service.serve(described.frame_query())
    return ConsultedKnowledge(
        service=service,
        core_record_id=core_record,
        perspective_record_ids=[
            service.serve(
                PERSPECTIVE_ROOTS[perspective].query(StoredRoot(record_id=core_record))
            )
            for perspective in perspectives
        ],
    )
