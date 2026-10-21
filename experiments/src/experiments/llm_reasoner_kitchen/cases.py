"""
The cases the resolver and the knowledge request are checked with on the breakfast scene:
cutting and pouring, from almost given to refusals, with and without a task, each with
what a correct resolution satisfies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from coraplex.datastructures.enums import Arms, CuttingTechnique
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    PouringAction,
)
from krrood.entity_query_language.factories import (
    an,
    and_,
    contains,
    exists,
    in_,
    not_,
    or_,
    the,
    variable,
)
from krrood.entity_query_language.query.match import Match
from llm_reasoner.knowledge.access import action_core
from llm_reasoner.resolver.exceptions import (
    LanguageModelBackendError,
    NoFittingReferent,
    UnsatisfiableRequest,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CuttingKnife,
    Food,
    PouringCup,
    Tool,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, Callable, Dict, List, Optional, Tuple

from experiments.llm_reasoner_kitchen.scene import SceneBody

SEVERAL = 3
"""
How many alternatives a case asking for several resolutions asks for.
"""

SENSIBLE_SLICE = (0.005, 0.05)
"""
The slice thicknesses, in metres, a cut without further constraint is expected within.
"""

# %% queries on the scene


@dataclass
class KitchenQueries:
    """
    Writes the cases' queries on one breakfast scene.
    """

    world: World
    """
    The world the scene is set out in.
    """

    def body(self, name: SceneBody) -> Body:
        """
        :return: The scene's body of the given name.
        """
        return self.world.get_body_by_name(name)

    def annotation(self, kind: type) -> Any:
        """
        :return: The scene's one annotation of exactly the given kind.
        """
        (found,) = [
            annotation
            for annotation in self.world.semantic_annotations
            if type(annotation) is kind
        ]
        return found

    @staticmethod
    def cutting_slots() -> Dict[str, Any]:
        """
        :return: A cutting action's usual open slots: any body, any tool, the arm, the
            technique and the slice thickness.
        """
        return dict(
            object_to_cut=variable(Body),
            tool=variable(Tool),
            arm=...,
            technique=...,
            slice_thickness=...,
        )

    @staticmethod
    def pouring_slots() -> Dict[str, Any]:
        """
        :return: A pouring action's usual open slots: any body as the target, any tool
            as the source, the arm and the tilt angle.
        """
        return dict(
            target_container=variable(Body),
            source_container=variable(Tool),
            arm=...,
            tilt_angle=...,
        )

    def cut(self, **given: Any) -> Match:
        """
        :return: A query for one cutting action, the given slots replacing the usual.
        """
        return the(CuttingAction)(**{**self.cutting_slots(), **given})

    def pour(self, **given: Any) -> Match:
        """
        :return: A query for one pouring action, the given slots replacing the usual.
        """
        return the(PouringAction)(**{**self.pouring_slots(), **given})

    def pour_from_the_cup_into_the_bowl(self) -> Match:
        """
        :return: A query for pouring from the cup into the bowl, the arm and the tilt
            left open.
        """
        return self.pour(
            target_container=variable(Body, domain=[self.body(SceneBody.BOWL)]),
            source_container=variable(Tool, domain=[self.annotation(PouringCup)]),
        )

    @staticmethod
    def several(action: type, slots: Dict[str, Any]) -> Match:
        """
        :return: A query for several alternative actions.
        """
        query = an(action)(**slots)
        query.expression.limit(SEVERAL)
        return query

    @staticmethod
    def where(query: Match, conditions: Callable[[Any], List[Any]]) -> Match:
        """
        :return: The query with the conditions on its variable.
        """
        return query.where(*conditions(query.variable))


# %% what a resolution satisfies


def name_of(body: Body) -> str:
    """
    :return: A body's name.
    """
    return body.name.name


def cuts_the_bread_with_the_knife(actions: List[CuttingAction]) -> bool:
    """
    :return: Whether every action cuts the bread with the cutting knife.
    """
    return all(
        name_of(action.object_to_cut) == SceneBody.BREAD
        and isinstance(action.tool, CuttingKnife)
        for action in actions
    )


def pours_from_the_cup_into_the_bowl(actions: List[PouringAction]) -> bool:
    """
    :return: Whether every action pours from the cup into the bowl.
    """
    return all(
        name_of(action.target_container) == SceneBody.BOWL
        and isinstance(action.source_container, PouringCup)
        for action in actions
    )


def tilts_within_a_turn(actions: List[PouringAction]) -> bool:
    """
    :return: Whether every action tilts by more than nothing and at most half a turn.
    """
    return all(0 < action.tilt_angle <= math.pi for action in actions)


# %% cases


@dataclass(frozen=True)
class ResolutionCase:
    """
    A query to resolve on the breakfast scene, and what a correct resolution satisfies.
    """

    identifier: str
    """
    The case's name, such as ``C1``.
    """

    expected: str
    """
    What a correct resolution is, in words.
    """

    query: Callable[[KitchenQueries], Match]
    """
    Writes the case's query on a scene.
    """

    meets: Callable[[List[Any]], bool] = field(default=lambda actions: bool(actions))
    """
    Whether the resolved actions are what the case expects.
    """

    task: str = ""
    """
    What the caller is trying to achieve, none when empty.
    """

    refusal: Optional[type[LanguageModelBackendError]] = None
    """
    The refusal a correct resolution ends in, when it ends in one.
    """

    furnished: bool = False
    """
    Whether the scene needs its furniture annotated.
    """


@dataclass(frozen=True)
class KnowledgeCase:
    """
    A request to pour from the cup into the bowl with a task, and the situation the
    knowledge consulted for it is expected to be for.
    """

    identifier: str
    """
    The case's name, such as ``K2``.
    """

    task: str
    """
    What the caller is trying to achieve, none when empty.
    """

    situation: Tuple[StrEnum, ...]
    """
    The situation features' values the consultation is expected to choose; a feature
    whose value is not listed is expected to stay unknown.
    """


RESOLUTION_CASES = (
    ResolutionCase(
        "C1",
        "a sensible slice thickness, about 1-3 cm",
        lambda q: q.cut(
            object_to_cut=variable(Body, domain=[q.body(SceneBody.BREAD)]),
            tool=variable(Tool, domain=[q.annotation(CuttingKnife)]),
            arm=Arms.RIGHT,
            technique=CuttingTechnique.SLICE,
        ),
        meets=lambda actions: all(
            SENSIBLE_SLICE[0] <= action.slice_thickness <= SENSIBLE_SLICE[1]
            for action in actions
        ),
    ),
    ResolutionCase(
        "P1",
        "an arm and a plausible tilt angle",
        lambda q: q.pour_from_the_cup_into_the_bowl(),
        meets=tilts_within_a_turn,
    ),
    ResolutionCase(
        "C2",
        "bread and the knife, not the cup",
        lambda q: q.cut(
            object_to_cut=variable(
                Body,
                domain=[
                    q.body(SceneBody.BREAD),
                    q.body(SceneBody.BOWL),
                    q.body(SceneBody.MILK),
                ],
            )
        ),
        meets=cuts_the_bread_with_the_knife,
    ),
    ResolutionCase(
        "P2",
        "the bowl, from the cup",
        lambda q: q.pour(
            target_container=variable(
                Body,
                domain=[
                    q.body(SceneBody.BOWL),
                    q.body(SceneBody.BREAD),
                    q.body(SceneBody.SPOON),
                ],
            )
        ),
        meets=pours_from_the_cup_into_the_bowl,
    ),
    ResolutionCase(
        "C3",
        "bread and the knife, found among all bodies",
        lambda q: q.cut(),
        meets=cuts_the_bread_with_the_knife,
    ),
    ResolutionCase(
        "P3",
        "the bowl, from the cup, found among all bodies",
        lambda q: q.pour(),
        meets=pours_from_the_cup_into_the_bowl,
    ),
    ResolutionCase(
        "C4",
        "a slice thickness below 0.015",
        lambda q: q.where(q.cut(), lambda v: [v.slice_thickness < 0.015]),
        meets=lambda actions: all(action.slice_thickness < 0.015 for action in actions),
    ),
    ResolutionCase(
        "C5",
        "SLICE or HALVING, and LEFT or RIGHT",
        lambda q: q.where(
            q.cut(),
            lambda v: [
                in_(v.technique, [CuttingTechnique.SLICE, CuttingTechnique.HALVING]),
                not_(v.arm == Arms.BOTH),
            ],
        ),
        meets=lambda actions: all(
            action.technique in (CuttingTechnique.SLICE, CuttingTechnique.HALVING)
            and action.arm != Arms.BOTH
            for action in actions
        ),
    ),
    ResolutionCase(
        "P4",
        "a tilt angle between 1.0 and 1.5",
        lambda q: q.where(
            q.pour(), lambda v: [and_(v.tilt_angle >= 1.0, v.tilt_angle <= 1.5)]
        ),
        meets=lambda actions: all(
            1.0 <= action.tilt_angle <= 1.5 for action in actions
        ),
    ),
    ResolutionCase(
        "P5",
        "the LEFT arm, or a tilt angle above 2.0",
        lambda q: q.where(
            q.pour(), lambda v: [or_(v.arm == Arms.LEFT, v.tilt_angle > 2.0)]
        ),
        meets=lambda actions: all(
            action.arm == Arms.LEFT or action.tilt_angle > 2.0 for action in actions
        ),
    ),
    ResolutionCase(
        "C6",
        "the bread, found by a search on its name",
        lambda q: q.where(
            q.cut(), lambda v: [contains(v.object_to_cut.name.name, SceneBody.BREAD)]
        ),
        meets=cuts_the_bread_with_the_knife,
    ),
    ResolutionCase(
        "C7",
        "the body of some Food, sensibly the bread",
        lambda q: q.where(
            q.cut(),
            lambda v: [exists(food := variable(Food), food.root == v.object_to_cut)],
        ),
        meets=cuts_the_bread_with_the_knife,
    ),
    ResolutionCase(
        "P6",
        "the bowl, never the cup's own body",
        lambda q: q.where(
            q.pour(), lambda v: [v.target_container != v.source_container.root]
        ),
        meets=pours_from_the_cup_into_the_bowl,
    ),
    ResolutionCase(
        "P7",
        "the bowl, with the given LEFT arm",
        lambda q: q.where(
            q.pour(arm=Arms.LEFT),
            lambda v: [contains(v.target_container.name.name, SceneBody.BOWL)],
        ),
        meets=pours_from_the_cup_into_the_bowl,
    ),
    ResolutionCase(
        "C8",
        "consistent values across all six open slots",
        lambda q: q.cut(number_of_cuts_on_local_x_axis=..., slicing_priority=...),
        meets=cuts_the_bread_with_the_knife,
    ),
    ResolutionCase(
        "P8",
        "consistent pouring geometry, the pour side agreeing with the arm",
        lambda q: q.pour(
            pour_side=...,
            pour_side_offset=...,
            pour_approach_offset=...,
            pour_height=...,
        ),
        meets=pours_from_the_cup_into_the_bowl,
    ),
    ResolutionCase(
        "C9",
        "up to three genuinely different cutting actions",
        lambda q: q.several(CuttingAction, q.cutting_slots()),
        meets=lambda actions: 1 <= len(actions) <= SEVERAL
        and cuts_the_bread_with_the_knife(actions),
    ),
    ResolutionCase(
        "P9",
        "up to three different pouring actions, the bowl first",
        lambda q: q.several(PouringAction, q.pouring_slots()),
        meets=lambda actions: 1 <= len(actions) <= SEVERAL
        and pours_from_the_cup_into_the_bowl(actions[:1]),
    ),
    ResolutionCase(
        "C10",
        "with the knife excluded, the cup is the only tool left and is taken",
        lambda q: q.where(
            q.cut(), lambda v: [not_(v.tool.root.name.name == SceneBody.KNIFE)]
        ),
        meets=lambda actions: all(
            isinstance(action.tool, PouringCup) for action in actions
        ),
    ),
    ResolutionCase(
        "P10",
        "only the knife may be the source, and it is taken",
        lambda q: q.where(
            q.pour(),
            lambda v: [contains(v.source_container.root.name.name, SceneBody.KNIFE)],
        ),
        meets=lambda actions: all(
            isinstance(action.source_container, CuttingKnife) for action in actions
        ),
    ),
    ResolutionCase(
        "C11",
        "a refusal: neither the bowl nor the spoon is something to cut",
        lambda q: q.cut(
            object_to_cut=variable(
                Body, domain=[q.body(SceneBody.BOWL), q.body(SceneBody.SPOON)]
            )
        ),
        refusal=NoFittingReferent,
    ),
    ResolutionCase(
        "C12",
        "a refusal: a thickness below 0.005 and above 0.01 cannot both hold",
        lambda q: q.where(
            q.cut(), lambda v: [v.slice_thickness < 0.005, v.slice_thickness > 0.01]
        ),
        refusal=UnsatisfiableRequest,
    ),
    ResolutionCase(
        "T1",
        "halving the bread",
        lambda q: q.cut(),
        meets=lambda actions: cuts_the_bread_with_the_knife(actions)
        and all(action.technique == CuttingTechnique.HALVING for action in actions),
        task="Cut the bread in half.",
    ),
    ResolutionCase(
        "T2",
        "the condition beats the task: a thickness below 0.015",
        lambda q: q.where(q.cut(), lambda v: [v.slice_thickness < 0.015]),
        meets=lambda actions: all(action.slice_thickness < 0.015 for action in actions),
        task="Cut thick slices for toast.",
    ),
    ResolutionCase(
        "T3",
        "the left arm",
        lambda q: q.pour(),
        meets=lambda actions: pours_from_the_cup_into_the_bowl(actions)
        and all(action.arm == Arms.LEFT for action in actions),
        task="Pour from the cup into the bowl slowly with the left arm.",
    ),
    ResolutionCase(
        "T4",
        "the milk is no tool, so the cup is the source",
        lambda q: q.pour(),
        meets=pours_from_the_cup_into_the_bowl,
        task="Pour the milk into the bowl.",
    ),
    ResolutionCase(
        "T5",
        "slicing variants of the bread",
        lambda q: q.several(CuttingAction, q.cutting_slots()),
        meets=cuts_the_bread_with_the_knife,
        task="Prepare the bread for sandwiches.",
    ),
    ResolutionCase(
        "T6",
        "the sink's body, among the furniture",
        lambda q: q.pour(),
        meets=lambda actions: all(
            name_of(action.target_container) == SceneBody.SINK for action in actions
        ),
        task="Empty the cup into the sink.",
        furnished=True,
    ),
)
"""
The resolver's cases, from almost given to refusals, the last ones with a task.
"""

KNOWLEDGE_CASES = (
    KnowledgeCase("K1", "", ()),
    KnowledgeCase(
        "K2",
        "Pour the milk from the cup into the bowl.",
        (action_core.MaterialRegime.LIQUID,),
    ),
    KnowledgeCase(
        "K3",
        "Pour the cereal from the cup into the bowl.",
        (action_core.MaterialRegime.GRANULAR,),
    ),
    KnowledgeCase(
        "K4",
        "Pour milk from the cup into the bowl until the bowl is half full.",
        (
            action_core.MaterialRegime.LIQUID,
            action_core.TransferTerminationReason.TARGET_QUANTITY_ATTAINED,
        ),
    ),
    KnowledgeCase(
        "K5",
        "Pour all the cereal from the cup into the bowl.",
        (
            action_core.MaterialRegime.GRANULAR,
            action_core.TransferTerminationReason.SOURCE_EXHAUSTED,
        ),
    ),
    KnowledgeCase(
        "K8",
        "Pour the milk from the cup into the bowl until I tell you to stop.",
        (
            action_core.MaterialRegime.LIQUID,
            action_core.TransferTerminationReason.INSTRUCTED_STOP,
        ),
    ),
)
"""
The knowledge request's cases: pouring from the cup into the bowl in different
situations, each expected to consult knowledge for the situation its task states.
"""
