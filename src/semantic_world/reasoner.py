from __future__ import annotations

from dataclasses import dataclass, field
from os.path import dirname

from ripple_down_rules.helpers import general_rdr_classify
from ripple_down_rules.utils import update_case
from typing_extensions import ClassVar, List, Dict, Any, TYPE_CHECKING, Optional, Callable, Type

from ripple_down_rules import GeneralRDR, CaseQuery
from semantic_world.world_entity import View

if TYPE_CHECKING:
    from semantic_world.world import World


@dataclass
class CaseReasoner:
    """
    The case reasoner is a class that uses Ripple Down Rules to reason on case related concepts.
    The reasoner can be used in two ways:
    1. Classification mode, where the reasoner infers all concepts that it has rules for at that time.
    >>> reasoner = CaseReasoner(case)
    >>> inferred_concepts = reasoner.reason()
    >>> inferred_attribute_values = inferred_concepts['attribute_name']
    2. Fitting mode, where the reasoner prompts the expert for answers given a query on a world concept. This allows
    incremental knowledge gain, improved reasoning capabilities, and an increased breadth of application with more
     usage.
     >>> reasoner = CaseReasoner(case)
     >>> reasoner.fit_attribute("attribute_name", [attribute_types,...], False)
    """
    case: Any
    """
    The case instance on which the reasoning is performed.
    """
    result: Optional[Dict[str, Any]] = field(init=False, default=None)
    """
    The latest result of the :py:meth:`reason` call.
    """
    rdr: ClassVar[GeneralRDR] = field(init=False, default=None)
    """
    This is a ripple down rules reasoner that infers concepts relevant to the view like finding out the components
    that exist in the current view.
    """

    def __post_init__(self):
        if self.__class__.rdr is None:
            self.__class__.rdr = GeneralRDR(save_dir=dirname(__file__),
                                            model_name=f"{type(self.case).__name__.lower()}_rdr")

    def reason(self) -> Dict[str, Any]:
        """
        Perform rule-based reasoning on the current view and infer all possible concepts.

        :return: The inferred concepts as a dictionary mapping concept name to all inferred values of that concept.
        """
        self.result = self.rdr.classify(self.case, modify_case=True)
        return self.result

    def fit_attribute(self, attribute_name: str, attribute_types: List[Type[Any]],
                      mutually_exclusive: bool,
                      update_existing_rules: bool = False,
                      case_factory: Optional[Callable] = None,
                      scenario: Optional[Callable] = None) -> None:
        """
        Fit the view RDR to the required attribute types.

        :param attribute_name: The attribute name that the RDR should be fitted to.
        :param attribute_types: A list of attribute types that the RDR should be fitted to.
        :param mutually_exclusive: whether the attribute values are mutually exclusive or not.
        :param update_existing_rules: If True, existing rules of the given types will be updated with new rules,
         else they will be skipped.
        :param case_factory: Optional callable that can be used to recreate the case object.
        :param scenario: Optional callable that represents the test method or scenario that is being executed.
        """
        case_query = CaseQuery(self.case, attribute_name, tuple(attribute_types), mutually_exclusive,
                               case_factory=case_factory, scenario=scenario)
        self.rdr.fit_case(case_query, update_existing_rules=update_existing_rules)


@dataclass
class WorldReasoner(CaseReasoner):
    """
    The world reasoner is a class that uses Ripple Down Rules to reason on world related concepts like world views.
    The reasoner can be used in two ways:
    1. Classification mode, where the reasoner infers all concepts that it has rules for at that time.
    >>> reasoner = WorldReasoner(world)
    >>> inferred_concepts = reasoner.reason()
    >>> inferred_views = inferred_concepts['views']
    2. Fitting mode, where the reasoner prompts the expert for answers given a query on a world concept. This allows
    incremental knowledge gain, improved reasoning capabilities, and an increased breadth of application with more
     usage.
     >>> reasoner = WorldReasoner(world)
     >>> reasoner.fit_attribute("views", [Handle, Drawer], False)
    """
    _last_world_model_version: Optional[int] = field(init=False, default=None)
    """
    The last world model version of the world used when :py:meth:`reason` 
    was last called.
    """

    def infer_views(self) -> List[View]:
        """
        Infer the views of the world by calling the :py:meth:`reason` method and extracting all inferred views.

        :return: The inferred views of the world.
        """
        result = self.reason()
        if 'views' in result:
            views = result['views']
        else:
            views = []
        return views

    def reason(self) -> Dict[str, Any]:
        """
        Perform rule-based reasoning on the current world and infer all possible concepts.

        :return: The inferred concepts as a dictionary mapping concept name to all inferred values of that concept.
        """
        if self.case._model_version != self._last_world_model_version:
            self.result = self.rdr.classify(self.case)
            self._update_world_attributes()
            self._last_world_model_version = self.case._model_version
        return self.result

    def _update_world_attributes(self):
        """
        Update the world attributes from the values in the result of the latest :py:meth:`reason` call.
        """
        for attr_name, attr_value in self.result.items():
            if isinstance(getattr(self.case, attr_name), list):
                attr_value = list(attr_value)
            if attr_name == 'views':
                for view in attr_value:
                    self.case.add_view(view, exists_ok=True)
            else:
                setattr(self.case, attr_name, attr_value)

    def fit_views(self, required_views: List[Type[View]],
                  update_existing_views: bool = False,
                  world_factory: Optional[Callable] = None,
                  scenario: Optional[Callable] = None) -> None:
        """
        Fit the world RDR to the required view types.

        :param required_views: A list of view types that the RDR should be fitted to.
        :param update_existing_views: If True, existing views will be updated with new rules, else they will be skipped.
        :param world_factory: Optional callable that can be used to recreate the world object.
        :param scenario: Optional callable that represents the test method or scenario that is being executed.
        """
        self.fit_attribute("views", required_views, False,
                           update_existing_rules=update_existing_views,
                           case_factory=world_factory, scenario=scenario)
