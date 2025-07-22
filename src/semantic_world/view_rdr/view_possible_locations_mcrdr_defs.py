from typing_extensions import List, Set, Union
from semantic_world.world_entity import Body, EnvironmentView, View
from ripple_down_rules.datastructures.tracked_object import TrackedObjectMixin, QueryObject
from ripple_down_rules import *


def conditions_169119401358620755610132125806000007134(case) -> bool:
    def conditions_for_view_possible_locations_of_type_view(case: View) -> bool:
        """Get conditions on whether it's possible to conclude a value for View.possible_locations  of type View."""
        return any(has(type(case), Body, recursive=True))
    return conditions_for_view_possible_locations_of_type_view(case)


def conclusion_169119401358620755610132125806000007134(case) -> List[Union[type, View]]:
    def view_possible_locations_of_type_view(case: View) -> List[Union[type, View]]:
        """Get possible value(s) for View.possible_locations  of type View."""
        return [res[0] for res in has(QueryObject, type(case), recursive=True) if isA(res[0], EnvironmentView)]
    return view_possible_locations_of_type_view(case)


