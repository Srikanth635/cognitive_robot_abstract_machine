from typing_extensions import Optional, Set, Union
from ripple_down_rules.helpers import get_an_updated_case_copy
from semantic_world.views.views import Fridge
from semantic_world.world_entity import View
from ripple_down_rules.datastructures.case import Case, create_case
from ripple_down_rules.utils import make_set
from .view_possible_locations_mcrdr_defs import *


attribute_name = 'possible_locations'
conclusion_type = (set, list, type, View,)
mutually_exclusive = False
name = 'possible_locations'
case_type = Fridge
case_name = 'Fridge'


def classify(case: Fridge, **kwargs) -> Set[Union[type, View]]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_169119401358620755610132125806000007134(case):
        conclusions.update(make_set(conclusion_169119401358620755610132125806000007134(case)))
    return conclusions
