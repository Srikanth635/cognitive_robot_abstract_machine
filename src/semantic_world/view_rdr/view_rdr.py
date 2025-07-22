from typing_extensions import Any, Dict
from ripple_down_rules.datastructures.case import Case, create_case
from semantic_world.views.views import Fridge
from ripple_down_rules.helpers import general_rdr_classify
from . import view_possible_locations_mcrdr as possible_locations_classifier

name = 'possible_locations'
case_type = Fridge
case_name = 'Fridge'
classifiers_dict = dict()
classifiers_dict['possible_locations'] = possible_locations_classifier


def classify(case: Fridge, **kwargs) -> Dict[str, Any]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    return general_rdr_classify(classifiers_dict, case, **kwargs)
