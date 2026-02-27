from dataclasses import dataclass
from typing import List

from pycram.datastructures.dataclasses import Context
from pycram.parameter_inference import InferenceRule, T


@dataclass
class ArmsFitGraspDescriptionRule(InferenceRule):

    def _apply(self, domain: List[T], context: Context) -> List[T]:
        return domain
