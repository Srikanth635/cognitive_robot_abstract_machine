from __future__ import annotations

import json

import pytest
from llm_reasoner.clients.base import ScriptedClient
from llm_reasoner.knowledge.generation.backend import (
    HypothesisGenerator,
    ResponseUnusable,
)
from llm_reasoner.knowledge.generation.schema import derive_schema
from llm_reasoner.knowledge.generation.wire import GenerationWireDecoder
from llm_reasoner.knowledge.microworlds.flanagan.core import MagnitudeRange
from llm_reasoner.knowledge.microworlds.registry import (
    DEFAULT_PROFILE,
    FLANAGAN_CORE,
    ModuleSelection,
)
from llm_reasoner.knowledge.ontological.constraints import ConstraintSuite
from llm_reasoner.knowledge.ontological.decorators import ConceptReference

# %% fixtures

UNDECLARED_TYPE_NAME = "NotAnOntologyType"
"""
A fragment type that no microworld declares.
"""

FALLBACK_TASK = ConceptReference(iri="soma:PhysicalTask", label="PhysicalTask")


def range_schema():
    """
    Return the schema whose root is a single magnitude range.
    """
    return derive_schema(modules=(FLANAGAN_CORE.name,), root=MagnitudeRange.__name__)


def fragment_payload(type_name: str) -> str:
    """
    Return a one-fragment response of the given type.
    """
    return json.dumps(
        {
            "fragments": [
                {
                    "id": "only-fragment",
                    "type": type_name,
                    "lower_bound": 0.2,
                    "upper_bound": 0.5,
                    "unit": "kg",
                }
            ]
        }
    )


# %% unknown fragment types


class TestUnknownFragmentTypeRejection:
    """
    A response naming an undeclared type is a rejection the caller can repair.
    """

    def test_decoder_reports_the_error_type_it_was_given(self) -> None:
        decoder = GenerationWireDecoder(
            schema=range_schema(), error_type=ResponseUnusable
        )

        with pytest.raises(ResponseUnusable, match=UNDECLARED_TYPE_NAME):
            decoder.decode(json.loads(fragment_payload(UNDECLARED_TYPE_NAME)))

    def test_generator_repairs_a_response_with_an_undeclared_type(self) -> None:
        client = ScriptedClient(
            responses=[
                fragment_payload(UNDECLARED_TYPE_NAME),
                fragment_payload(MagnitudeRange.__name__),
            ]
        )
        generator = HypothesisGenerator(
            client=client,
            suite=ConstraintSuite(name="construction", constraints=()),
            root_class=MagnitudeRange.__name__,
            max_repairs=1,
        )
        selection = ModuleSelection(
            task=FALLBACK_TASK,
            profile=DEFAULT_PROFILE,
            modules=(FLANAGAN_CORE,),
        )

        result = generator.generate(FALLBACK_TASK, selection=selection)

        assert result.admitted is True
        assert len(client.calls) == 2
        assert UNDECLARED_TYPE_NAME in client.calls[1][1]
