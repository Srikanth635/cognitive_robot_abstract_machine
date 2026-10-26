"""
Tests of open value partitions: a class whose members every module may add to, read from
every module, generated as one enumeration in the module declaring it, and offered to a
hypothesis only as far as its root's modules and the shapes of those modules allow.
"""

from __future__ import annotations

import importlib
import json
import sys
from enum import StrEnum
from pathlib import Path
from types import ModuleType

import pytest
from rdflib import OWL, RDF, Graph, URIRef
from typing_extensions import Iterator

from llm_reasoner.knowledge.generation.schema import (
    AnswerKey,
    GenerationSchema,
    TypeScope,
    type_schema_for,
)
from llm_reasoner.knowledge.generation.wire import DecodingResult, FragmentDecoder
from llm_reasoner.knowledge.ontologies import (
    ActionCoreAnnotation,
    ActionPerspectiveFile,
)
from llm_reasoner.knowledge.ontologies.action_perspectives import (
    action_core,
    failures,
)
from llm_reasoner.knowledge.ontologies.predictive_manipulation import core, pouring
from llm_reasoner.knowledge.ontologies.terms import (
    DescriptionEntity,
    OpenValuePartition,
)
from llm_reasoner.knowledge.ontologies.vocabulary import Vocabulary
from llm_reasoner.knowledge.ontology_translation.exceptions import (
    MembersNotStatedDistinct,
    PartitionClosureConflict,
)
from llm_reasoner.knowledge.ontology_translation.reading import OntologyReader
from llm_reasoner.knowledge.ontology_translation.writing import VocabularyWriter

from .open_partition_modules import OpenPartitionModuleFile

# %% cases


class OpenPartitionCase(StrEnum):
    """
    Statements added to one module of the family to exercise the reader's refusals.
    """

    MEMBER_NOT_STATED_DISTINCT = "member_not_stated_distinct.ttl"
    """
    A member no statement of distinctness lists.
    """

    MEMBERS_NOT_STATED_DISTINCT_FROM_IMPORTED = (
        "members_not_stated_distinct_from_imported.ttl"
    )
    """
    Members stated distinct from each other but not from those of the imported hub.
    """

    PARTITION_OPEN_AND_ENUMERATED = "partition_open_and_enumerated.ttl"
    """
    The open value partition, also defined by enumerating its members.
    """

    @property
    def path(self) -> Path:
        """
        The location of the case's Turtle file.
        """
        return Path(__file__).parent / "ontology_translation_cases" / self.value

    @property
    def module_file(self) -> OpenPartitionModuleFile:
        """
        The module the case's statements are added to.
        """
        if self is OpenPartitionCase.PARTITION_OPEN_AND_ENUMERATED:
            return OpenPartitionModuleFile.HUB
        return OpenPartitionModuleFile.SIDEWAYS_CORRECTIONS


# %% fixtures


def reader_with(*cases: OpenPartitionCase) -> OntologyReader:
    """
    :return: A reader over every module of the family, with the cases' statements added
        to the modules they name.
    """
    module_graphs = {
        module_file: Graph().parse(module_file.path, format="turtle")
        for module_file in OpenPartitionModuleFile
    }
    for case in cases:
        module_graphs[case.module_file].parse(case.path, format="turtle")
    return OntologyReader(module_graphs)


@pytest.fixture(scope="module")
def reader() -> OntologyReader:
    """
    A reader over every module of the family.
    """
    return reader_with()


@pytest.fixture(scope="module")
def generated(
    reader: OntologyReader, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[dict[OpenPartitionModuleFile, ModuleType]]:
    """
    The family's vocabulary modules, generated into a scratch package and imported.
    """
    package_name = OpenPartitionModuleFile.HUB.vocabulary_package
    package = tmp_path_factory.mktemp("vocabulary") / package_name
    package.mkdir()
    (package / "__init__.py").write_text("")
    writer = VocabularyWriter(reader)
    for module_file in OpenPartitionModuleFile:
        (package / f"{module_file.vocabulary_module_name}.py").write_text(
            writer.rendered(module_file)
        )
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(package.parent))
        yield {
            module_file: importlib.import_module(
                f"{package_name}.{module_file.vocabulary_module_name}"
            )
            for module_file in OpenPartitionModuleFile
        }
        for name in [name for name in sys.modules if name.startswith(package_name)]:
            del sys.modules[name]


@pytest.fixture(scope="module")
def direction(generated: dict[OpenPartitionModuleFile, ModuleType]) -> type[StrEnum]:
    """
    The enumeration of the open value partition, generated in the hub.
    """
    return generated[OpenPartitionModuleFile.HUB].Direction


def scope_of(
    module_file: OpenPartitionModuleFile,
    generated: dict[OpenPartitionModuleFile, ModuleType],
) -> TypeScope:
    """
    :return: The scope of a hypothesis rooted in a class of the module.
    """
    return TypeScope(
        usable_modules=frozenset(module_file.usable_modules),
        vocabulary=Vocabulary(modules=generated),
    )


def decoded(
    root: type[DescriptionEntity],
    direction_name: str,
    scope: TypeScope,
    generated: dict[OpenPartitionModuleFile, ModuleType],
) -> DecodingResult:
    """
    :return: An answer of one correction of the root's type moving in the named
        direction, decoded against the root's schema in the scope.
    """
    schema = GenerationSchema(
        root=root, types={root.__name__: type_schema_for(root, scope)}
    )
    answer = {
        AnswerKey.FRAGMENTS: [
            {
                AnswerKey.IDENTIFIER: "correction",
                AnswerKey.TYPE: root.__name__,
                "has_direction": direction_name,
            }
        ]
    }
    return FragmentDecoder(schema, vocabulary=Vocabulary(modules=generated)).decode(
        json.dumps(answer)
    )


# %% the marker


class TestMarker:
    def test_the_action_core_declares_the_marker_of_an_open_partition(self):
        graph = Graph().parse(ActionPerspectiveFile.ACTION_CORE.path, format="turtle")
        marker = URIRef(ActionCoreAnnotation.IS_OPEN_VALUE_PARTITION)
        assert (marker, RDF.type, OWL.AnnotationProperty) in graph


# %% members


class TestMembers:
    def test_the_members_are_read_from_every_module(self, reader):
        hub, target, sideways = (module.terms for module in OpenPartitionModuleFile)
        assert set(reader.value_partitions[hub.Direction]) == {
            hub.Increases,
            hub.Decreases,
            target.TowardsTarget,
            target.AwayFromTarget,
            sideways.Sideways,
        }

    def test_the_members_follow_the_modules_then_each_modules_distinctness(
        self, reader
    ):
        hub, target, sideways = (module.terms for module in OpenPartitionModuleFile)
        assert reader.value_partitions[hub.Direction] == [
            hub.Increases,
            hub.Decreases,
            target.TowardsTarget,
            target.AwayFromTarget,
            sideways.Sideways,
        ]

    def test_an_open_partition_is_no_description_class(self, reader):
        hub = OpenPartitionModuleFile.HUB.terms
        assert hub.Direction not in reader.description_classes

    def test_a_closed_partition_keeps_its_enumerated_members(self, reader):
        hub = OpenPartitionModuleFile.HUB.terms
        assert reader.value_partitions[hub.Axis] == [hub.Magnitude, hub.Place]


# %% refusals


class TestRefusals:
    @pytest.mark.parametrize(
        "case",
        [
            OpenPartitionCase.MEMBER_NOT_STATED_DISTINCT,
            OpenPartitionCase.MEMBERS_NOT_STATED_DISTINCT_FROM_IMPORTED,
        ],
    )
    def test_members_a_module_does_not_state_distinct_are_refused(self, case):
        with pytest.raises(MembersNotStatedDistinct):
            reader_with(case).module_declaration(OpenPartitionModuleFile.HUB)

    def test_a_partition_both_open_and_enumerated_is_refused(self):
        reader = reader_with(OpenPartitionCase.PARTITION_OPEN_AND_ENUMERATED)
        with pytest.raises(PartitionClosureConflict):
            reader.module_declaration(OpenPartitionModuleFile.HUB)


# %% the generated enumeration


class TestEnumeration:
    def test_the_enumeration_of_every_member_is_generated_in_the_hub(
        self, reader, generated, direction
    ):
        hub = OpenPartitionModuleFile.HUB.terms
        assert [member.value for member in direction] == [
            str(member) for member in reader.value_partitions[hub.Direction]
        ]
        assert [
            "Direction" in vars(module)
            for module in (
                generated[OpenPartitionModuleFile.TARGET_CORRECTIONS],
                generated[OpenPartitionModuleFile.SIDEWAYS_CORRECTIONS],
            )
        ] == [False, False]

    def test_the_enumeration_is_an_open_value_partition(self, direction):
        assert issubclass(direction, OpenValuePartition)

    def test_each_member_names_the_module_declaring_it(self, direction):
        assert [member.declaring_module for member in direction] == [
            OpenPartitionModuleFile.HUB,
            OpenPartitionModuleFile.HUB,
            OpenPartitionModuleFile.TARGET_CORRECTIONS,
            OpenPartitionModuleFile.TARGET_CORRECTIONS,
            OpenPartitionModuleFile.SIDEWAYS_CORRECTIONS,
        ]

    def test_a_property_of_the_members_is_stated_of_members_of_every_module(
        self, generated, direction
    ):
        axis = generated[OpenPartitionModuleFile.HUB].Axis
        assert [member.changes_along for member in direction] == [
            axis.MAGNITUDE,
            axis.MAGNITUDE,
            axis.PLACE,
            axis.PLACE,
            axis.PLACE,
        ]


# %% the scope of a hypothesis


class TestScope:
    def test_the_schema_offers_only_the_members_of_the_roots_modules(
        self, generated, direction
    ):
        module_file = OpenPartitionModuleFile.TARGET_CORRECTIONS
        correction = generated[module_file].TargetCorrection
        field_schema = type_schema_for(
            correction, scope_of(module_file, generated)
        ).field_named("has_direction")
        assert field_schema.options == (
            direction.INCREASES.name,
            direction.DECREASES.name,
            direction.TOWARDS_TARGET.name,
            direction.AWAY_FROM_TARGET.name,
        )

    def test_a_shape_listing_the_values_narrows_the_members_further(
        self, generated, direction
    ):
        module_file = OpenPartitionModuleFile.SIDEWAYS_CORRECTIONS
        correction = generated[module_file].SidewaysCorrection
        field_schema = type_schema_for(
            correction, scope_of(module_file, generated)
        ).field_named("has_direction")
        assert field_schema.options == (
            direction.DECREASES.name,
            direction.SIDEWAYS.name,
        )

    def test_an_unrestricted_scope_offers_every_member(self, generated, direction):
        correction = generated[
            OpenPartitionModuleFile.TARGET_CORRECTIONS
        ].TargetCorrection
        field_schema = type_schema_for(correction, TypeScope()).field_named(
            "has_direction"
        )
        assert field_schema.options == tuple(member.name for member in direction)

    def test_the_members_of_the_declaring_module_are_offered_beyond_its_scope(
        self, generated, direction
    ):
        module_file = OpenPartitionModuleFile.TARGET_CORRECTIONS
        correction = generated[module_file].TargetCorrection
        scope = TypeScope(
            usable_modules=frozenset({module_file}),
            vocabulary=Vocabulary(modules=generated),
        )
        field_schema = type_schema_for(correction, scope).field_named("has_direction")
        assert field_schema.options == (
            direction.INCREASES.name,
            direction.DECREASES.name,
            direction.TOWARDS_TARGET.name,
            direction.AWAY_FROM_TARGET.name,
        )

    def test_a_pouring_structure_is_offered_every_material_regime_of_the_core(self):
        schema = GenerationSchema.for_root(pouring.PouringPhaseStructure)
        regime = schema.get(pouring.PouringPhaseStructure.__name__).field_named(
            "has_material_regime"
        )
        assert regime.options == tuple(
            member.name for member in action_core.MaterialRegime
        )


# %% decoding


class TestDecoding:
    def test_a_member_of_the_roots_modules_is_read(self, generated, direction):
        module_file = OpenPartitionModuleFile.TARGET_CORRECTIONS
        result = decoded(
            generated[module_file].TargetCorrection,
            direction.TOWARDS_TARGET.name,
            scope_of(module_file, generated),
            generated,
        )
        assert result.hypothesis.root.has_direction is direction.TOWARDS_TARGET

    def test_a_member_of_another_module_is_refused(self, generated, direction):
        module_file = OpenPartitionModuleFile.TARGET_CORRECTIONS
        result = decoded(
            generated[module_file].TargetCorrection,
            direction.SIDEWAYS.name,
            scope_of(module_file, generated),
            generated,
        )
        assert (result.hypothesis, result.partial.root.has_direction) == (None, None)

    def test_a_member_the_shapes_leave_out_is_refused(self, generated, direction):
        module_file = OpenPartitionModuleFile.SIDEWAYS_CORRECTIONS
        result = decoded(
            generated[module_file].SidewaysCorrection,
            direction.INCREASES.name,
            scope_of(module_file, generated),
            generated,
        )
        assert (result.hypothesis, result.partial.root.has_direction) == (None, None)


# %% the shipped lists


class TestShippedLists:
    def test_only_complete_lists_and_lists_sharing_core_members_stay_closed(self):
        closed = {
            definition
            for module in Vocabulary().modules.values()
            for definition in vars(module).values()
            if isinstance(definition, type)
            and issubclass(definition, StrEnum)
            and definition.__module__ == module.__name__
            and not issubclass(definition, OpenValuePartition)
        }
        assert closed == {
            action_core.ConditionPolarity,
            core.ComparisonOutcome,
            core.EvidenceStatus,
            core.ParameterValueForm,
            core.CorrectionDirection,
            core.EstimateUpdateDirection,
            failures.ParameterChangeDirection,
        }
