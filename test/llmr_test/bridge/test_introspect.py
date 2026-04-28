"""Tests for :mod:`llmr.bridge.introspect` — action dataclass field classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing_extensions import List, Optional, Type

import pytest
from krrood.symbol_graph.symbol_graph import Symbol

from llmr.bridge.introspect import (
    _MISSING_DEFAULT,
    ActionSpec,
    FieldKind,
    DiscoveredField,
    DeclaredFieldsIntrospector,
    ActionFieldIntrospector,
    introspect_action,
)

from .._fixtures.actions import (
    GraspType,
    MockContainerAction,
    MockGraspDescription,
    MockNavigateAction,
    MockPickUpAction,
    MockPose,
    MockPoseAction,
    MockRequiredManipulatorAction,
    MockTypeRefAction,
)
from .._fixtures.symbols import Manipulator, ParallelGripperLike


class TestIntrospectSchema:
    """:meth:`ActionFieldIntrospector.introspect_action` returns a fully populated :class:`ActionSpec`."""

    def test_action_type_name_is_class_name(
        self, introspector: ActionFieldIntrospector
    ) -> None:
        schema = introspector.introspect_action(MockPickUpAction)
        assert schema.action_type == "MockPickUpAction"
        assert schema.action_cls is MockPickUpAction

    def test_class_docstring_is_captured(
        self, introspector: ActionFieldIntrospector
    ) -> None:
        schema = introspector.introspect_action(MockPickUpAction)
        assert schema.docstring == "Minimal stand-in for PyCRAM PickUpAction."

    def test_own_fields_extracted(self, introspector: ActionFieldIntrospector) -> None:
        schema = introspector.introspect_action(MockPickUpAction)
        names = {f.name for f in schema.fields}
        assert names == {"object_designator", "grasp_description", "timeout"}

    def test_non_dataclass_raises(self, introspector: ActionFieldIntrospector) -> None:
        class NotADataclass:
            pass

        with pytest.raises(TypeError):
            introspector.introspect_action(NotADataclass)

    def test_entity_field_spec(self, introspector: ActionFieldIntrospector) -> None:
        schema = introspector.introspect_action(MockPickUpAction)
        field = next(f for f in schema.fields if f.name == "object_designator")
        assert field.kind == FieldKind.ENTITY
        assert field.is_optional is False
        assert field.default is _MISSING_DEFAULT
        assert field.docstring == "The object to pick up."
        assert field.raw_type is Symbol

    def test_optional_primitive_field_spec(
        self, introspector: ActionFieldIntrospector
    ) -> None:
        schema = introspector.introspect_action(MockPickUpAction)
        field = next(f for f in schema.fields if f.name == "timeout")
        assert field.kind == FieldKind.PRIMITIVE
        assert field.is_optional is True

    def test_complex_field_expands_sub_fields(
        self, introspector: ActionFieldIntrospector
    ) -> None:
        schema = introspector.introspect_action(MockPickUpAction)
        field = next(f for f in schema.fields if f.name == "grasp_description")
        assert field.kind == FieldKind.COMPLEX
        sub_names = {sf.name for sf in field.sub_fields}
        assert sub_names == {"grasp_type", "manipulator"}

    def test_enum_field_lists_members(self, introspector: ActionFieldIntrospector) -> None:
        schema = introspector.introspect_action(MockGraspDescription)
        field = next(f for f in schema.fields if f.name == "grasp_type")
        assert field.kind == FieldKind.ENUM
        assert set(field.enum_members) == {"FRONT", "TOP", "SIDE"}

    def test_type_ref_keeps_inner_type(self, introspector: ActionFieldIntrospector) -> None:
        schema = introspector.introspect_action(MockTypeRefAction)
        field = next(f for f in schema.fields if f.name == "annotation_type")
        assert field.kind == FieldKind.TYPE_REF
        assert field.raw_type is Symbol

    def test_required_symbol_subclass_entity(
        self, introspector: ActionFieldIntrospector
    ) -> None:
        schema = introspector.introspect_action(MockRequiredManipulatorAction)
        field = next(f for f in schema.fields if f.name == "manipulator")
        assert field.kind == FieldKind.ENTITY
        assert field.is_optional is False

    def test_pose_field_classified_via_mro_name(
        self, introspector: ActionFieldIntrospector
    ) -> None:
        """``target_pose: MockPose`` hits the POSE branch via class-name MRO match."""
        schema = introspector.introspect_action(MockPoseAction)
        field = next(f for f in schema.fields if f.name == "target_pose")
        assert field.kind == FieldKind.POSE

    def test_container_field_falls_back_to_primitive(
        self, introspector: ActionFieldIntrospector
    ) -> None:
        schema = introspector.introspect_action(MockContainerAction)
        field = next(f for f in schema.fields if f.name == "object_designators")
        assert field.kind == FieldKind.PRIMITIVE


class TestClassifyType:
    """Direct :meth:`classify_type` coverage for each :class:`FieldKind` branch."""

    @pytest.mark.parametrize("primitive", [bool, int, float, str, bytes])
    def test_scalars_are_primitive(
        self, introspector: ActionFieldIntrospector, primitive: type
    ) -> None:
        assert introspector.classify_type(primitive) is FieldKind.PRIMITIVE

    def test_none_is_primitive(self, introspector: ActionFieldIntrospector) -> None:
        assert introspector.classify_type(None) is FieldKind.PRIMITIVE
        assert introspector.classify_type(type(None)) is FieldKind.PRIMITIVE

    def test_non_type_is_primitive(self, introspector: ActionFieldIntrospector) -> None:
        """Non-type inputs (e.g. string annotations) fall through to PRIMITIVE."""
        assert introspector.classify_type("not-a-type") is FieldKind.PRIMITIVE

    def test_enum_subclass(self, introspector: ActionFieldIntrospector) -> None:
        assert introspector.classify_type(GraspType) is FieldKind.ENUM

    def test_symbol_subclass_is_entity(self, introspector: ActionFieldIntrospector) -> None:
        assert introspector.classify_type(Symbol) is FieldKind.ENTITY
        assert introspector.classify_type(Manipulator) is FieldKind.ENTITY
        assert introspector.classify_type(ParallelGripperLike) is FieldKind.ENTITY

    def test_dataclass_is_complex(self, introspector: ActionFieldIntrospector) -> None:
        assert introspector.classify_type(MockGraspDescription) is FieldKind.COMPLEX

    def test_type_annotation_is_type_ref(
        self, introspector: ActionFieldIntrospector
    ) -> None:
        assert introspector.classify_type(Type[Symbol]) is FieldKind.TYPE_REF

    def test_pose_name_match(self, introspector: ActionFieldIntrospector) -> None:
        """Any class whose MRO contains a name in POSE_TYPE_NAMES is classified POSE."""

        class Pose:
            pass

        class CustomPose(Pose):
            pass

        assert introspector.classify_type(Pose) is FieldKind.POSE
        assert introspector.classify_type(CustomPose) is FieldKind.POSE


class TestDocstringExtraction:
    """AST-based field docstring extraction covers multiple layouts."""

    def test_attribute_docstrings_collected(
        self, introspector: ActionFieldIntrospector
    ) -> None:
        schema = introspector.introspect_action(MockPickUpAction)
        docs = {f.name: f.docstring for f in schema.fields}
        assert docs["object_designator"] == "The object to pick up."
        assert docs["timeout"] == "Maximum seconds to attempt the action."

    def test_fields_without_docstring_default_to_empty(
        self, introspector: ActionFieldIntrospector
    ) -> None:
        @dataclass
        class PlainAction:
            value: int

        schema = introspector.introspect_action(PlainAction)
        assert schema.fields[0].docstring == ""

    def test_source_unavailable_returns_empty_docs(self) -> None:
        """When ``inspect.getsource`` fails, docstrings come back empty, not broken."""
        local_cls = type(
            "DynamicAction",
            (object,),
            {"__annotations__": {"x": int}, "__module__": __name__},
        )
        # Force a dataclass with no accessible source (built from type()).
        from dataclasses import dataclass as _dc

        local_cls = _dc(local_cls)
        docs = ActionFieldIntrospector._extract_field_docstrings(local_cls)
        assert docs == {}


class TestOwnDataclassIntrospector:
    """:class:`DeclaredFieldsIntrospector` filters inherited fields."""

    def test_inherited_fields_excluded(self) -> None:
        @dataclass
        class Parent:
            parent_field: int = 0

        @dataclass
        class Child(Parent):
            child_field: int = 0

        introspector = DeclaredFieldsIntrospector()
        names = {attr.public_name for attr in introspector.discover(Child)}
        assert names == {"child_field"}


class TestModuleIntrospectHelper:
    """The module-level ``introspect_action()`` convenience wrapper."""

    def test_returns_action_schema(self) -> None:
        schema = introspect_action(MockNavigateAction)
        assert isinstance(schema, ActionSpec)
        assert schema.action_cls is MockNavigateAction
        assert any(f.name == "target_location" for f in schema.fields)


class TestFieldSpecDefaults:
    """Field defaults and sentinel values."""

    def test_required_field_uses_no_default_sentinel(
        self, introspector: ActionFieldIntrospector
    ) -> None:
        schema = introspector.introspect_action(MockPickUpAction)
        required = next(f for f in schema.fields if f.name == "object_designator")
        assert required.default is _MISSING_DEFAULT

    def test_field_spec_instantiates_with_defaults(self) -> None:
        spec = DiscoveredField(name="x", raw_type=int, kind=FieldKind.PRIMITIVE)
        assert spec.default is _MISSING_DEFAULT
        assert spec.enum_members == []
        assert spec.sub_fields == []
