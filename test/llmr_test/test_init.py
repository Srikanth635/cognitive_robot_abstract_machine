"""Tests for :mod:`llmr` public surface."""

from __future__ import annotations


class TestPackagePublicSurface:
    """``llmr.__all__`` exports its advertised public API."""

    def test_top_level_exports(self) -> None:
        import llmr

        expected = {
            "LLMBackend",
            "plan_from_instruction",
            "sequential_plan_from_instruction",
            "plan_from_match",
            "instance_from_match",
            "LLMActionClassificationFailed",
            "LLMActionRegistryEmpty",
            "LLMProviderNotSupported",
            "LLMSlotFillingFailed",
            "LLMUnresolvedRequiredFields",
        }
        assert expected.issubset(set(llmr.__all__))
        for name in expected:
            assert hasattr(llmr, name), f"llmr missing advertised export {name}"


class TestPycramBridgeSurface:
    """:mod:`llmr.pycram` exposes the PyCRAM adapter surface."""

    def test_adapter_exports(self) -> None:
        from llmr.pycram import (
            PycramContext,
            PycramPlanNode,
            discover_action_classes,
            execute_single,
        )

        assert callable(discover_action_classes)
        assert callable(execute_single)
        assert PycramContext is not None
        assert PycramPlanNode is not None


class TestBridgeSurface:
    """:mod:`llmr.bridge` submodules are importable and expose the documented symbols."""

    def test_introspect_exports(self) -> None:
        from llmr.bridge.introspect import (
            ActionSpec,
            FieldKind,
            DiscoveredField,
            ActionFieldIntrospector,
        )

        assert FieldKind.ENTITY.name == "ENTITY"
        assert DiscoveredField is not None
        assert ActionSpec is not None
        assert ActionFieldIntrospector is not None

    def test_match_reader_exports(self) -> None:
        from llmr.bridge.match_reader import (
            MatchSnapshot,
            MatchField,
            construct_action,
            snapshot_match,
            underspecified_match,
            missing_required_fields,
            bind_slot_value,
        )

        assert MatchSnapshot is not None
        assert MatchField is not None
        for fn in (
            snapshot_match,
            bind_slot_value,
            construct_action,
            underspecified_match,
            missing_required_fields,
        ):
            assert callable(fn)
