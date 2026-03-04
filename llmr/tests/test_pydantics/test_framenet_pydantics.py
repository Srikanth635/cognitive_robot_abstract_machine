"""Tests for FrameNet Pydantic models."""

import pytest
from pydantic import ValidationError

from llmr.workflows.pydantics.framenet_pydantics import (
    CoreElements,
    FrameNetRepresentation,
    PeripheralElements,
)


class TestFrameNetRepresentation:
    def _valid_payload(self) -> dict:
        return {
            "framenet": "picking_up_object",
            "frame": "Getting",
            "lexical-unit": "pick_up.v",
            "core": {},
            "peripheral": {},
        }

    def test_valid_construction(self) -> None:
        fn = FrameNetRepresentation(**self._valid_payload())
        assert fn.frame == "Getting"
        assert fn.lexical_unit == "pick_up.v"

    def test_invalid_lexical_unit_without_dot(self) -> None:
        payload = self._valid_payload()
        payload["lexical-unit"] = "pickup"
        with pytest.raises(ValidationError):
            FrameNetRepresentation(**payload)

    def test_framenet_field_stored(self) -> None:
        fn = FrameNetRepresentation(**self._valid_payload())
        assert fn.framenet == "picking_up_object"

    def test_core_assigned(self) -> None:
        fn = FrameNetRepresentation(**self._valid_payload())
        assert isinstance(fn.core, CoreElements)

    def test_peripheral_assigned(self) -> None:
        fn = FrameNetRepresentation(**self._valid_payload())
        assert isinstance(fn.peripheral, PeripheralElements)


class TestCoreElements:
    def test_all_optional(self) -> None:
        core = CoreElements()
        assert core.agent is None
        assert core.patient is None

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            CoreElements(unknown_field="value")

    def test_partial_construction(self) -> None:
        core = CoreElements(agent="robot", patient="apple")
        assert core.agent == "robot"
        assert core.instrument is None


class TestPeripheralElements:
    def test_all_optional(self) -> None:
        peri = PeripheralElements()
        assert peri.location is None
        assert peri.manner is None

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            PeripheralElements(bad_field="x")

    def test_partial_construction(self) -> None:
        peri = PeripheralElements(location="kitchen", manner="carefully")
        assert peri.location == "kitchen"
        assert peri.speed is None
