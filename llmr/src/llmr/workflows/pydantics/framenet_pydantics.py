"""Pydantic models for FrameNet semantic representation."""

from typing_extensions import Optional

from pydantic import BaseModel, Field, field_validator


class CoreElements(BaseModel):
    """Core Frame Elements — conceptually necessary participants."""

    model_config = {"extra": "forbid"}

    agent: Optional[str] = None
    theme: Optional[str] = None
    patient: Optional[str] = None
    instrument: Optional[str] = None
    source: Optional[str] = None
    goal: Optional[str] = None
    result: Optional[str] = None

    other_core_elements: Optional[str] = Field(
        default=None,
        description="Additional core elements as comma-separated key:value pairs "
        "(e.g., 'recipient:user, beneficiary:guest')",
    )


class PeripheralElements(BaseModel):
    """Peripheral Frame Elements — optional circumstantial details."""

    model_config = {"extra": "forbid"}

    location: Optional[str] = None
    manner: Optional[str] = None
    direction: Optional[str] = None
    time: Optional[str] = None
    purpose: Optional[str] = None
    quantity: Optional[str] = None
    portion: Optional[str] = None
    speed: Optional[str] = None
    path: Optional[str] = None

    other_peripheral_elements: Optional[str] = Field(
        default=None,
        description="Additional peripheral elements as comma-separated key:value pairs",
    )


class FrameNetRepresentation(BaseModel):
    """Complete FrameNet-style semantic representation of an instruction."""

    model_config = {
        "extra": "forbid",
        "populate_by_name": True,
    }

    framenet: str = Field(
        description="Snake_case semantic label for the action type "
        "(e.g., picking_up_object, cutting_food)"
    )
    frame: str = Field(
        description="Official FrameNet frame name in CamelCase (e.g., Getting, Placing, Cutting)"
    )
    lexical_unit: str = Field(
        alias="lexical-unit",
        description="Lexical unit that evokes the frame in format: lemma.pos (e.g., pick_up.v, place.v)",
    )
    core: CoreElements = Field(
        description="Core frame elements — conceptually necessary participants"
    )
    peripheral: PeripheralElements = Field(
        description="Peripheral frame elements — optional circumstantial modifiers"
    )

    @field_validator("lexical_unit")
    @classmethod
    def validate_lexical_unit_format(cls, v: str) -> str:
        """Ensure lexical unit follows lemma.pos format."""
        if "." not in v:
            raise ValueError("Lexical unit must follow format: lemma.pos (e.g., cut.v)")
        return v
