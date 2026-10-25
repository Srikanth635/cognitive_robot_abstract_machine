from __future__ import annotations

from dataclasses import dataclass, field

from krrood.exceptions import DataclassException

from semantic_digital_twin.exceptions import ParsingError

# %% parsing and writing USD stages


@dataclass
class UnsupportedUsdPhysicsJointTypeError(ParsingError):
    """
    Raised when a stage contains a physics joint of a type this parser does not build a
    Connection for.

    Silently skipping it would build a World missing whatever link that joint's body1 is
    the only connection to.
    """

    joint_path: str = field(kw_only=True)
    """
    The unsupported joint prim's stage path.
    """

    joint_type: str = field(kw_only=True)
    """
    The unsupported joint prim's USD type name.
    """

    supported_types: list[str] = field(kw_only=True, default_factory=list)
    """
    The joint types that can be mapped to connections.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has a joint of unsupported type "
            f"'{self.joint_type}' at '{self.joint_path}'."
        )

    def suggest_correction(self) -> str:
        if not self.supported_types:
            return ""
        return f"Use one of the supported types: {', '.join(sorted(self.supported_types))}."


@dataclass
class UsdPhysicsJointMissingChildBodyError(ParsingError):
    """
    Raised when a physics joint's ``body1`` relationship (its child link) has no target.

    Unlike ``body0``, where an unset target is the USD convention for "the stage's own
    root frame", ``body1`` has no such meaning here - every joint is expected to connect
    a link into the world, so a joint with no child leaves nothing to build a Connection
    to.
    """

    joint_path: str = field(kw_only=True)
    """
    The joint prim's stage path.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has a joint at '{self.joint_path}' with no "
            f"body1 target."
        )

    def suggest_correction(self) -> str:
        return "Inspect the joint prim's body1 relationship on the stage."


@dataclass
class UnsupportedUsdGeometryTypeError(ParsingError):
    """
    Raised when a stage contains a renderable geometric primitive
    (:class:`~pxr.UsdGeom.Gprim`) of a type this parser does not build a Shape for.

    Silently skipping it would build a World missing that piece of a link's geometry,
    the same way silently skipping an unsupported joint would build one missing a link.
    """

    prim_path: str = field(kw_only=True)
    """
    The unsupported geometry prim's stage path.
    """

    geometry_type: str = field(kw_only=True)
    """
    The unsupported geometry prim's USD type name.
    """

    supported_types: list[str] = field(kw_only=True, default_factory=list)
    """
    The geometry types that can be mapped to shapes.
    """

    def error_message(self) -> str:
        return (
            f"Stage '{self.file_path}' has geometry of unsupported type "
            f"'{self.geometry_type}' at '{self.prim_path}'."
        )

    def suggest_correction(self) -> str:
        if not self.supported_types:
            return ""
        return f"Use one of the supported types: {', '.join(sorted(self.supported_types))}."


@dataclass
class UnsupportedUsdConnectionError(DataclassException):
    """
    Raised when a world holds a connection of a kind the USD builder does not write a
    joint for.

    Leaving it out would let the body it connects fall or float free under physics.
    """

    connection_name: str
    """
    The name of the unsupported connection.
    """

    connection_type: str
    """
    The name of the unsupported connection's class.
    """

    def error_message(self) -> str:
        return (
            f"The connection '{self.connection_name}' is a {self.connection_type}, "
            f"which the USD builder does not write a joint for."
        )

    def suggest_correction(self) -> str:
        return "Build the world for a simulator that supports this kind of connection."


@dataclass
class UnsupportedFollowingModeError(DataclassException):
    """
    Raised when a world holds a camera or light that follows its body in a way other
    than fixed to it, which the USD builder does not write.
    """

    element_name: str
    """
    The name of the camera or light.
    """

    mode: str
    """
    How the camera or light follows its body, by MuJoCo's name for it.
    """

    def error_message(self) -> str:
        return (
            f"'{self.element_name}' follows its body as {self.mode}; the USD builder "
            f"writes cameras and lights fixed to their body only."
        )

    def suggest_correction(self) -> str:
        return "Fix the camera or light to its body, or build the world for MuJoCo."


@dataclass
class UnplacedSimulatorElementError(DataclassException):
    """
    Raised when a world holds a camera or light of a kind that does not say where it is
    or how it looks, which the USD builder therefore cannot write.
    """

    element_name: str
    """
    The name of the camera or light.
    """

    element_type: str
    """
    The name of its class.
    """

    def error_message(self) -> str:
        return (
            f"'{self.element_name}' is a {self.element_type}, which says nothing of "
            f"where it is or how it looks."
        )

    def suggest_correction(self) -> str:
        return "Describe it as a MujocoCamera or a MujocoLight."
