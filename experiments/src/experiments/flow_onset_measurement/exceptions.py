"""
What measuring a flow onset in the simulation refuses.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass
class NoSimulatedTilt(DataclassException):
    """
    A measurement is asked for from runs of which there are none.
    """

    mesh_file_name: str
    """
    The mesh file of the container the measurement would be about.
    """

    def error_message(self) -> str:
        return f"no tilt of {self.mesh_file_name} was simulated"

    def suggest_correction(self) -> str:
        return "simulate the tilt at least once before keeping its measurement"
