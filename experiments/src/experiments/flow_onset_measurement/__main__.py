"""
``python -m experiments.flow_onset_measurement MESH --fills F...

--store DIR``: measure in Isaac Sim, headless on the GPU, the tilt at which a liquid filling the container of the
mesh to each fill starts to leave it, and keep each as a physical measurement in the
store.
"""

from __future__ import annotations

import argparse
import logging
from importlib.metadata import version
from pathlib import Path

import numpy as np
import trimesh
from physics_simulators.isaac_application import IsaacApplication
from typing_extensions import Sequence

# %% Isaac Sim's application, which runs before any of its modules is imported

IsaacApplication(headless=True).start()

from cram_viz.resolution.flow_onset import ContainerShapes
from llm_reasoner.knowledge.access import MeasurementStore

from experiments.flow_onset_measurement.filling import ParticleFilling
from experiments.flow_onset_measurement.runs import (
    ParticlesLeaving,
    SimulatedFlowOnset,
    SimulatedTilt,
    SimulationSettings,
)
from experiments.flow_onset_measurement.stage import ParticleOffsets, TiltedContainer

logger = logging.getLogger(__spec__.name)
"""
This module's logger, named for the module even when ``python -m`` runs it as
``__main__``.
"""

# %% one measurement


def measured(
    mesh_file: Path,
    fill_fraction: float,
    gravity_azimuth: float,
    run_count: int,
    settings: SimulationSettings,
    calibrations: int = 1,
) -> SimulatedFlowOnset:
    """
    :param calibrations: How often the particles are settled first to find how full
        they must start to settle to the fill.
    :return: The tilts of the container of the mesh, filled to the share of its basin
        and tilted toward the side, simulated the number of times.
    """
    shapes = ContainerShapes()
    basin = shapes.basin(mesh_file)
    filling = ParticleFilling(
        basin=basin,
        spacing=settings.particle_spacing,
        wall_clearance=ParticleOffsets(
            spacing=settings.particle_spacing
        ).solid_rest_offset,
    )
    leaving = ParticlesLeaving(
        hull=basin.mesh.convex_hull, particle_count=settings.onset_particle_count
    )
    simulated = SimulatedFlowOnset(
        container=shapes.container_model(mesh_file),
        gravity_azimuth=gravity_azimuth,
        settings=settings,
    )
    start_fill = fill_fraction
    for _ in range(calibrations):
        settled = settled_particles(basin.mesh, filling, start_fill, settings)
        held = basin.mesh.convex_hull.contains(settled)
        start_fill = filling.start_fill_settling_at(
            fill_fraction, start_fill, filling.settled_fill_fraction(settled[held])
        )
    for run in range(run_count):
        tilted = TiltedContainer(
            mesh=basin.mesh,
            particles=filling.positions(start_fill),
            gravity_azimuth=gravity_azimuth,
            settings=settings,
        )
        tilted.build()
        settled = tilted.settle()
        held = basin.mesh.convex_hull.contains(settled)
        onset = tilted.onset(held, leaving)
        settled_fill = filling.settled_fill_fraction(settled[held])
        logger.info(
            "run %d of %s at fill %.2f: settled to %.3f, onset %s degrees",
            run + 1,
            mesh_file.name,
            fill_fraction,
            settled_fill,
            onset,
        )
        if onset is not None:
            simulated.runs.append(
                SimulatedTilt(onset_tilt=onset, settled_fill_fraction=settled_fill)
            )
    return simulated


def settled_particles(
    mesh: trimesh.Trimesh,
    filling: ParticleFilling,
    fill_fraction: float,
    settings: SimulationSettings,
) -> np.ndarray:
    """
    :return: Where the particles filling the container to the share of its basin end
        once settled upright, in its own frame.
    """
    tilted = TiltedContainer(
        mesh=mesh,
        particles=filling.positions(fill_fraction),
        gravity_azimuth=0.0,
        settings=settings,
    )
    tilted.build()
    return tilted.settle()


# %% the command line


def main(arguments: Sequence[str] | None = None) -> None:
    """
    Measure the flow onsets the command line asks for and store them.
    """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--fills", type=float, nargs="+", required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tilt-rate", type=float, default=5.0)
    parser.add_argument("--spacing", type=float, default=0.002)
    parsed = parser.parse_args(arguments)
    settings = SimulationSettings(
        engine_version=version("isaacsim"),
        tilt_rate=parsed.tilt_rate,
        particle_spacing=parsed.spacing,
    )
    store = MeasurementStore(directory=parsed.store)
    for fill_fraction in parsed.fills:
        simulated = measured(
            parsed.mesh, fill_fraction, parsed.azimuth, parsed.runs, settings
        )
        logger.info("stored %s", store.put(simulated.measurement()))


if __name__ == "__main__":
    main()
