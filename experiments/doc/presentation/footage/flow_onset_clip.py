"""
Records clip A: a container of simulated water tilting until the water leaves it.

The stage, the filling and the tilt are the flow-onset harness's own
(:mod:`experiments.flow_onset_measurement`), so the clip shows the measurement as it is
made; this script only adds a camera, lights and frames. Run it with Isaac Sim's Python
(``isaac-cram-env``), the worktree's packages first on ``PYTHONPATH``::

    OMNI_KIT_ACCEPT_EULA=YES python footage/flow_onset_clip.py MESH --fill 0.5 --out clip_a

It writes ``clip_a.mp4`` and ``clip_a.json`` (the onset it saw and the frame it saw it in).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from physics_simulators.isaac_application import IsaacApplication

# %% Isaac Sim's application, which runs before any of its modules is imported

IsaacApplication(headless=True).start()

import imageio.v2 as imageio
import omni.kit.app
import omni.replicator.core as replicator
from isaacsim.core.rendering_manager import RenderingManager
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

from cram_viz.resolution.flow_onset import ContainerShapes
from experiments.flow_onset_measurement.filling import ParticleFilling
from experiments.flow_onset_measurement.runs import ParticlesLeaving, SimulationSettings
from experiments.flow_onset_measurement.stage import ParticleOffsets, TiltedContainer

# %% the camera and the lights


@dataclass
class ClipCamera:
    """
    A camera looking at the container from the side it tilts toward, slightly above the
    rim, with the lights the stage needs to be seen.
    """

    bounds: np.ndarray
    """
    The container's bounding box in its own frame, two corners.
    """

    width: int = 1920
    """
    The frame width in pixels.
    """

    height: int = 1080
    """
    The frame height in pixels.
    """

    path: str = "/World/ClipCamera"
    """
    The camera prim's path on the stage.
    """

    def add(self, stage, tilt_axis: np.ndarray) -> None:
        """
        Add the camera and the lights to the stage.

        :param stage: The stage the container stands on, upright at the origin.
        :param tilt_axis: The axis the container turns about; the camera looks along it
            so the tilt is seen from the side.
        """
        camera = UsdGeom.Camera.Define(stage, self.path)
        camera.CreateFocalLengthAttr().Set(35.0)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        centre = self.bounds.mean(axis=0)
        size = float(np.max(self.bounds[1] - self.bounds[0]))
        distance = 3.4 * size
        look_at = Gf.Vec3d(*centre)
        sideways = np.array([tilt_axis[0], tilt_axis[1], 0.0])
        eye = Gf.Vec3d(
            *(centre + sideways * distance + np.array([0.0, 0.0, 0.45 * size]))
        )
        transform = (
            Gf.Matrix4d().SetLookAt(eye, look_at, Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
        )
        UsdGeom.Xformable(camera).AddTransformOp().Set(transform)
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
        dome.CreateIntensityAttr(1000.0)
        sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
        sun.CreateIntensityAttr(2500.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(40.0, 0.0, 30.0))

    def add_water(
        self,
        stage,
        points_path: str,
        radius: float,
        instancer_path: str = "/World/water",
    ) -> UsdGeom.PointInstancer:
        """
        Add a point instancer drawing a sphere at every particle, and hide the particle
        set's own points.

        :return: The instancer, whose positions are set from the particles each frame.
        """
        instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)
        sphere = UsdGeom.Sphere.Define(
            stage, Sdf.Path(instancer_path).AppendChild("drop")
        )
        sphere.CreateRadiusAttr().Set(radius)
        instancer.CreatePrototypesRel().SetTargets([sphere.GetPath()])
        UsdGeom.Imageable(stage.GetPrimAtPath(points_path)).MakeInvisible()
        return instancer

    def colour(
        self,
        stage,
        prim_path: str,
        rgb: tuple[float, float, float],
        opacity: float = 1.0,
    ) -> None:
        """
        Give a prim a plain material of one colour.
        """
        material_path = Sdf.Path(prim_path).AppendChild("material")
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendChild("shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*rgb)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path)).Bind(material)


# %% the recording


@dataclass
class FrameTaker:
    """
    Takes colour frames of a camera through a replicator render product.
    """

    camera: ClipCamera
    """
    The camera to take frames of.
    """

    warm_up_renders: int = 8
    """
    How many renders the camera gets before its first picture, as the simulator does.
    """

    renders_per_frame: int = 2
    """
    How many renders a picture takes to be drawn.
    """

    def __post_init__(self) -> None:
        product = replicator.create.render_product(
            self.camera.path, (self.camera.width, self.camera.height)
        )
        self.annotator = replicator.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach([product])
        for _ in range(self.warm_up_renders):
            RenderingManager.render()

    def frame(self, attempts: int = 10) -> np.ndarray:
        """
        :return: The camera's picture of the stage as it is now.
        :raises RuntimeError: When the renderer gives no picture within the attempts.
        """
        for attempt in range(attempts):
            replicator.orchestrator.step(
                rt_subframes=self.renders_per_frame,
                delta_time=0.0,
                pause_timeline=False,
            )
            data = np.asarray(self.annotator.get_data())
            if data.ndim == 3 and data.shape[0] == self.camera.height:
                return data[:, :, :3].copy()
            print("no frame yet:", attempt, data.shape, data.dtype, flush=True)
        raise RuntimeError("the renderer produced no frame")


def record(
    mesh_file: Path,
    fill: float,
    azimuth: float,
    out: Path,
    frames_per_second: int,
    speed: float,
) -> None:
    """
    Settle the particles, tilt the container at the harness's rate until the water
    leaves, a little beyond, and write the frames as a video.

    :param speed: How many simulated seconds one second of video shows.
    """
    settings = SimulationSettings(engine_version="clip")
    shapes = ContainerShapes()
    basin = shapes.basin(mesh_file)
    filling = ParticleFilling(
        basin=basin,
        spacing=settings.particle_spacing,
        wall_clearance=ParticleOffsets(
            spacing=settings.particle_spacing
        ).solid_rest_offset,
    )
    # as the harness does, settle once to find how full the particles must start to settle to the fill
    calibration = TiltedContainer(
        mesh=basin.mesh,
        particles=filling.positions(fill),
        gravity_azimuth=azimuth,
        settings=settings,
    )
    calibration.build()
    settled_once = calibration.settle()
    held_once = basin.mesh.convex_hull.contains(settled_once)
    start_fill = filling.start_fill_settling_at(
        fill, fill, filling.settled_fill_fraction(settled_once[held_once])
    )
    container = TiltedContainer(
        mesh=basin.mesh,
        particles=filling.positions(start_fill),
        gravity_azimuth=azimuth,
        settings=settings,
    )
    container.build()
    stage = omni.usd.get_context().get_stage()
    camera = ClipCamera(bounds=np.asarray(basin.mesh.bounds, dtype=float))
    camera.add(stage, container.tilt_axis)
    camera.colour(stage, "/World/container/mesh", (0.85, 0.87, 0.9), opacity=0.35)
    water = camera.add_water(
        stage, "/World/particles", radius=container.offsets.fluid_rest_offset
    )
    camera.colour(stage, "/World/water", (0.18, 0.45, 0.85))
    taker = FrameTaker(camera=camera)

    def take_frame() -> np.ndarray:
        positions = np.asarray(container._points.GetPointsAttr().Get(), dtype=float)
        water.CreatePositionsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*p) for p in positions.tolist()])
        )
        water.CreateProtoIndicesAttr().Set(Vt.IntArray([0] * len(positions)))
        return taker.frame()

    leaving = ParticlesLeaving(
        hull=basin.mesh.convex_hull, particle_count=settings.onset_particle_count
    )

    steps_per_frame = max(1, round(speed / (frames_per_second * settings.time_step)))
    frames: list[np.ndarray] = []
    onset: dict = {}
    # settle upright, then tilt
    settled = container.settle()
    held = basin.mesh.convex_hull.contains(settled)
    tilt = 0.0
    step_index = 0
    extra_steps_after_onset = round(2.5 / settings.time_step)
    remaining_after_onset = None
    while tilt < container.maximum_tilt:
        container.step(tilt)
        step_index += 1
        tilt = step_index * settings.time_step * settings.tilt_rate
        if step_index % steps_per_frame == 0:
            frames.append(take_frame())
        if (
            remaining_after_onset is None
            and step_index % container.onset_check_interval == 0
        ):
            if leaving.has_left(container.particles_in_container_frame(tilt)[held]):
                onset = {"onset_degrees": round(tilt, 1), "frame": len(frames)}
                remaining_after_onset = extra_steps_after_onset
        elif remaining_after_onset is not None:
            remaining_after_onset -= 1
            if remaining_after_onset <= 0:
                break
    writer = imageio.get_writer(
        str(out.with_suffix(".mp4")),
        fps=frames_per_second,
        codec="libx264",
        quality=8,
        macro_block_size=8,
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "mesh": mesh_file.name,
                "fill": fill,
                "azimuth": azimuth,
                "frames_per_second": frames_per_second,
                "speed": speed,
                **onset,
            },
            indent=2,
        )
    )
    print("wrote", out.with_suffix(".mp4"), len(frames), "frames", onset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--fill", type=float, default=0.5)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--speed", type=float, default=2.0, help="simulated seconds per second of video"
    )
    arguments = parser.parse_args()
    record(
        arguments.mesh,
        arguments.fill,
        arguments.azimuth,
        arguments.out,
        arguments.fps,
        arguments.speed,
    )


if __name__ == "__main__":
    main()
