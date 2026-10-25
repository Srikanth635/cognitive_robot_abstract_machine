"""
The simulated tilt of a filled container in Isaac Sim: a stage holding the container as
a kinematic body with its exact mesh as collider and the substance as PhysX fluid
particles, stepped on the GPU while the container turns at a constant rate.

Isaac Sim's application must run before this module is imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import carb
import numpy as np
import omni.usd
import trimesh
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import (
    SETTING_SUPPRESS_READBACK,
    SETTING_UPDATE_PARTICLES_TO_USD,
    SETTING_UPDATE_TO_USD,
)
from omni.physx.scripts import particleUtils, physicsUtils
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdUtils, Vt
from typing_extensions import Optional

from experiments.flow_onset_measurement.runs import (
    ParticlesLeaving,
    SimulationSettings,
)

# %% particle sizes


@dataclass(frozen=True)
class ParticleOffsets:
    """
    The distances PhysX keeps between particles and between particles and the container,
    following PhysX's own fluid demonstrations for a given spacing of particles at rest.
    """

    spacing: float
    """
    The distance, in metres, between neighbouring particles at rest.
    """

    rest_share: float = 0.99
    """
    The share of the contact offset a particle keeps from a solid at rest.
    """

    fluid_share: float = 0.6
    """
    The share of the solid rest offset fluid particles keep from each other at rest.
    """

    @property
    def particle_contact_offset(self) -> float:
        """
        The distance at which particles start to interact.
        """
        return self.spacing / (2 * self.rest_share * self.fluid_share)

    @property
    def solid_rest_offset(self) -> float:
        """
        The distance particles keep from the container's surface.
        """
        return self.rest_share * self.particle_contact_offset

    @property
    def fluid_rest_offset(self) -> float:
        """
        Half the distance fluid particles keep from each other, half the spacing.
        """
        return self.fluid_share * self.solid_rest_offset


# %% the stage


@dataclass
class TiltedContainer:
    """
    A container filled with fluid particles on a stage of its own, settled upright and
    then tilted toward one side of its own until the substance starts to leave it.
    """

    mesh: trimesh.Trimesh
    """
    The container's shape, in its own frame, its up axis along z.
    """

    particles: np.ndarray
    """
    Where the particles start, in the container's own frame, one row each.
    """

    gravity_azimuth: float
    """
    The direction, in degrees from the container's own x axis about its up axis, toward
    which it is tilted.
    """

    settings: SimulationSettings
    """
    The settings the tilt is simulated with.
    """

    settling_time: float = 1.0
    """
    How long, in seconds, the particles settle upright before the tilt.
    """

    maximum_tilt: float = 170.0
    """
    The tilt, in degrees, at which the tilt stops without an onset.
    """

    gravity: float = 9.81
    """
    The acceleration, in metres per second squared, gravity pulls the particles with.
    """

    solver_position_iterations: int = 16
    """
    How often per step PhysX resolves the particles' positions.
    """

    onset_check_interval: int = 6
    """
    How many steps pass between two checks of whether the particles have left.
    """

    particle_friction: float = 0.1
    """
    The friction between the particles and the container; PhysX's own setting for water.
    """

    _rotation: Optional[UsdGeom.XformOp] = field(init=False, default=None)
    """
    The orientation of the container on the stage.
    """

    _points: Optional[UsdGeom.Points] = field(init=False, default=None)
    """
    The particles on the stage.
    """

    _simulated_time: float = field(init=False, default=0.0)
    """
    How much time has been simulated.
    """

    @property
    def offsets(self) -> ParticleOffsets:
        """
        The distances PhysX keeps around the particles.
        """
        return ParticleOffsets(spacing=self.settings.particle_spacing)

    @property
    def tilt_axis(self) -> np.ndarray:
        """
        The axis, in the container's own frame and the world's, turning the container
        toward :attr:`gravity_azimuth`.
        """
        azimuth = np.radians(self.gravity_azimuth)
        return np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])

    def build(self) -> None:
        """
        Build the stage, the container upright at the origin, and attach PhysX to it.
        """
        settings = carb.settings.get_settings()
        settings.set(SETTING_UPDATE_TO_USD, True)
        settings.set(SETTING_UPDATE_PARTICLES_TO_USD, True)
        settings.set(SETTING_SUPPRESS_READBACK, False)
        get_physx_simulation_interface().detach_stage()
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = self.build_scene(stage)
        self.build_container(stage)
        self.build_particles(stage, scene)
        get_physx_simulation_interface().attach_stage(
            UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        )

    def build_scene(self, stage) -> UsdPhysics.Scene:
        """
        :return: The physics scene, stepped on the GPU, which fluid particles need.
        """
        scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(self.gravity)
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
        physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
        physx_scene.CreateSolverTypeAttr().Set("TGS")
        physx_scene.CreateTimeStepsPerSecondAttr().Set(
            round(1.0 / self.settings.time_step)
        )
        return scene

    def build_container(self, stage) -> None:
        """
        Add the container as a kinematic body whose exact mesh collides, so it holds the
        particles in its hollow.
        """
        body = UsdGeom.Xform.Define(stage, "/World/container")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr().Set(
            True
        )
        self._rotation = body.AddOrientOp()
        self._rotation.Set(Gf.Quatf(1.0))
        geometry = UsdGeom.Mesh.Define(stage, "/World/container/mesh")
        geometry.CreatePointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*vertex) for vertex in self.mesh.vertices])
        )
        geometry.CreateFaceVertexCountsAttr().Set(
            Vt.IntArray([3] * len(self.mesh.faces))
        )
        geometry.CreateFaceVertexIndicesAttr().Set(
            Vt.IntArray(self.mesh.faces.flatten().tolist())
        )
        UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(
            geometry.GetPrim()
        ).CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(geometry.GetPrim())
        collision.CreateContactOffsetAttr().Set(self.settings.particle_spacing / 2)
        collision.CreateRestOffsetAttr().Set(0.0)

    def build_particles(self, stage, scene: UsdPhysics.Scene) -> None:
        """
        Add the substance as fluid particles with the settings' material.
        """
        offsets = self.offsets
        system_path = "/World/particleSystem"
        particleUtils.add_physx_particle_system(
            stage,
            system_path,
            simulation_owner=scene.GetPath(),
            contact_offset=offsets.solid_rest_offset
            + self.settings.particle_spacing / 2,
            rest_offset=offsets.solid_rest_offset,
            particle_contact_offset=offsets.particle_contact_offset,
            solid_rest_offset=offsets.solid_rest_offset,
            fluid_rest_offset=offsets.fluid_rest_offset,
            solver_position_iterations=self.solver_position_iterations,
        )
        material_path = "/World/particleMaterial"
        particleUtils.add_pbd_particle_material(
            stage,
            material_path,
            cohesion=self.settings.fluid_cohesion,
            damping=0.0,
            friction=self.particle_friction,
            surface_tension=self.settings.fluid_surface_tension,
            viscosity=self.settings.fluid_viscosity,
            vorticity_confinement=0.0,
        )
        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath(system_path), material_path
        )
        count = len(self.particles)
        self._points = particleUtils.add_physx_particleset_points(
            stage,
            "/World/particles",
            [Gf.Vec3f(*position) for position in self.particles.tolist()],
            [Gf.Vec3f(0.0, 0.0, 0.0)] * count,
            [2 * offsets.fluid_rest_offset] * count,
            system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=0.0,
            density=1000.0,
        )

    # %% stepping

    def step(self, tilt: float) -> None:
        """
        Turn the container to a tilt and simulate one step.

        :param tilt: The tilt from upright, in degrees.
        """
        self._rotation.Set(
            Gf.Quatf(Gf.Rotation(Gf.Vec3d(*self.tilt_axis), tilt).GetQuat())
        )
        get_physx_simulation_interface().simulate(
            self.settings.time_step, self._simulated_time
        )
        get_physx_simulation_interface().fetch_results()
        self._simulated_time += self.settings.time_step

    def particles_in_container_frame(self, tilt: float) -> np.ndarray:
        """
        :param tilt: The container's tilt now, in degrees.
        :return: Where the particles are now, in the container's own frame.
        """
        world = np.asarray(self._points.GetPointsAttr().Get(), dtype=float)
        rotation = np.array(
            Gf.Matrix3d(Gf.Rotation(Gf.Vec3d(*self.tilt_axis), tilt)).GetTranspose()
        )
        return world @ rotation

    def settle(self) -> np.ndarray:
        """
        Let the particles settle upright.

        :return: Where the settled particles are, in the container's own frame.
        """
        for _ in range(round(self.settling_time / self.settings.time_step)):
            self.step(0.0)
        return self.particles_in_container_frame(0.0)

    def onset(self, held: np.ndarray, leaving: ParticlesLeaving) -> Optional[float]:
        """
        Tilt the settled container at the settings' rate until the particles it held
        once settled count as leaving it.

        :param held: Which particles the container held once settled.
        :param leaving: When the particles count as leaving.
        :return: The tilt, in degrees, at which they left; ``None`` when they had not
            left at :attr:`maximum_tilt`.
        """
        increment = self.settings.tilt_rate * self.settings.time_step
        steps = 0
        while steps * increment < self.maximum_tilt:
            steps += 1
            tilt = steps * increment
            self.step(tilt)
            if steps % self.onset_check_interval == 0 and leaving.has_left(
                self.particles_in_container_frame(tilt)[held]
            ):
                return tilt
        return None
