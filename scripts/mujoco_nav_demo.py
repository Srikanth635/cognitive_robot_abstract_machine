"""
MuJoCo navigation demo.

Loads a simple scene with a blue cylinder robot, red/orange obstacles, and
a green goal marker. The robot navigates through a sequence of waypoints that
route around the obstacles to reach the goal.

Run from the repo root:
    python scripts/mujoco_nav_demo.py

Viewer controls:
    Space           pause / resume physics
    Right-click drag  rotate camera
    Scroll          zoom
    Ctrl+C          exit
"""

import math
import pathlib
import time

import mujoco
import numpy as np

from physics_simulators.mujoco_simulator import MujocoSimulator

# ── Config ────────────────────────────────────────────────────────────────────

_SCENE = str(pathlib.Path(__file__).parent / "nav_world.xml")
STEP_SIZE   = 2e-3   # physics timestep (s)
SPEED       = 0.8    # robot translation speed (m/s in sim time)
YAW_SPEED   = 2.0    # robot rotation speed (rad/s in sim time)
GOAL_RADIUS = 0.25   # distance to waypoint considered "reached" (m)
HEIGHT      = 0.07   # robot z stays fixed at this height

# Waypoints the robot follows (x, y) — chosen to route around obstacles.
# Final waypoint matches the green goal marker in nav_world.xml.
WAYPOINTS = [
    ( 0.4, -0.7),   # 1 — slip south of wall_b
    ( 1.3, -0.8),   # 2 — clear wall_b, approach pillar from south
    ( 2.2, -0.4),   # 3 — east of pillar
    ( 2.5,  0.6),   # 4 — arc north
    ( 2.5,  1.8),   # 5 — GOAL (matches goal body in scene)
]


# ── Math helpers ──────────────────────────────────────────────────────────────

def _yaw_from_quat(q: np.ndarray) -> float:
    """Extract yaw from WXYZ quaternion."""
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _yaw_to_quat(yaw: float) -> np.ndarray:
    """Build WXYZ quaternion for a pure yaw rotation."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _angle_diff(a: float, b: float) -> float:
    """Signed difference (a − b) wrapped to (−π, π]."""
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


# ── Navigation controller ─────────────────────────────────────────────────────

class WaypointNavigator:
    """Simple kinematic waypoint follower.

    At each call to update():
      - Rotates toward the next waypoint (capped at YAW_SPEED per second).
      - When roughly aligned, translates forward (capped at SPEED per second).
      - Advances to the next waypoint once within GOAL_RADIUS.
    """

    def __init__(self, waypoints):
        self._wps   = list(waypoints)
        self._idx   = 0
        self.done   = False

    @property
    def current_goal(self):
        return self._wps[self._idx] if not self.done else self._wps[-1]

    def update(self, pos: np.ndarray, yaw: float, dt: float):
        """Return (new_pos, new_yaw). Mutates internal state."""
        if self.done:
            return pos, yaw

        gx, gy = self.current_goal
        dx, dy = gx - pos[0], gy - pos[1]
        dist   = math.hypot(dx, dy)

        if dist < GOAL_RADIUS:
            self._idx += 1
            if self._idx >= len(self._wps):
                self.done = True
                label = "GOAL" if self._idx == len(self._wps) else f"wp{self._idx}"
                print(f"[Nav] Reached {label}!")
            else:
                print(f"[Nav] Reached wp{self._idx}, heading to wp{self._idx + 1} …")
            return pos, yaw

        # Desired heading toward goal
        desired_yaw = math.atan2(dy, dx)
        err_yaw     = _angle_diff(desired_yaw, yaw)
        max_dyaw    = YAW_SPEED * dt

        # Rotate step
        dyaw = math.copysign(min(abs(err_yaw), max_dyaw), err_yaw)
        new_yaw = yaw + dyaw

        # Translate only when roughly aligned (within 20°)
        new_pos = pos.copy()
        if abs(err_yaw) < math.radians(20):
            step    = min(SPEED * dt, dist)
            new_pos[0] += math.cos(new_yaw) * step
            new_pos[1] += math.sin(new_yaw) * step

        return new_pos, new_yaw


# ── Sim helpers ───────────────────────────────────────────────────────────────

def _get_robot_state(sim: MujocoSimulator):
    """Read robot position [x,y,z] and yaw from qpos."""
    pos_result = sim.get_body_position("robot")
    if pos_result.result is None:
        raise RuntimeError("Could not read robot position")
    pos = pos_result.result.copy()

    # Quaternion lives at qpos_adr + 3 for the freejoint
    body_id  = mujoco.mj_name2id(sim._mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    jnt_adr  = sim._mj_model.body(body_id).jntadr[0]
    qpos_adr = sim._mj_model.jnt_qposadr[jnt_adr]
    q = sim._mj_data.qpos[qpos_adr + 3 : qpos_adr + 7]  # WXYZ
    return pos, _yaw_from_quat(q)


def _set_robot_state(sim: MujocoSimulator, pos: np.ndarray, yaw: float):
    """Write robot pose directly to qpos.

    We bypass sim.set_body_position() / set_body_quaternion() because those
    methods call mujoco.mj_step1() internally to update FK, which uses the
    MuJoCo stack.  The passive viewer's render thread calls mj_copyDataVisual
    at the same time, and the two collide:
        mj_copyDataVisual: attempting to copy mjData while stack is in use

    Writing to qpos is a plain numpy slice — no MuJoCo functions are called,
    so there is no stack conflict.  The full mj_step() inside sim.step()
    updates forward kinematics correctly under the viewer's render lock.
    """
    body_id  = mujoco.mj_name2id(sim._mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    jnt_adr  = sim._mj_model.body(body_id).jntadr[0]
    qpos_adr = int(sim._mj_model.jnt_qposadr[jnt_adr])
    sim._mj_data.qpos[qpos_adr:qpos_adr + 3] = pos
    sim._mj_data.qpos[qpos_adr + 3:qpos_adr + 7] = _yaw_to_quat(yaw)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sim = MujocoSimulator(file_path=_SCENE, _headless=False, _step_size=STEP_SIZE)
    sim.start(simulate_in_thread=False, render_in_thread=True)

    nav  = WaypointNavigator(WAYPOINTS)
    pos  = np.array([0.0, 0.0, HEIGHT])
    yaw  = 0.0

    # Print the plan
    print("[Nav] Plan: navigate around obstacles to the green goal marker")
    for i, (wx, wy) in enumerate(WAYPOINTS):
        label = "GOAL" if i == len(WAYPOINTS) - 1 else f"wp{i+1}"
        print(f"       {label}: ({wx:.1f}, {wy:.1f})")
    print()

    try:
        while sim.renderer.is_running():
            pos, yaw = nav.update(pos, yaw, STEP_SIZE)
            _set_robot_state(sim, pos, yaw)

            sim.step()
            time.sleep(STEP_SIZE)

            if nav.done:
                # Hold at goal, keep rendering
                print("[Nav] Navigation complete — viewer stays open.")
                while sim.renderer.is_running():
                    sim.step()
                    time.sleep(STEP_SIZE)
                break

    except KeyboardInterrupt:
        pass

    sim.stop()


if __name__ == "__main__":
    main()
