---
id: giskardpy.qp.pos_in_vel_limits
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/qp/pos_in_vel_limits.py
    lines: [1, 149]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by:
  - giskardpy.qp.adapters
status: stable
tags: [giskardpy, qp, velocity, profile, mpc, position-limits, jerk, slowdown]
last_ingest: 2026-05-17
---

_Symbolic velocity profile functions for position-aware MPC velocity limits: compute "slow-down ASAP" profiles and position-limit-aware shifted profiles as CasADi symbolic expressions._

## Purpose

When a DOF has position limits, the QP must not only respect instantaneous velocity bounds but also ensure the robot can **stop before hitting the limit**. This module provides three layers of symbolic computation that `FreeVariableBounds.b_profile()` in [[giskardpy.qp.adapters]] uses to compute tight but safe velocity bounds for each prediction-horizon step.

All returned vectors are CasADi symbolic (`sm.Vector`, `sm.Scalar`) — they compile into the constraint expressions, not numerical values.

---

## `shifted_velocity_profile(vel_profile, acc_profile, distance, dt)`

Given a pre-computed MPC velocity profile (from `simple_mpc` at max velocity), shifts the
profile backwards depending on the remaining distance to a position limit.

**Logic:** if `distance` is small, the robot must start decelerating sooner — the velocity
profile is shifted right (future velocities move to current time). Implemented as a
piecewise `sm.if_less_eq_cases` chain: for each horizon step `x`, if
`dt * sum(vel_profile[x:]) > distance`, the profile starting at step `x+1` is used.

Returns `(shifted_vel_profile, shifted_acc_profile)` as symbolic vectors.

**Used in:** `FreeVariableBounds.b_profile()` to compute `pos_vel_profile_lb` and
`pos_vel_profile_ub` — the per-step velocity bounds that prevent position limit violations.

---

## `compute_next_vel_and_acc(current_vel, current_acc, vel_limit, jerk_limit, dt, remaining_ph, no_cap)`

Single-step computation of the next velocity and acceleration given jerk constraints and
a target velocity:

1. `acc_cap1` = `acc_cap(current_vel, jerk_limit, dt)` — max acceleration reachable before
   hitting the velocity limit, computed from the Gaussian summation formula.
2. `acc_cap2` = `remaining_ph * jerk_limit * dt` — max acc reachable given horizon length.
3. `acc_ph_max = min(acc_cap1, acc_cap2)` — tightest bound.
4. `acc_to_vel = (vel_limit - current_vel) / dt` — required acceleration to reach target.
5. Clamps to `[next_acc_min, next_acc_max]` = `[current_acc ± jerk_limit * dt]`.

`no_cap` flag skips the `acc_ph_max` clamp — used for step `i=0` with `skip_first`.

---

## `acc_cap(current_vel, jerk_limit, dt)` — closed-form stopping acceleration

```python
def r_gauss(integral):
    return sqrt(2 * integral + 0.25) - 0.5

def acc_cap(current_vel, jerk_limit, dt):
    acc_integral = abs(current_vel) / dt
    jerk_step = jerk_limit * dt
    n = floor(r_gauss(abs(acc_integral / jerk_step)))
    x = (-gauss(n) * jerk_limit * dt + acc_integral) / (n + 1)
    return abs(n * jerk_limit * dt + x)
```

`r_gauss(k)` = `sqrt(2k + 0.25) - 0.5` is the inverse of the Gaussian summation
`G(n) = n*(n+1)/2`. It solves "how many jerk steps of size `jerk_limit*dt` fit in an
acceleration integral of `|current_vel|/dt`". This gives the maximum reachable
acceleration before the velocity limit is hit, expressed in closed form without iteration.

Both `acc_cap` and `compute_next_vel_and_acc` are decorated with `@substitution_cache` —
equivalent to memoization over symbolic inputs, avoiding redundant CasADi subgraph
construction.

---

## `compute_slowdown_asap_vel_profile(current_vel, current_acc, target_vel_profile, jerk_limit, dt, ph, skip_first)`

Builds the full horizon velocity/acceleration/jerk profile by iterating `compute_next_vel_and_acc`
across `ph` steps with a decreasing `remaining_ph`:

```python
for i in range(ph):
    next_vel, next_acc = compute_next_vel_and_acc(next_vel, next_acc, target_vel_profile[i],
                                                   jerk_limit, dt, ph - i - 1, ...)
```

The jerk profile is derived as `(acc[t] - acc[t-1]) / dt`.

**Used in `FreeVariableBounds.b_profile()`** to compute:
- `proj_vel_profile` — projected velocity if the robot slows down ASAP toward `goal_profile`
- `proj_vel_profile_violated` — same without jerk limits (to estimate emergency jerk)

If `proj_vel_profile` violates `pos_vel_profile_lb/ub`, the jerk limits for steps 0-2 are
relaxed to the emergency jerk needed to stop in time.

---

## `implicit_vel_profile(acc_limit, jerk_limit, dt, ph)` — numerical utility

Pure Python (non-symbolic) function that builds the velocity profile starting from
zero with maximum acceleration buildup:

```python
vel_profile = [0, 0]  # last two velocities always zero (system must stop)
for i in range(ph - 2):
    acc = min(acc + jerk_limit * dt, acc_limit)
    vel += acc * dt
    vel_profile.append(vel)
return reversed(vel_profile)
```

Returns a reversed profile — maximum reachable velocities from the end of the horizon
backward to now. Used for bound validation (not in the main symbolic path).

---

## Design pattern

This module forms the **symbolic velocity profile layer** of the MPC:
- It creates CasADi symbolic expressions that depend on current joint state (position, velocity, acceleration).
- These expressions compile into the QP's box constraints (via `FreeVariableBounds`).
- Per-tick, the world state (current pos/vel/acc) is substituted numerically — the bounds adapt automatically to the robot's current state.

## Related

- **Consumer:** [[giskardpy.qp.adapters]] (`FreeVariableBounds.b_profile()`)
- **Concept:** [[concept.qp-controller]] (QP pipeline overview)

## Provenance

- `giskardpy/src/giskardpy/qp/pos_in_vel_limits.py:1-149` at commit `0528d8cf3`.
