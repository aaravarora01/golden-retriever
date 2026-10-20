"""Inverse kinematics for the RoboCasa arm, and what it does and does not solve.

RoboCasa's own controllers do not get the gripper to the object: `OSC_POSE`
stalls 0.288 m short with no joint at a limit, and `WHOLE_BODY_IK`'s flat action
is an IK target the solver converts rather than the documented pose, which
diverges when driven. So the solve is done here instead.

**The solver works.** `solve_posture` finds a collision-free joint configuration
placing the gripper 9 cm above the object with a residual of 0.0008 m, and
forward kinematics on that configuration confirms it. The object was always
reachable from the robot's existing base position — brute-force sampling finds a
posture within 0.052 m without moving the base at all, which retires the earlier
theory that these tasks need base navigation.

**What is not solved is the path.** The start posture and the goal posture are
both collision-free, and straight-line interpolation between them is not: the
arm self-collides at reset (`link5` against the gripper hand) and the gripper
wedges under `cab_1_main_group_bottom` partway along. Commanding the goal
directly leaves the shoulder pitch and elbow 0.400 and 0.598 rad short — not
droop, since integral correction drives the other five joints to 0.002 rad, but
physical blocking. Closing that needs a motion planner that checks collisions
along the path, which MuJoCo can support and this module deliberately does not
attempt.

So this is the foundation a working reference policy needs rather than the
policy itself, and it is committed for three reasons: the solver is correct and
a planner would call it, the measurements it produced retire two wrong theories,
and the joint-control findings below are not discoverable from the documentation.

Three facts about commanding this arm, each found by measurement:

* The arm takes **absolute joint targets** only if the body-part controller is
  reconfigured to `JOINT_POSITION`; the default is `OSC_POSE`. Doing so widens
  the action vector to 13.
* `output_max` **clips the absolute target**, so it must span the joint ranges
  rather than the default narrow band.
* The controller **droops under gravitational load**. A small move from
  equilibrium tracks exactly; a large one leaves the shoulder and elbow short,
  which integral correction fixes where the path is clear.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

#: Damping for the least-squares solve. Large enough to stay stable near
#: singularities, small enough not to stall the step.
DEFAULT_DAMPING = 0.02
#: Cap on one solver iteration, in radians, so a step cannot leap the arm.
MAX_SOLVER_STEP = 0.2
#: Cap on one commanded increment during execution.
MAX_COMMAND_STEP = 0.05
#: Residual at which a posture counts as reaching its target, in metres.
SOLVED_TOLERANCE = 0.004
#: Gain on the integral correction that compensates controller droop.
DROOP_GAIN = 0.35
#: Bound on the accumulated correction, so a blocked joint cannot wind up.
DROOP_LIMIT = 1.5

ARM_JOINTS = tuple(f"robot0_joint{index + 1}" for index in range(7))
GRIP_SITE = "gripper0_right_grip_site"


def joint_position_controller_config(robot: str = "PandaOmron") -> dict:
    """A controller config that accepts absolute joint targets.

    The default composite controller drives the arm in operational space, which
    means handing it a pose and trusting its solve — the thing that does not
    work here. `output_max` must span the joint ranges because it clips the
    absolute target: left at a narrow band, a commanded target outside it is
    silently truncated and the arm settles somewhere else entirely.
    """
    from robosuite.controllers import load_composite_controller_config

    config = copy.deepcopy(load_composite_controller_config(robot=robot))
    config["body_parts"]["right"].update({
        "type": "JOINT_POSITION", "input_type": "absolute",
        "output_max": 3.2, "output_min": -3.2,
    })
    return config


@dataclass
class ArmModel:
    """Indices and limits for the arm, resolved once against a live model."""

    env: object
    site: int = -1
    dof: tuple = ()
    qadr: tuple = ()
    limits: object = None
    robot_geoms: frozenset = frozenset()
    object_geoms: frozenset = frozenset()

    def __post_init__(self) -> None:
        import mujoco

        model = self.env.sim.model._model
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
        if self.site < 0:
            raise ValueError(f"missing grip site {GRIP_SITE!r}")
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
        if any(identifier < 0 for identifier in ids):
            raise ValueError("missing an arm joint; is this a Panda-family robot?")
        self.dof = tuple(int(model.jnt_dofadr[identifier]) for identifier in ids)
        self.qadr = tuple(int(model.jnt_qposadr[identifier]) for identifier in ids)
        self.limits = np.asarray([model.jnt_range[identifier] for identifier in ids])
        self.robot_geoms = frozenset(
            geom for geom in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith(
                ("robot0", "gripper0")))
        # Taken from the environment's own target object rather than matched on
        # a name. The drawer world's object is a jar and this one is a sugar
        # cube, so a substring filter would classify the grasp contact itself
        # as an obstruction and make every grasp look like a collision.
        target = getattr(self.env, "objects", {}).get("obj")
        names = set(getattr(target, "contact_geoms", ()) or ())
        self.object_geoms = frozenset(
            geom for geom in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "") in names)

    @property
    def mid_range(self):
        return self.limits.mean(axis=1)

    def joints(self):
        data = self.env.sim.data._data
        return np.asarray([data.qpos[address] for address in self.qadr])

    def hand(self):
        return np.asarray(self.env.sim.data._data.site_xpos[self.site]).copy()

    def jacobian(self):
        """Position Jacobian for the arm joints: how each joint moves the hand."""
        import mujoco

        model, data = self.env.sim.model._model, self.env.sim.data._data
        translation = np.zeros((3, model.nv))
        rotation = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, translation, rotation, self.site)
        return translation[:, list(self.dof)]

    def obstructions(self) -> list:
        """Robot contacts that are not with the target object.

        Contact with the jar is the point of the exercise; contact with a
        cabinet or with the robot's own links is what blocks a path, and the
        Jacobian cannot see either.
        """
        import mujoco

        model, data = self.env.sim.model._model, self.env.sim.data._data
        found = []
        for contact in data.contact[:data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            if first not in self.robot_geoms and second not in self.robot_geoms:
                continue
            if first in self.object_geoms or second in self.object_geoms:
                continue
            found.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, first) or "",
                          mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, second) or ""))
        return found

    def clamp(self, joints):
        return np.clip(joints, self.limits[:, 0] + 1e-3, self.limits[:, 1] - 1e-3)


def damped_step(arm: ArmModel, target, *, damping: float = DEFAULT_DAMPING,
                posture_bias: float = 0.0):
    """One damped-least-squares increment toward `target`.

    `posture_bias` adds a nullspace pull toward mid-range joints, which keeps
    the elbow from folding into a self-collision — a correction the Jacobian
    cannot make on its own because it does not model contact.
    """
    jacobian = arm.jacobian()
    error = np.asarray(target, dtype=float) - arm.hand()
    square = jacobian @ jacobian.T + damping ** 2 * np.eye(3)
    increment = jacobian.T @ np.linalg.solve(square, error)
    if posture_bias:
        bias = posture_bias * (arm.mid_range - arm.joints())
        increment += bias - jacobian.T @ np.linalg.solve(square, jacobian @ bias)
    norm = float(np.linalg.norm(increment))
    if norm > MAX_SOLVER_STEP:
        increment *= MAX_SOLVER_STEP / norm
    return increment, float(np.linalg.norm(error))


@dataclass
class PostureSolution:
    """A solved joint configuration, with what it cost."""

    joints: object
    residual_m: float
    obstructions: int
    restarts: int

    @property
    def reached(self) -> bool:
        return self.residual_m <= SOLVED_TOLERANCE

    @property
    def clear(self) -> bool:
        return self.obstructions == 0

    def to_dict(self) -> dict:
        return {"residual_m": round(self.residual_m, 5), "obstructions": self.obstructions,
                "restarts": self.restarts, "reached": self.reached, "clear": self.clear}


def solve_posture(arm: ArmModel, target, *, restarts: int = 30, iterations: int = 120,
                  seed: int = 0) -> PostureSolution:
    """Find a joint configuration putting the hand at `target`.

    Random restarts matter: from the reset posture alone the solve converges
    into the cabinet, and a restart from elsewhere in joint space finds a clear
    configuration. The simulation state is restored before returning, so this is
    a query and not a move.

    Scoring prefers a clear posture over a marginally closer one, because a
    posture in collision cannot be commanded even when its residual is small.
    """
    import mujoco

    if type(restarts) is not int or not 1 <= restarts <= 500:
        raise ValueError("restarts must be an int from 1 through 500")
    if type(iterations) is not int or not 1 <= iterations <= 2000:
        raise ValueError("iterations must be an int from 1 through 2000")
    model, data = arm.env.sim.model._model, arm.env.sim.data._data
    random = np.random.default_rng(seed)
    saved = arm.joints().copy()
    target = np.asarray(target, dtype=float)

    def apply(joints):
        for address, value in zip(arm.qadr, joints):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)

    best, best_joints, best_clash = float("inf"), saved.copy(), 0
    for restart in range(restarts):
        joints = saved.copy() if restart == 0 else random.uniform(
            arm.limits[:, 0], arm.limits[:, 1])
        for _ in range(iterations):
            apply(joints)
            increment, residual = damped_step(arm, target)
            if residual < SOLVED_TOLERANCE:
                break
            joints = arm.clamp(joints + increment)
        apply(joints)
        residual = float(np.linalg.norm(target - arm.hand()))
        clash = len(arm.obstructions())
        score = residual + 0.5 * clash
        if score < best:
            best, best_joints, best_clash = score, joints.copy(), clash
    apply(saved)
    return PostureSolution(joints=best_joints,
                           residual_m=float(np.linalg.norm(target - _hand_at(arm, best_joints))),
                           obstructions=best_clash, restarts=restarts)


def _hand_at(arm: ArmModel, joints):
    """Forward kinematics: where a configuration would put the hand."""
    import mujoco

    model, data = arm.env.sim.model._model, arm.env.sim.data._data
    saved = arm.joints().copy()
    for address, value in zip(arm.qadr, joints):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)
    position = arm.hand()
    for address, value in zip(arm.qadr, saved):
        data.qpos[address] = value
    mujoco.mj_forward(model, data)
    return position


@dataclass
class DroopCompensator:
    """Integral correction for the joint controller's load-dependent droop.

    Measured: commanding a solved posture directly leaves the shoulder pitch and
    elbow 0.400 and 0.598 rad short, while the other five joints arrive. Feeding
    the tracking error back brings those five to about 0.002 rad. It does not
    fix a joint that is physically blocked, which is why `wound_up` is exposed —
    a saturated correction is the signal that the path, not the controller, is
    the problem.
    """

    gain: float = DROOP_GAIN
    limit: float = DROOP_LIMIT
    bias: object = field(default=None)

    def reset(self, width: int = 7) -> None:
        self.bias = np.zeros(width)

    def command(self, want, actual):
        if self.bias is None:
            self.reset(len(want))
        self.bias = np.clip(self.bias + self.gain * (np.asarray(want) - np.asarray(actual)),
                            -self.limit, self.limit)
        return np.asarray(want) + self.bias

    @property
    def wound_up(self) -> bool:
        """Whether any correction has saturated, meaning a joint is blocked."""
        return self.bias is not None and bool(np.any(np.abs(self.bias) >= self.limit - 1e-9))


# --- motion planning ------------------------------------------------------
#
# A collision-checked path is the piece the solver alone does not supply. The
# start and goal postures are each collision-free while the straight line
# between them is not, so commanding the goal directly leaves the shoulder and
# elbow blocked against the cabinet.
#
# Single-waypoint routing is too weak: 400 random intermediate postures produced
# no clear two-leg path, and the mid-range posture sits in 19 collisions because
# it is inside the kitchen furniture. But the space is not cramped — 54% of
# sampled joint space is collision-free, and 95% of a small neighbourhood around
# the start posture is. The failure was straight-line *segments* clipping
# obstacles, not a shortage of free space, which is exactly the case a tree-based
# planner growing in small checked steps handles.

#: Planner step, in radians. Small enough that a step cannot tunnel through a
#: thin obstacle, large enough to cross the workspace in a tractable number.
PLANNER_STEP = 0.15
#: Collision checks interpolated within one step, so a step is verified rather
#: than assumed.
STEP_CHECKS = 4


@dataclass
class PlannedPath:
    """A collision-checked joint-space path, and what finding it cost."""

    waypoints: list = field(default_factory=list)
    iterations: int = 0
    found: bool = False
    smoothed_from: int = 0

    def to_dict(self) -> dict:
        return {"found": self.found, "waypoints": len(self.waypoints),
                "iterations": self.iterations, "smoothed_from": self.smoothed_from}


class _CollisionCache:
    """Posture collision queries against a restored simulation state.

    Every query writes `qpos` and runs forward kinematics, so the caller's state
    has to be put back — a planner that left the arm at its last sampled posture
    would corrupt the episode it was planning for.
    """

    def __init__(self, arm: ArmModel):
        import mujoco

        self.arm = arm
        self.mujoco = mujoco
        self.model = arm.env.sim.model._model
        self.data = arm.env.sim.data._data
        self.saved = arm.joints().copy()
        self.checks = 0

    def clear(self, joints) -> bool:
        self.checks += 1
        for address, value in zip(self.arm.qadr, joints):
            self.data.qpos[address] = value
        self.mujoco.mj_forward(self.model, self.data)
        return not self.arm.obstructions()

    def segment_clear(self, start, end, checks: int = STEP_CHECKS) -> bool:
        start, end = np.asarray(start), np.asarray(end)
        for index in range(1, checks + 1):
            if not self.clear(start + (end - start) * (index / checks)):
                return False
        return True

    def restore(self) -> None:
        for address, value in zip(self.arm.qadr, self.saved):
            self.data.qpos[address] = value
        self.mujoco.mj_forward(self.model, self.data)


def plan_path(arm: ArmModel, goal, *, iterations: int = 4000, step: float = PLANNER_STEP,
              seed: int = 0, smooth: bool = True) -> PlannedPath:
    """Bidirectional RRT-Connect from the current posture to `goal`.

    Two trees grown toward each other, each extension verified by interpolated
    collision checks rather than assumed clear. Bidirectional because the free
    space around the goal is markedly tighter than around the start — 35% of a
    small neighbourhood versus 95% — so growing only forward would spend most of
    its samples far from where the hard part is.

    The simulation state is restored before returning, so this is a query.
    """
    if type(iterations) is not int or not 1 <= iterations <= 100_000:
        raise ValueError("iterations must be an int from 1 through 100000")
    if type(step) is not float or not 0.0 < step <= 1.0:
        raise ValueError("step must be a float in (0, 1]")
    goal = np.asarray(goal, dtype=float)
    cache = _CollisionCache(arm)
    try:
        start = cache.saved.copy()
        if not cache.clear(goal):
            return PlannedPath(iterations=0, found=False)
        # The reset posture self-collides, so the start is allowed to be in
        # contact; the path out of it is what must be clear.
        trees = ([start.copy()], [goal.copy()])
        parents = ([-1], [-1])
        random = np.random.default_rng(seed)
        low, high = arm.limits[:, 0], arm.limits[:, 1]

        def extend(tree, parent, toward):
            distances = [float(np.linalg.norm(node - toward)) for node in tree]
            nearest = int(np.argmin(distances))
            direction = toward - tree[nearest]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                return nearest
            candidate = np.clip(tree[nearest] + direction * (min(step, norm) / norm), low, high)
            if not cache.segment_clear(tree[nearest], candidate):
                return None
            tree.append(candidate)
            parent.append(nearest)
            return len(tree) - 1

        connection = None
        for iteration in range(1, iterations + 1):
            sample = random.uniform(low, high)
            active = iteration % 2
            grown = extend(trees[active], parents[active], sample)
            if grown is None:
                continue
            other = 1 - active
            reached = extend(trees[other], parents[other], trees[active][grown])
            if reached is not None and cache.segment_clear(
                    trees[other][reached], trees[active][grown]):
                connection = (iteration, active, grown, other, reached)
                break
        if connection is None:
            return PlannedPath(iterations=iterations, found=False)

        iteration, active, grown, other, reached = connection

        def chain(tree, parent, index):
            path = []
            while index != -1:
                path.append(tree[index])
                index = parent[index]
            return path

        forward = chain(trees[active], parents[active], grown)[::-1]
        backward = chain(trees[other], parents[other], reached)
        waypoints = forward + backward if active == 0 else (backward + forward[::-1])[::-1]
        raw = len(waypoints)
        if smooth:
            waypoints = _shortcut(cache, waypoints, random)
        return PlannedPath(waypoints=[np.asarray(point) for point in waypoints],
                           iterations=iteration, found=True, smoothed_from=raw)
    finally:
        cache.restore()


def _shortcut(cache: _CollisionCache, waypoints, random, rounds: int = 120):
    """Replace detours with straight segments where they are collision-free.

    An RRT path wanders, and a wandering path costs control steps — one of the
    study's reported metrics — so leaving it unsmoothed would inflate a number
    the harness exists to measure.
    """
    points = [np.asarray(point) for point in waypoints]
    for _ in range(rounds):
        if len(points) <= 2:
            break
        first, second = sorted(random.integers(0, len(points), 2))
        if second - first < 2:
            continue
        if cache.segment_clear(points[first], points[second], checks=8):
            points = points[:first + 1] + points[second:]
    return points
