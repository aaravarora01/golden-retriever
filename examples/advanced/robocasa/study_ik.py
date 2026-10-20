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

**And the execution gap is closed.** `follow_path` commands densely along each
planned segment, advancing the commanded point only as the arm keeps up. With
it the gripper reaches a point 9 cm above the object to within **0.7 mm with
zero collisions**, from 0.512 m at reset and against the 0.288 m where
RoboCasa's own controller stalls.

Why that was the fix, and why the obvious alternatives were not: the planner
verifies the straight line between waypoints, but *commanding* a waypoint lets
the arm take its own dynamic route there, which is not the line that was
checked. Ramping relative to the achieved pose stalls, because the commanded
point never gets ahead of a lagging arm. A fixed time schedule overruns it, so
the arm falls up to 0.8 rad behind and collides. Gating the advance on the arm
keeping up is what makes the executed motion follow the verified path.

What remains is the grasp itself, not the reach. See the end of this module.

Three facts about commanding this arm, each found by measurement:

* The arm takes **absolute joint targets** only if the body-part controller is
  reconfigured to `JOINT_POSITION`; the default is `OSC_POSE`. Doing so widens
  the action vector to 13.
* `output_max` does **not** clip an absolute target. It scales the *delta*
  path only: with `input_type: absolute` the controller takes `goal_qpos =
  action` verbatim, and `env.step` does not clip to `action_spec` either — a
  commanded 2.5 rad arrives as 2.5 even though the spec advertises ±1. An
  earlier version of this note claimed otherwise.
* The controller **already compensates gravity** — `use_torque_compensation`
  defaults on and the torque is `mass_matrix @ (kp·e + kd·ė) + qfrc_bias`. It
  tracks a 1.0 rad single-joint move to 0.0008 rad when the path is clear. What
  looked like droop was a *blocked* joint: commanding the shoulder +1.0 rad
  drives the arm into contact, and the torque then saturates against the
  actuator limit while the joint does not arrive.
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
#: A top-down grasp wants the gripper's approach axis pointing here.
DOWN = np.array([0.0, 0.0, -1.0])
#: Weight on orientation when choosing between position-accurate solutions, in
#: metres per radian. Calibrated: the solution manifold is wide enough that
#: random restarts already contain near-vertical postures, so this only has to
#: break the tie, not steer the solve.
ORIENTATION_WEIGHT = 0.08
#: Restarts to use when an approach direction is requested. The default budget
#: is calibrated for position alone; orientation needs a wider search — 60
#: restarts settled for 34.4 degrees off vertical where 200 reached 13.6.
ORIENTATION_RESTARTS = 200


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

    def approach_axis(self):
        """The gripper's approach direction in world coordinates."""
        return np.asarray(
            self.env.sim.data._data.site_xmat[self.site]).reshape(3, 3)[:, 2]

    def tilt_from(self, direction=DOWN) -> float:
        """Angle between the approach axis and `direction`, in radians."""
        cosine = float(np.dot(self.approach_axis(), np.asarray(direction, dtype=float)))
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


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
    tilt_rad: float = 0.0

    @property
    def reached(self) -> bool:
        return self.residual_m <= SOLVED_TOLERANCE

    @property
    def clear(self) -> bool:
        return self.obstructions == 0

    @property
    def tilt_deg(self) -> float:
        """How far the gripper points from the requested approach direction."""
        return float(np.degrees(self.tilt_rad))

    def to_dict(self) -> dict:
        return {"residual_m": round(self.residual_m, 5), "obstructions": self.obstructions,
                "restarts": self.restarts, "reached": self.reached, "clear": self.clear,
                "tilt_deg": round(self.tilt_deg, 2)}


def solve_posture(arm: ArmModel, target, *, restarts: int = 30, iterations: int = 120,
                  seed: int = 0, approach=None) -> PostureSolution:
    """Find a joint configuration putting the hand at `target`.

    Random restarts matter: from the reset posture alone the solve converges
    into the cabinet, and a restart from elsewhere in joint space finds a clear
    configuration. The simulation state is restored before returning, so this is
    a query and not a move.

    Scoring prefers a clear posture over a marginally closer one, because a
    posture in collision cannot be commanded even when its residual is small.

    The default restart budget is calibrated rather than guessed: from eight
    different starting postures, 10 restarts reaches a clear solution 5 times
    and 30 reaches it 8 times out of 8, with a worst residual of 0.0013 m.

    A target *at* the object's centre is a different matter and will not solve —
    the grip site cannot occupy the counter surface the object rests on. Aim a
    centimetre or two above it and let the fingers close around it.

    `approach` requests a gripper direction, `DOWN` for a top-down grasp. It is
    a *selection* criterion rather than a second task in the solve, and that is
    deliberate: with seven joints and a three-axis position target the solution
    manifold is four-dimensional, and random restarts already contain postures
    pointing almost straight down — 13.6 degrees at the approach height and
    11.9 at grasp height, both with sub-millimetre position error and no
    collisions. Across restarts the orientation spans 13 to 159 degrees, so
    without this the solver simply returns whichever posture it landed on, which
    measured 49 to 52 degrees off vertical.

    Trying to steer orientation *inside* the solve was tried first and is worse:
    stacking it as a weighted second task costs position (0.39 m against
    0.0008 m), and so does projecting it into the nullspace of the position
    task — the exact projector leaks nothing, but the clamped joint motion no
    longer preserves position, and separating the step caps did not recover it.
    Selection among restarts needs none of that machinery.

    Note that clear solutions get scarcer closer to the object: 27 of 200 at the
    approach height against 4 of 200 at grasp height, so a low target wants a
    larger restart budget. The budget also bounds the orientation it can find —
    60 restarts settled for 34.4 degrees where 200 reached 13.6 — so
    `ORIENTATION_RESTARTS` is the figure to use when `approach` is requested,
    rather than the default that suffices for position alone.
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

    want = None if approach is None else np.asarray(approach, dtype=float)
    best, best_joints, best_clash, best_tilt = float("inf"), saved.copy(), 0, 0.0
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
        # Measured either way, so a solution always reports how the gripper is
        # pointing; only the *scoring* is opt-in. Reporting 0.0 for "not
        # requested" would read as a perfect top-down posture.
        tilt = arm.tilt_from(DOWN if want is None else want)
        score = residual + 0.5 * clash + (0.0 if want is None
                                          else ORIENTATION_WEIGHT * tilt)
        if score < best:
            best, best_joints, best_clash, best_tilt = score, joints.copy(), clash, tilt
    apply(saved)
    return PostureSolution(joints=best_joints,
                           residual_m=float(np.linalg.norm(target - _hand_at(arm, best_joints))),
                           obstructions=best_clash, restarts=restarts, tilt_rad=best_tilt)


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

        # Tree 0 is rooted at the start and tree 1 at the goal, so name the
        # connecting node on each side explicitly rather than reasoning about
        # which tree happened to grow last. Getting this wrong produced paths
        # that did not end at the goal on odd iterations, which made every
        # downstream measurement depend on the parity of the connecting step.
        start_side = grown if active == 0 else reached
        goal_side = reached if active == 0 else grown
        waypoints = (chain(trees[0], parents[0], start_side)[::-1]
                     + chain(trees[1], parents[1], goal_side))
        raw = len(waypoints)
        if smooth:
            waypoints = _shortcut(cache, waypoints, random)
        points = [np.asarray(point) for point in waypoints]
        # A path that does not run from the start to the goal is worse than no
        # path, because a follower reports success against whatever endpoint it
        # was handed. Checked rather than assumed.
        if (not np.allclose(points[0], start, atol=1e-9)
                or not np.allclose(points[-1], goal, atol=1e-9)):
            raise AssertionError(
                "planned path does not run from the start posture to the goal")
        return PlannedPath(waypoints=points, iterations=iteration, found=True,
                           smoothed_from=raw)
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


# --- following a planned path --------------------------------------------
#
# The piece that makes a verified path actually get executed. The planner checks
# the straight line between two waypoints; commanding the second waypoint does
# not make the arm travel that line, it makes the arm find its own way there
# under its own dynamics, which is how a checked path still ends in a collision.
#
# Three approaches were measured. Commanding each waypoint directly leaves later
# ones 1.8-2.1 rad short and collides. Ramping relative to the arm's achieved
# pose never gets ahead of a lagging arm and stalls 0.5 rad out. A fixed time
# schedule runs away from the arm, which falls up to 0.8 rad behind and hits
# things the kinematic check called clear. Advancing the commanded point only
# once the arm is within tolerance of it keeps the error small enough that the
# executed motion follows the verified line.

#: How far the commanded point advances per tick, in radians.
FOLLOW_INCREMENT = 0.02
#: How far behind the commanded point the arm may fall before it stops
#: advancing. Larger is faster and tracks the planned line less closely.
FOLLOW_TOLERANCE = 0.05
#: Ticks spent settling onto each waypoint before moving to the next.
FOLLOW_SETTLE = 80


@dataclass
class FollowReport:
    """What executing a path actually did, as opposed to what was planned."""

    ticks: int = 0
    worst_obstructions: int = 0
    joint_error: float = 0.0
    stalled: bool = False

    def to_dict(self) -> dict:
        return {"ticks": self.ticks, "worst_obstructions": self.worst_obstructions,
                "joint_error": round(self.joint_error, 5), "stalled": self.stalled,
                "clean": self.worst_obstructions == 0 and not self.stalled}


def follow_path(arm: ArmModel, waypoints, *, command, increment: float = FOLLOW_INCREMENT,
                tolerance: float = FOLLOW_TOLERANCE, settle: int = FOLLOW_SETTLE,
                budget: int = 1200) -> FollowReport:
    """Drive the arm along `waypoints`, keeping the command near the arm.

    `command` is called with a joint target each tick and is expected to step
    the simulation — the caller owns the gripper and any other action segments,
    so this function never constructs an action vector itself.

    `worst_obstructions` is the honest measure of whether the verified path was
    actually followed: a non-zero value means the executed motion left the line
    the planner checked, whatever the final error says.
    """
    if type(increment) is not float or not 0.0 < increment <= 0.5:
        raise ValueError("increment must be a float in (0, 0.5]")
    if type(tolerance) is not float or not 0.0 < tolerance <= 1.0:
        raise ValueError("tolerance must be a float in (0, 1]")
    points = [np.asarray(point, dtype=float) for point in waypoints]
    if len(points) < 2:
        raise ValueError("following needs at least a start and a target")
    report = FollowReport()
    for target in points[1:]:
        start = arm.joints().copy()
        span = float(np.linalg.norm(target - start))
        if span < 1e-9:
            continue
        direction = (target - start) / span
        travelled, used = 0.0, 0
        while travelled < span and used < budget:
            if float(np.linalg.norm(arm.joints() - (start + direction * travelled))) < tolerance:
                travelled = min(span, travelled + increment)
            command(arm.clamp(start + direction * travelled))
            used += 1
            report.ticks += 1
            report.worst_obstructions = max(report.worst_obstructions,
                                            len(arm.obstructions()))
        if used >= budget:
            report.stalled = True
        for _ in range(settle):
            command(arm.clamp(target))
            report.ticks += 1
    report.joint_error = float(np.linalg.norm(arm.joints() - points[-1]))
    return report


# --- what is still open --------------------------------------------------
#
# Reaching is solved; grasping is not. With a planned, followed approach the
# gripper sits 0.7 mm above the object, and a second solve-and-follow descends
# to within about 3 cm of it, but the fingers do not close on the object.
#
# The likely cause is orientation, which nothing above constrains: `solve_posture`
# uses the position Jacobian only, so a solved posture is free to rotate the
# gripper away from the top-down approach the reset posture starts in — the grip
# site's z axis points within 8 degrees of straight down at reset.
#
# A first attempt at a combined position-and-orientation solve made things worse
# rather than better: weighting the rotational error at 0.6 against position in
# one damped solve left a 0.41 m position error, against 0.0008 m for the
# position-only solve, and did not achieve the orientation either. The solve is
# ill-conditioned that way. A two-stage approach — solve position, then rotate
# within the nullspace of the position task — is the next thing to try, and is
# recorded here rather than guessed at again.


# --- chaining stages reliably ---------------------------------------------
#
# One solve-plan-follow to a single target is repeatable. Three in sequence were
# not: the same pipeline produced a 0.0012 m final distance on one invocation
# and 0.4865 m on the next, because every stage begins from wherever the last
# one left the arm, which changes both the IK seed and the tree the planner
# grows from. A stage that quietly fails hands a bad posture onward and the
# next stage plans from somewhere unintended.
#
# `reach` closes that by validating each stage against the thing that actually
# matters — where the gripper ended up — and retrying from the arm's real
# posture when it did not arrive. `FollowReport.stalled` was already the signal
# and was simply never checked.

#: Hand-position error, in metres, at which a stage counts as arrived.
STAGE_TOLERANCE = 0.01
#: Attempts per stage before giving up. Each retry re-solves and re-plans from
#: the arm's current posture with a fresh seed, so a retry is a genuinely
#: different attempt rather than the same one repeated.
STAGE_ATTEMPTS = 4


@dataclass
class StageOutcome:
    """One stage of a chained reach, and why it ended as it did."""

    target: object = None
    attempts: int = 0
    arrived: bool = False
    residual_m: float = 0.0
    tilt_deg: float = 0.0
    obstructions: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {"attempts": self.attempts, "arrived": self.arrived,
                "residual_m": round(self.residual_m, 5),
                "tilt_deg": round(self.tilt_deg, 2),
                "obstructions": self.obstructions, "reason": self.reason}


def reach(arm: ArmModel, target, *, command, approach=None, tolerance: float = STAGE_TOLERANCE,
          attempts: int = STAGE_ATTEMPTS, restarts: int | None = None,
          planner_iterations: int = 6000) -> StageOutcome:
    """Solve, plan and follow to one Cartesian target, retrying on failure.

    Validation is on the *gripper's* position rather than on joint error,
    because a stage can land its joint target while the hand is somewhere else
    — that was how a silent failure propagated. A retry re-solves and re-plans
    from the arm's current posture with a new seed, which is what makes it a
    different attempt rather than a repeat of the same one.
    """
    if type(tolerance) is not float or not 0.0 < tolerance <= 0.5:
        raise ValueError("tolerance must be a float in (0, 0.5]")
    if type(attempts) is not int or not 1 <= attempts <= 20:
        raise ValueError("attempts must be an int from 1 through 20")
    target = np.asarray(target, dtype=float)
    budget = restarts if restarts is not None else (
        ORIENTATION_RESTARTS if approach is not None else 30)
    outcome = StageOutcome(target=target)
    for attempt in range(1, attempts + 1):
        outcome.attempts = attempt
        solution = solve_posture(arm, target, restarts=budget, seed=attempt - 1,
                                 approach=approach)
        if not solution.reached:
            outcome.reason = f"no posture reaches the target (residual {solution.residual_m:.4f})"
            continue
        if not solution.clear:
            outcome.reason = f"solved posture is in {solution.obstructions} contact(s)"
            continue
        path = plan_path(arm, solution.joints, seed=attempt - 1,
                         iterations=planner_iterations)
        if not path.found:
            outcome.reason = "no collision-free path to the solved posture"
            continue
        report = follow_path(arm, path.waypoints, command=command)
        residual = float(np.linalg.norm(target - arm.hand()))
        if report.stalled:
            outcome.reason = f"following stalled with {residual:.4f} m remaining"
            continue
        if residual > tolerance:
            outcome.reason = f"followed the path but ended {residual:.4f} m away"
            continue
        outcome.arrived = True
        outcome.residual_m = residual
        outcome.tilt_deg = float(np.degrees(arm.tilt_from(DOWN)))
        outcome.obstructions = report.worst_obstructions
        outcome.reason = ""
        return outcome
    outcome.residual_m = float(np.linalg.norm(target - arm.hand()))
    outcome.tilt_deg = float(np.degrees(arm.tilt_from(DOWN)))
    return outcome
