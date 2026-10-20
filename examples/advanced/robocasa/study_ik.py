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
