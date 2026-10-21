"""A working scripted reference: the pick, as a harness method.

Every arm in this lane scored zero until this existed, which made the metrics
unreadable — a success rate and a partial score cannot distinguish policies when
none of them can engage the object. This method reaches, grasps and lifts, so a
model policy finally has something to be read against.

It is a *reference*, not a policy under test, and the distinction is the same one
`study_scripted` draws: it reads the object's pose straight out of the simulator,
which no policy in the study may do. What it does not get is any relief from the
rules a policy faces — its actions go through the same `SafetyEnvelope`, the same
normalised bound, and the same one-action-per-control-step dispatch.

**Why joint deltas rather than absolute targets.** The envelope bounds every
action element to 1.0 and a joint target is in radians up to about 3, so an
absolute target would be rejected outright. `joint_delta` is a normalised step on
the declared `JOINT_DELTA_SCALE`, which keeps the policy inside the same bound
every other method is held to.

**Why horizon 1 by default.** The approach needs the commanded point held near
the arm — that gating is what makes executed motion follow a collision-checked
path, and getting it wrong was the single hardest bug in this lane. The harness
dispatches a whole chunk before calling `predict` again, so a horizon of n means
the gate is only consulted every n control steps. At 1 the method sees every
step. A longer horizon is allowed and is expected to track worse, which makes
the study's action-horizon axis measurable on a policy that actually works
rather than on two that score zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from examples.advanced.openpi_policy.common import ActionChunk, PolicyObservation

from .study_ik import (
    DOWN, FOLLOW_INCREMENT, FOLLOW_SETTLE, FOLLOW_TOLERANCE, ORIENTATION_RESTARTS, ArmModel,
    plan_path, solve_posture)
from .study_representations import JOINT_DELTA_SCALE

OPEN, CLOSE = -1.0, 1.0
#: Height above the object to stage the approach, in metres.
APPROACH_HEIGHT = 0.085
#: Height at which the fingers straddle the object. Measured: the working band
#: is 0.6 to 1.2 cm, 1.8 cm only brushes the top, and 0 has no clear solution.
GRASP_HEIGHT = 0.012
#: Height to lift to, above the object's starting height.
LIFT_HEIGHT = 0.16
#: Control steps spent closing the gripper before trusting the grasp.
CLOSE_STEPS = 260
#: Solve/plan attempts per phase. A retry re-solves from the arm's current
#: posture with a new seed, which is what makes it a different attempt rather
#: than a repeat; the lift in particular needs it, because clear postures get
#: scarce near the object and seed 0 is not reliably one of them.
PLAN_ATTEMPTS = 6
#: RRT iterations per attempt, matching what the chained stages needed.
PLANNER_ITERATIONS = 6000
#: Phases, in order. `done` holds position so the harness can run out its
#: budget without the method inventing motion.
PHASES = ("approach", "descend", "close", "lift", "done")


@dataclass
class ScriptedPickMethod:
    """Reach, grasp and lift, emitting normalised joint deltas.

    `environment` is the `RoboCasaPnPEnvironment` being driven; holding it is
    the privileged channel, explicit here rather than smuggled through the
    observation.
    """

    environment: object
    horizon: int = 1
    source: str = "scripted"
    grasp_height: float = GRASP_HEIGHT
    lift_height: float = LIFT_HEIGHT
    phase: str = "approach"
    calls: int = 0
    closed_for: int = 0
    history: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    _arm: object = None
    _path: list = field(default_factory=list)
    _leg: int = 0
    _travelled: float = 0.0
    _settling: int = 0
    _start_height: float | None = None

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or not 1 <= self.horizon <= 32:
            raise ValueError("horizon must be an int from 1 through 32")
        for name in ("grasp_height", "lift_height"):
            value = getattr(self, name)
            if type(value) is not float or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be a float in (0, 1]")

    # --- privileged readings -------------------------------------------

    @property
    def arm(self) -> ArmModel:
        if self._arm is None or self._arm.env is not self.environment.env:
            self._arm = ArmModel(env=self.environment.env)
        return self._arm

    def _object_height(self) -> float:
        if self._start_height is None:
            self._start_height = float(self._object_position()[2])
        return self._start_height

    def _object_position(self):
        import mujoco

        env = self.environment.env
        model, data = env.sim.model._model, env.sim.data._data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                 env.objects["obj"].root_body)
        return np.asarray(data.xpos[body]).copy()

    def _grasped(self) -> bool:
        env = self.environment.env
        return bool(env._check_grasp(gripper=env.robots[0].gripper["right"],
                                    object_geoms=env.objects["obj"].contact_geoms))

    # --- the phase machine ---------------------------------------------

    def _target_for(self, phase: str):
        position = self._object_position()
        if phase == "approach":
            return position + np.array([0.0, 0.0, APPROACH_HEIGHT])
        if phase == "descend":
            return position + np.array([0.0, 0.0, self.grasp_height])
        if phase == "lift":
            hand = self.arm.hand()
            return np.array([hand[0], hand[1], self._object_height() + self.lift_height])
        return None

    def _plan_for(self, phase: str) -> bool:
        """Solve and plan to the phase's target. False if every attempt fails.

        Each failure is recorded rather than swallowed: a phase that silently
        does not happen is indistinguishable from one that happened badly, and
        telling those apart is most of debugging a policy like this.
        """
        target = self._target_for(phase)
        if target is None:
            self.failures.append({"phase": phase, "reason": "no target"})
            return False
        # The lift keeps the grasp, so it may not request a top-down approach —
        # re-orienting a held object is how it gets dropped.
        approach = None if phase == "lift" else DOWN
        restarts = 30 if approach is None else ORIENTATION_RESTARTS
        for attempt in range(PLAN_ATTEMPTS):
            solution = solve_posture(self.arm, target, restarts=restarts,
                                     seed=attempt, approach=approach)
            if not solution.reached:
                reason = f"residual {solution.residual_m:.4f} m"
            elif not solution.clear:
                reason = f"{solution.obstructions} contact(s)"
            else:
                path = plan_path(self.arm, solution.joints, seed=attempt,
                                 iterations=PLANNER_ITERATIONS)
                if path.found:
                    self._path = list(path.waypoints)
                    self._leg, self._travelled = 1, 0.0
                    return True
                reason = "no collision-free path"
            self.failures.append({"phase": phase, "attempt": attempt, "reason": reason})
        return False

    def _commanded_point(self):
        """The next joint target along the current leg, gated on the arm.

        The gate is the whole mechanism: the commanded point advances only once
        the arm is within tolerance of it, so the executed motion stays on the
        line the planner verified rather than taking its own dynamic route.

        Reaching the end of a leg is not the same as arriving at it. The
        command leads the arm by up to the follow tolerance, so a leg that
        advances to its final point still has the arm short of it — and the
        first version of this method moved on immediately, which left the hand
        2.5 cm above the object and closed the fingers on each other. Each leg
        therefore holds its target for a settle window, exactly as `follow_path`
        does, before the next one starts.
        """
        if self._leg >= len(self._path):
            return None
        start, target = self._path[self._leg - 1], self._path[self._leg]
        span = float(np.linalg.norm(target - start))
        if span < 1e-9:
            self._leg += 1
            self._travelled, self._settling = 0.0, 0
            return self._commanded_point()
        if self._travelled >= span:
            self._settling += 1
            if self._settling >= FOLLOW_SETTLE:
                self._leg += 1
                self._travelled, self._settling = 0.0, 0
            return target
        direction = (target - start) / span
        here = start + direction * self._travelled
        if float(np.linalg.norm(self.arm.joints() - here)) < FOLLOW_TOLERANCE:
            self._travelled = min(span, self._travelled + FOLLOW_INCREMENT)
            here = start + direction * self._travelled
        return here

    def _advance(self) -> None:
        """Move to the next phase, planning for it when it needs motion."""
        order = list(PHASES)
        self.phase = order[min(order.index(self.phase) + 1, len(order) - 1)]
        if self.phase in ("approach", "descend", "lift") and not self._plan_for(self.phase):
            # A phase that cannot be planned is skipped rather than retried
            # forever; the episode then ends with whatever rung it reached,
            # which is what the partial score is for.
            self._advance()

    # --- the harness contract ------------------------------------------

    def predict(self, observation: PolicyObservation) -> ActionChunk:
        self.calls += 1
        if self._start_height is None:
            self._start_height = float(self._object_position()[2])
        if not self._path and self.phase == "approach":
            if not self._plan_for("approach"):
                self._advance()
        rows = []
        for _ in range(self.horizon):
            rows.append(self._row())
        actions = np.asarray(rows, dtype=np.float32)
        self.history.append({"call": self.calls, "phase": self.phase,
                             "leg": self._leg, "grasped": self._grasped()})
        return ActionChunk(actions=actions, horizon=self.horizon, dof=8,
                           source=self.source)

    def _row(self):
        """One normalised joint-delta row for the current phase."""
        grip = CLOSE if self.phase in ("close", "lift", "done") else OPEN
        if self.phase == "close":
            self.closed_for += 1
            if self.closed_for >= CLOSE_STEPS:
                self._advance()
            return np.concatenate([np.zeros(7), [CLOSE]])
        if self.phase == "done":
            return np.concatenate([np.zeros(7), [grip]])
        point = self._commanded_point()
        if point is None:
            self._advance()
            return np.concatenate([np.zeros(7), [grip]])
        # Normalise the remaining joint motion by the declared scale, so one
        # action commands at most one scale-step and stays inside the bound the
        # envelope enforces on every method.
        delta = (point - self.arm.joints()) / JOINT_DELTA_SCALE
        return np.concatenate([np.clip(delta, -1.0, 1.0), [grip]])

    def describe(self) -> dict:
        return {"method": "scripted_pick", "privileged": True, "horizon": self.horizon,
                "grasp_height_m": self.grasp_height, "lift_height_m": self.lift_height,
                "phase": self.phase, "calls": self.calls}
