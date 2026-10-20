"""A privileged scripted reference for pick-and-place.

This is a *reference*, not a policy under test, and the distinction is load
bearing. It reads the object's pose straight out of the simulator, which no
policy in the study may do — `PolicyObservation` deliberately carries only
images and the robot's own proprioception. Its job is to establish that the task
is solvable under the study's action caps and that the scorer fires on real
progress, so that a model policy's number has something to be read against. The
Astranomy study uses a trained base policy (Cosmos3-Nano) for the analogous
role; a waypoint script is what is available here.

It is also what makes the horizon axis measurable, because it plans a *chunk* of
future actions toward the current waypoint rather than one step at a time.

The method emits intent, not motion. A waypoint delta is handed over whole and
the representation codec clips it to the study's per-step cap, so the script
cannot evade a limit a model policy is held to — which is why a 0.5 m approach
takes about twenty-six control steps rather than one.

Phases are explicit and ordered, and the method reports which one it is in, so a
trace shows where a failure happened rather than only that the episode failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from examples.advanced.openpi_policy.common import ActionChunk, PolicyObservation

from .study_representations import normalised_dof

#: Height above the object to stage the approach, in metres.
APPROACH_CLEARANCE = 0.12
#: Height above the grasp to lift to, in metres.
LIFT_CLEARANCE = 0.25
#: Distance at which a waypoint counts as reached, in metres.
WAYPOINT_TOLERANCE = 0.015
#: Control steps to hold the gripper closed before trusting the grasp.
GRASP_HOLD_STEPS = 8
#: Gripper commands, in RoboSuite's convention.
OPEN, CLOSE = -1.0, 1.0

PHASES = ("approach_above", "descend", "grasp", "lift", "transport", "release", "done")


@dataclass
class PrivilegedScriptedPnP:
    """Reach, grasp, lift and release, driven by privileged object pose.

    `environment` is the `RoboCasaPnPEnvironment` being driven. Holding a
    reference to it is the privileged channel, kept explicit here rather than
    smuggled through the observation.
    """

    environment: object
    representation: str = "eef_delta"
    horizon: int = 4
    source: str = "scripted"
    place_target: tuple | None = None
    calls: int = 0
    phase: str = "approach_above"
    held_steps: int = 0
    grasp_height: float | None = None
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.representation != "eef_delta":
            raise ValueError("the scripted reference drives end-effector deltas")
        if type(self.horizon) is not int or not 1 <= self.horizon <= 32:
            raise ValueError("horizon must be an int from 1 through 32")

    # --- privileged readings -------------------------------------------

    def _object_position(self) -> np.ndarray:
        env = self.environment.env
        body = env.sim.model.body_name2id(env.objects["obj"].root_body)
        return np.asarray(env.sim.data.body_xpos[body], dtype=np.float64)

    def _gripper_position(self) -> np.ndarray:
        return np.asarray(self.environment._eef_position(), dtype=np.float64)

    def _grasped(self) -> bool:
        env = self.environment.env
        target = env.objects["obj"]
        return bool(env._check_grasp(gripper=env.robots[0].gripper["right"],
                                    object_geoms=target.contact_geoms))

    # --- the phase machine ---------------------------------------------

    def _waypoint(self):
        """Return `(target_xyz, gripper)` for the current phase, or None when done."""
        obj = self._object_position()
        gripper = self._gripper_position()
        if self.phase == "approach_above":
            target = obj + np.array([0.0, 0.0, APPROACH_CLEARANCE])
            if np.linalg.norm(gripper - target) <= WAYPOINT_TOLERANCE:
                self.phase = "descend"
            return target, OPEN
        if self.phase == "descend":
            target = obj.copy()
            if np.linalg.norm(gripper - target) <= WAYPOINT_TOLERANCE * 2:
                self.phase, self.held_steps = "grasp", 0
            return target, OPEN
        if self.phase == "grasp":
            self.held_steps += 1
            if self._grasped() and self.held_steps >= GRASP_HOLD_STEPS:
                self.phase = "lift"
                self.grasp_height = float(obj[2])
            return gripper, CLOSE
        if self.phase == "lift":
            target = np.array([gripper[0], gripper[1], (self.grasp_height or obj[2]) + LIFT_CLEARANCE])
            if gripper[2] >= target[2] - WAYPOINT_TOLERANCE:
                self.phase = "transport" if self.place_target is not None else "done"
            return target, CLOSE
        if self.phase == "transport":
            destination = np.asarray(self.place_target, dtype=np.float64)
            target = np.array([destination[0], destination[1], gripper[2]])
            if np.linalg.norm(gripper[:2] - destination[:2]) <= WAYPOINT_TOLERANCE * 3:
                self.phase = "release"
            return target, CLOSE
        if self.phase == "release":
            self.phase = "done"
            return gripper, OPEN
        return gripper, OPEN

    def predict(self, observation: PolicyObservation) -> ActionChunk:
        """Plan `horizon` capped steps toward the current waypoint.

        The waypoint is recomputed once per call, not once per step: that is
        what a chunk *is*, and it is why a longer horizon reacts later to the
        object having moved.
        """
        self.calls += 1
        dof = normalised_dof(self.representation)
        target, grip = self._waypoint()
        start = self._gripper_position()
        rows = []
        for _ in range(self.horizon):
            # Hand over the whole remaining delta; the codec clips it to the
            # study's per-step cap, so the script obeys the same limit a model
            # policy does.
            rows.append(np.concatenate([target - start, np.zeros(3), [grip]]))
        self.history.append({"call": self.calls, "phase": self.phase,
                             "distance_m": round(float(np.linalg.norm(target - start)), 4)})
        actions = np.asarray(rows, dtype=np.float32)
        # The envelope bounds every element; a long approach would otherwise
        # exceed it before the codec ever sees the row.
        actions[:, :6] = np.clip(actions[:, :6], -1.0, 1.0)
        return ActionChunk(actions=actions, horizon=self.horizon, dof=dof,
                           source=self.source)
