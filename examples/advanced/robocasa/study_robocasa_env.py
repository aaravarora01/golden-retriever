"""RoboCasa pick-and-place behind the harness's `EnvironmentAdapter` Protocol.

This is the backend that runs today, on CPU, on this machine. It satisfies the
same `reset / step / verify / close` contract as `LiftHarnessEnvironment`, so
`MethodHarness` drives it unchanged, and the same contract the RoboLab backend
satisfies — which is what makes the GPU port an adapter swap rather than a
rewrite.

Three things about RoboCasa that the adapter has to absorb, each discovered by
running it rather than reading about it.

**Registration is an import side effect.** The 38 pick-and-place environments
appear in RoboSuite's registry only once `robocasa` is imported; without it
`robosuite.make` reports the task as unknown among the 19 base environments.

**Layout and style ids are 1-based and must be lists.** `layout_ids=0` raises a
`KeyError` deep inside the scene registry, because 0 is not a `LayoutType`
value — the negative values are group selectors and the concrete layouts start
at 1. Pinning both, plus the seed, is what makes an episode reproducible.

**The action vector is 12-wide and mostly not the arm.** The `PandaOmron`
`HybridMobileBase` controller splits it as arm `0:6` (an operational-space delta
pose), gripper `6:7`, base `7:10`, torso `10:11`, with a trailing mode flag. The
study's policies command an arm, so the base and torso segments are held at zero
and only the arm-plus-gripper slice is exposed. That also makes the exposed
action 7-wide, which is the `eef_delta` representation; RoboLab's native 8-wide
joint-velocity row is the other end of the same axis.

The plain `Panda` robot is not an option here: RoboCasa's kitchen tasks expect
the mobile-base camera rig and fail on a missing `robot0_agentview_center`.

**Freezing the base makes these tasks unsolvable, and that is measured rather
than assumed.** Across the ten pick-and-place tasks tried, the object starts
0.33-0.66 m from the end-effector, and a Cartesian controller driven straight at
it stalls: 0.337 m under `HybridMobileBase` OSC and 0.288 m with the per-step cap
applied correctly, with no joint at a limit. The cause is orientation, not reach
— `init_robot_base_ref` places the base facing the *target* fixture, so the
object sits outside the arm's workspace and closing the gap needs base yaw and
translation. The base segment is therefore exposed, not frozen, whenever
`command_base` is set; a run that leaves it frozen is measuring an arm against a
task that needs a mobile manipulator, which would report a policy failure where
the real limit is the action space it was given.

Two controllers are available and they differ in representation, which is the
study's first axis rather than an implementation detail. The default
`HybridMobileBase` takes a 6-wide operational-space *delta* pose plus a gripper,
which is `eef_delta`. `WHOLE_BODY_IK` instead takes 7 *absolute joint* targets
plus a gripper through the flat action vector, which is `joint_position` and the
nearer analogue of RoboLab's 8-wide joint-velocity row. The config's
`ik_input_ref_frame` is `mobilebase0_base`, so nothing here is a world-frame
pose target.

Imports are deferred into the methods, so this module can be imported — and its
observation plumbing unit-tested — with no simulator installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .study_progress import ProgressTracker
from .study_representations import EefDeltaCodec, UnsupportedRepresentation

#: RoboCasa camera names, mapped to the roles RoboLab's DROID registration uses.
CAMERA_ROLES = {
    "over_shoulder": "robot0_agentview_left",
    "wrist": "robot0_eye_in_hand",
}
#: Representations this backend can express natively; see `study_representations`.
SUPPORTED_REPRESENTATIONS = ("eef_delta", "eef_absolute", "eef_delta_base")
#: Arm-plus-gripper width exposed to a policy.
EXPOSED_DOF = 7
_ARM = slice(0, 6)
_GRIPPER = slice(6, 7)
_BASE = slice(7, 10)


@dataclass(frozen=True)
class ObservationConfig:
    """Which views and how much history a policy receives.

    The study's prompt-geometry axis is which camera views are included, and its
    observation-history axis is how many past frames accompany the current one.
    Both are declared here so a run records what the policy could see.
    """

    views: tuple = ("over_shoulder",)
    history: int = 0
    image_size: int = 128

    def __post_init__(self) -> None:
        if not self.views or any(view not in CAMERA_ROLES for view in self.views):
            raise ValueError(f"views must be a non-empty subset of {tuple(CAMERA_ROLES)}")
        if len(set(self.views)) != len(self.views):
            raise ValueError("views contains a duplicate")
        if type(self.history) is not int or not 0 <= self.history <= 8:
            raise ValueError("history must be an int from 0 through 8")
        if type(self.image_size) is not int or not 32 <= self.image_size <= 512:
            raise ValueError("image_size must be an int from 32 through 512")

    @property
    def camera_names(self) -> list:
        return [CAMERA_ROLES[view] for view in self.views]

    def to_dict(self) -> dict:
        return {"views": list(self.views), "history": self.history,
                "image_size": self.image_size}


@dataclass
class RoboCasaPnPEnvironment:
    """A RoboCasa pick-and-place task as a harness environment."""

    task: str = "PickPlaceCounterToCabinet"
    robot: str = "PandaOmron"
    layout_id: int = 1
    style_id: int = 1
    representation: str = "eef_delta"
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    #: Expose the 3-wide base segment to the policy. Required for any task whose
    #: object starts outside the arm's workspace, which is all of them so far.
    command_base: bool = False
    #: The study's task-specification axis: RoboCasa's own goal string, or the
    #: derived instruction that withholds the object's identity.
    specification_style: str = "default"
    env: object = None
    codec: EefDeltaCodec | None = None
    progress: ProgressTracker = field(default_factory=ProgressTracker)
    prompt: str = ""
    specification: dict = field(default_factory=dict)
    _frames: list = field(default_factory=list)
    _last_success: bool = False
    _last_raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.representation not in SUPPORTED_REPRESENTATIONS:
            raise UnsupportedRepresentation(
                f"the RoboCasa backend cannot express {self.representation!r}; "
                f"it supports {', '.join(SUPPORTED_REPRESENTATIONS)}. A joint "
                "representation is native to the RoboLab backend instead.")
        if type(self.layout_id) is not int or self.layout_id < 1:
            raise ValueError("layout_id must be an int of 1 or more; 0 is not a LayoutType")
        if type(self.style_id) is not int or self.style_id < 1:
            raise ValueError("style_id must be an int of 1 or more")
        from .study_tasks import SPECIFICATION_STYLES

        if self.specification_style not in SPECIFICATION_STYLES:
            raise ValueError(f"specification_style must be one of {SPECIFICATION_STYLES}")

    # --- the harness Protocol ------------------------------------------

    def reset(self, request):
        import robocasa  # noqa: F401 - importing registers the RoboCasa task suite
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        self.close()
        config = load_composite_controller_config(robot=self.robot)
        self.env = robosuite.make(
            self.task, robots=self.robot, controller_configs=config,
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
            camera_names=self.observation.camera_names,
            camera_heights=self.observation.image_size,
            camera_widths=self.observation.image_size,
            seed=request.seed, layout_ids=[self.layout_id], style_ids=[self.style_id],
        )
        raw = self.env.reset()
        self.codec = EefDeltaCodec(self.representation)
        self.progress = ProgressTracker()
        self._frames = []
        self._last_success = False
        # The instruction a policy receives is an axis, not a fixed string. The
        # default style is RoboCasa's own goal; `vague` withholds the object
        # identity. The environment's own text is kept as the fallback so a task
        # outside the committed suite still runs.
        from .study_tasks import specification

        try:
            self.specification = specification(self.task, self.specification_style)
        except ValueError:
            self.specification = {"text": str(self.env.get_ep_meta().get("lang") or self.task),
                                  "style": "default", "derived": False, "names_object": True}
        self.prompt = self.specification["text"]
        self._track(raw)
        return self._observation(raw)

    def step(self, action):
        row = self.codec.to_delta(
            action,
            current_position=self._eef_position(),
            current_rotation=self._eef_rotation(),
        )
        full = np.zeros(self.env.action_dim, dtype=np.float32)
        full[_ARM] = row[:6]
        full[_GRIPPER] = row[6]
        if self.command_base:
            full[_BASE] = row[7:10]
            # The trailing element is `base_mode`: above zero drives the base and
            # updates the arm goal from its *desired* pose, at or below zero
            # drives the arm from its *achieved* pose. Deriving it from whether a
            # base command was given keeps a policy from having to learn a flag.
            full[-1] = 1.0 if np.any(np.abs(row[7:10]) > 1e-6) else -1.0
        else:
            full[-1] = -1.0
        raw, reward, done, _ = self.env.step(full)
        self._track(raw)
        from .method_harness import Transition

        return Transition(observation=self._observation(raw), reward=float(reward),
                          terminated=bool(done) or self._last_success)

    def verify(self):
        from .method_harness import Verification

        success = self._last_success
        return Verification(
            success=success, reward=float(self.progress.score) / 100.0,
            message=("RoboCasa task predicate satisfied" if success else
                     f"predicate unmet; partial score {self.progress.score}/100"))

    def close(self) -> None:
        environment, self.env = self.env, None
        if environment is not None:
            environment.close()

    # --- observation and scoring ---------------------------------------

    def _observation(self, raw):
        from .study_observation import policy_observation

        images = [np.asarray(raw[f"{name}_image"], dtype=np.uint8)
                  for name in self.observation.camera_names]
        self._frames.append(images)
        if self.observation.history:
            # Oldest first, current last, so a policy reads the frames in time
            # order whatever the configured depth.
            window = self._frames[-(self.observation.history + 1):]
        else:
            window = self._frames[-1:]
        del self._frames[:-(self.observation.history + 1)]
        return policy_observation(
            frames=window,
            state=self._proprioception(raw),
            prompt=self.prompt,
        )

    def _proprioception(self, raw) -> np.ndarray:
        """Joint positions, gripper opening and end-effector pose.

        Mirrors RoboLab's DROID registration, which gives `arm_joint_pos` (7,)
        and a single `gripper_pos`, with the end-effector pose appended because
        an absolute end-effector representation needs it.
        """
        gripper = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)
        return np.concatenate([
            np.asarray(raw["robot0_joint_pos"], dtype=np.float32),
            np.asarray([float(np.sum(np.abs(gripper)))], dtype=np.float32),
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
        ])

    def _eef_position(self):
        # Read from the latest observation rather than the simulator's private
        # observables: the pose is already public there, and an absolute
        # representation must be converted against the same frame the policy saw.
        return np.asarray(self._last_raw["robot0_eef_pos"], dtype=np.float32)

    def _eef_rotation(self):
        from robosuite.utils.transform_utils import quat2axisangle

        quaternion = np.asarray(self._last_raw["robot0_eef_quat"], dtype=np.float64)
        return np.asarray(quat2axisangle(quaternion), dtype=np.float32)

    def _track(self, raw) -> None:
        """Update the partial score from privileged state.

        Privileged on purpose, and never placed in a `PolicyObservation`: a
        scorer may read the object's pose, a policy may not.
        """
        self._last_raw = raw
        target = self.env.objects["obj"]
        object_position = self.env.sim.data.body_xpos[
            self.env.sim.model.body_name2id(target.root_body)]
        grasped = bool(self.env._check_grasp(
            gripper=self.env.robots[0].gripper["right"],
            object_geoms=target.contact_geoms))
        self._last_success = bool(self.env._check_success())
        self.progress.observe(
            gripper_position=self._eef_position(), object_position=object_position,
            grasped=grasped, success=self._last_success)

    # --- what a run records about this backend -------------------------

    def describe(self) -> dict:
        return {"backend": "robocasa", "task": self.task, "robot": self.robot,
                "layout_id": self.layout_id, "style_id": self.style_id,
                "representation": self.representation, "exposed_dof": EXPOSED_DOF,
                "observation": self.observation.to_dict(),
                "command_base": self.command_base,
                "specification": dict(self.specification),
                "frozen_segments": ["torso"] if self.command_base else ["base", "torso"]}
