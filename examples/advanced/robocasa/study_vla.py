"""A released vision-language-action policy as a study method.

Every method in this study so far was written here: two baselines that score
zero and a privileged scripted reference that scores 75. Neither tells you what
a *trained* policy does on these tasks, and that is the number the study needs
to be read against. `lerobot/smolvla_robocasa` is a SmolVLA checkpoint
fine-tuned on `lerobot/robocasa_target_human_unified` — 25,307 episodes over 50
RoboCasa tasks — and five of its tasks are in this study's committed suite.

**Why it fits without reinterpretation.** The policy emits a 12-wide action and
this robot takes a 12-wide action, but the orderings differ, and feeding one to
the other unpermuted would look like it was working while commanding the base
with end-effector deltas. The mapping below is not taken on trust from a
README: it was derived from the dataset's own per-dimension statistics and then
cross-checked against this environment's action slices.

The giveaways were the degenerate dimensions. In 15 million frames, action
dimension 3 is *exactly* zero — a frozen torso. Dimensions 4 and 11 are exactly
binary: a ±1 variable satisfies `std = sqrt(1 - mean**2)`, and dimension 4's
measured −0.780 / 0.626 and dimension 11's −0.172 / 0.985 both match to three
decimals, which no continuous signal would. That identifies the mode flag and
the gripper, and the remaining translation/rotation split falls out of the
magnitudes: dimensions 5–7 have standard deviations near 0.37 against 0.11 for
8–10, and a position delta moves more than a rotation delta.

The state vector was settled the same way and then confirmed by physics: the
16 values are this robot's `base_pos`, `base_quat`, `eef_pos`, `eef_quat` and
`gripper_qpos` concatenated in that order, which is why state dimensions 3 and 4
are identically zero (a base that only yaws has no quaternion x or y) and
dimension 2 sits at 0.70 with a standard deviation of 0.002 (a base of fixed
height).

**What this measures and what it does not.** This is a comparable number on the
same task, not a published one: the checkpoint's model card reports no scores,
so what comes out is our measurement of a public policy under our scorer. It is
honest about being a model — `source` is `"model"`, not `"scripted"` — so the
safety envelope has to be widened deliberately to admit it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from examples.advanced.openpi_policy.common import ActionChunk, PolicyObservation

#: The checkpoint this module is written against.
DEFAULT_CHECKPOINT = "lerobot/smolvla_robocasa"
#: The dataset it was fine-tuned on, recorded so a run says what it faced.
TRAINING_DATASET = "lerobot/robocasa_target_human_unified"
#: Camera views the policy was trained with, in the order its config names them.
#: `agentview_right` is a RoboCasa-only role, so an observation configured for
#: this policy reports `portable: False` — DROID has no second agentview.
LEROBOT_VIEWS = ("over_shoulder", "agentview_right", "wrist")
#: The feature keys its preprocessor expects for those views.
LEROBOT_IMAGE_KEYS = ("observation.images.robot0_agentview_left",
                      "observation.images.robot0_agentview_right",
                      "observation.images.robot0_eye_in_hand")
#: Robosuite observables that concatenate, in order, into the 16-wide state.
LEROBOT_STATE_KEYS = ("robot0_base_pos", "robot0_base_quat", "robot0_eef_pos",
                      "robot0_eef_quat", "robot0_gripper_qpos")
#: Width of that state, asserted rather than assumed.
LEROBOT_STATE_DOF = 16
#: Width of one action row.
LEROBOT_ACTION_DOF = 12
#: Native image side the checkpoint's config declares.
LEROBOT_IMAGE_SIZE = 256

#: Where each native action element comes from in a LeRobot row.
#:
#: Native layout is `eef_pos(3) | eef_rot(3) | gripper | base(3) | torso | mode`
#: and the LeRobot row is `base(3) | torso | mode | eef_pos(3) | eef_rot(3) |
#: gripper`, so this permutation is the whole conversion.
NATIVE_FROM_LEROBOT = (5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4)
#: Dimensions the training data never moved. Commanding them would be acting
#: outside the distribution the policy was fitted on, so a run reports whether
#: the policy tried.
FROZEN_LEROBOT_DIMS = (3,)


class VlaUnavailable(RuntimeError):
    """The checkpoint or its dependencies could not be loaded."""


def to_native_action(row) -> np.ndarray:
    """Permute one LeRobot action row into this environment's ordering."""
    values = np.asarray(row, dtype=np.float32)
    if values.shape != (LEROBOT_ACTION_DOF,):
        raise ValueError(f"a lerobot action row must be {LEROBOT_ACTION_DOF} wide")
    if not np.isfinite(values).all():
        raise ValueError("a lerobot action row contains a non-finite value")
    return values[list(NATIVE_FROM_LEROBOT)]


def lerobot_state(raw) -> np.ndarray:
    """Concatenate this robot's observables into the policy's 16-wide state."""
    parts = []
    for key in LEROBOT_STATE_KEYS:
        if key not in raw:
            raise KeyError(f"observation is missing {key!r}")
        parts.append(np.asarray(raw[key], dtype=np.float32).reshape(-1))
    state = np.concatenate(parts)
    if state.shape != (LEROBOT_STATE_DOF,):
        raise ValueError(f"state assembled to {state.shape}, expected "
                         f"({LEROBOT_STATE_DOF},)")
    if not np.isfinite(state).all():
        raise ValueError("assembled state contains a non-finite value")
    return state


def untile(image, *, views: int = len(LEROBOT_VIEWS)) -> list:
    """Recover the separate camera views from one tiled observation image.

    `tile_frames` lays views out along the width, so slicing the width into
    equal parts inverts it exactly. Going through the tiled image rather than
    around it keeps the policy on the same observation contract every other
    method uses, instead of reaching into the environment for raw frames.
    """
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("a tiled observation must be (H, W, 3)")
    if type(views) is not int or views < 1:
        raise ValueError("views must be a positive int")
    height, width = frame.shape[:2]
    if width % views:
        raise ValueError(f"a width of {width} does not divide into {views} views")
    side = width // views
    if side != height:
        raise ValueError(f"each view must be square, got {height}x{side}")
    return [np.ascontiguousarray(frame[:, i * side:(i + 1) * side]) for i in range(views)]


@dataclass
class SmolVlaMethod:
    """`lerobot/smolvla_robocasa` behind the study's method contract.

    The policy plans its own chunk — `n_action_steps` is 50 in the released
    config — and this holds that queue, handing the harness `horizon` rows per
    call. So one model call covers up to 50 control steps, and the study's
    model-call count stays the honest measure of what the policy cost.
    """

    checkpoint: str = DEFAULT_CHECKPOINT
    horizon: int = 1
    source: str = "model"
    device: str = "cpu"
    policy: object = None
    calls: int = 0
    inferences: int = 0
    frozen_commands: int = 0
    _queue: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or not 1 <= self.horizon <= 50:
            raise ValueError("horizon must be an int from 1 through 50")
        if not isinstance(self.checkpoint, str) or not self.checkpoint.strip():
            raise ValueError("checkpoint must be a non-empty string")

    def load(self) -> None:
        """Build the policy. Separate from `__init__` so import stays cheap."""
        if self.policy is not None:
            return
        try:
            import torch
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise VlaUnavailable(f"lerobot/transformers unavailable: {error}") from error
        try:
            policy = SmolVLAPolicy.from_pretrained(self.checkpoint)
        except Exception as error:  # pragma: no cover - network-dependent
            raise VlaUnavailable(f"could not load {self.checkpoint}: {error}") from error
        policy.eval()
        policy.to(torch.device(self.device))
        self.policy = policy

    def reset(self) -> None:
        self._queue.clear()
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    # --- the harness contract ------------------------------------------

    def predict(self, observation: PolicyObservation) -> ActionChunk:
        self.calls += 1
        rows = []
        for _ in range(self.horizon):
            if not self._queue:
                self._queue.extend(self._infer(observation))
            rows.append(self._queue.pop(0))
        actions = np.asarray(rows, dtype=np.float32)
        self.frozen_commands += int(
            np.count_nonzero(actions[:, list(FROZEN_LEROBOT_DIMS)]))
        return ActionChunk(actions=actions, horizon=self.horizon,
                           dof=LEROBOT_ACTION_DOF, source=self.source)

    def _infer(self, observation: PolicyObservation) -> list:
        """One forward pass, returning the policy's whole action chunk."""
        import torch

        self.load()
        self.inferences += 1
        batch = self._batch(observation)
        with torch.no_grad():
            if hasattr(self.policy, "predict_action_chunk"):
                chunk = self.policy.predict_action_chunk(batch)
            else:  # pragma: no cover - depends on the lerobot version
                chunk = self.policy.select_action(batch).unsqueeze(1)
        values = np.asarray(chunk.squeeze(0).float().cpu().numpy(), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] < LEROBOT_ACTION_DOF:
            raise ValueError(f"policy returned {values.shape}, expected "
                             f"(steps, >={LEROBOT_ACTION_DOF})")
        # The checkpoint pads to `max_action_dim`; only the declared width is
        # the robot's, and the padding is not ours to command.
        return [np.clip(row[:LEROBOT_ACTION_DOF], -1.0, 1.0) for row in values]

    def _batch(self, observation: PolicyObservation) -> dict:
        """Shape one observation the way the policy's preprocessor wants it."""
        import torch

        views = untile(observation.image)
        state = np.asarray(observation.state, dtype=np.float32)
        if state.shape != (LEROBOT_STATE_DOF,):
            raise ValueError(f"the policy needs a {LEROBOT_STATE_DOF}-wide state, "
                             f"got {state.shape}; is the environment in lerobot mode?")
        batch = {"observation.state": torch.from_numpy(state).unsqueeze(0),
                 "task": [observation.prompt]}
        for key, view in zip(LEROBOT_IMAGE_KEYS, views):
            # Channels-first float in [0, 1], which is what the normaliser's
            # IDENTITY mapping for visual features assumes.
            image = torch.from_numpy(view.astype(np.float32) / 255.0)
            batch[key] = image.permute(2, 0, 1).unsqueeze(0)
        return batch

    def describe(self) -> dict:
        return {"method": "smolvla", "checkpoint": self.checkpoint,
                "training_dataset": TRAINING_DATASET, "privileged": False,
                "horizon": self.horizon, "device": self.device,
                "source": self.source, "calls": self.calls,
                "inferences": self.inferences,
                "frozen_commands": self.frozen_commands}
