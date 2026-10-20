"""RoboLab-120 behind the same Protocol, for when a GPU is available.

This is the port target. It satisfies the identical
`reset / step / verify / close` contract as `study_robocasa_env`, so the sweep,
the scorer, the telemetry and every method run unchanged against it — that is
the whole point of having put RoboCasa behind a Protocol rather than wiring it
in directly.

**It cannot run on this machine and does not pretend to.** RoboLab requires an
NVIDIA RTX GPU with 48 GB+ VRAM recommended, Ubuntu 22.04+, and Isaac Sim
5.0/5.1 with Isaac Lab 2.2/2.3; there is no CPU fallback and no macOS or Apple
Silicon support, and `--headless` removes the window, not the GPU. So the
imports are deferred and a missing stack raises a message that says which
requirement is unmet rather than a bare `ImportError` three frames deep.

It is written from RoboLab's published policy contract, and it is **untested
against a live instance** — which is stated here rather than discovered by
whoever first runs it. What the contract says:

* A policy subclasses `robolab.eval.InferenceClient` and implements
  `_extract_observation(raw_obs, *, env_id=0)`, `_pack_request(obs, instruction)`,
  `_query_server(request)` and `_unpack_response(response)`, with an optional
  `_postprocess_chunk(chunk)`.
* `raw_obs` carries `image_obs["over_shoulder_left_camera"]` and
  `image_obs["wrist_cam"]`, both `(N, H, W, 3)` uint8 torch tensors, plus
  `proprio_obs["arm_joint_pos"]` `(N, 7)` and `proprio_obs["gripper_pos"]`
  `(N, 1)` float32.
* `_unpack_response` returns `(horizon, action_dim)`; the standard DROID row is
  8-wide, seven joint velocities plus a gripper command, and the base class
  handles chunking.

Two differences from the RoboCasa backend are real and are declared rather than
smoothed over. DROID is a **fixed-base** arm, so there is no base segment and no
`base_mode` flag, and the native representation is `joint_velocity` rather than
`eef_delta` — the backends sit at two points on the study's action-representation
axis. And RoboLab reports its own success predicate per task, so the partial
score here comes from the same ladder only where the equivalent privileged
readings are available; where they are not, the score is the predicate alone and
says so.

The inverse mapping — this adapter drives RoboLab *through* the local harness —
is deliberate. The alternative, implementing `InferenceClient` directly, would
put RoboLab's control loop in charge and leave the study's budgets, scoring and
telemetry outside it. `inference_client_adapter` is provided for the case where
RoboLab must drive instead, so a method written here can be served to a real
RoboLab run without being rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .study_observation import policy_observation
from .study_progress import ProgressTracker
from .study_representations import UnsupportedRepresentation

#: Camera roles, named as RoboLab's DROID registration names them.
CAMERA_ROLES = {
    "over_shoulder": "over_shoulder_left_camera",
    "wrist": "wrist_cam",
}
#: Representations DROID's registration expresses natively.
SUPPORTED_REPRESENTATIONS = ("joint_velocity", "joint_position")
#: Seven joint commands plus one gripper command.
EXPOSED_DOF = 8

_REQUIREMENTS = (
    "RoboLab needs an NVIDIA RTX GPU (48GB+ VRAM recommended), Ubuntu 22.04+, "
    "and Isaac Sim 5.0/5.1 with Isaac Lab 2.2/2.3. There is no CPU fallback, and "
    "macOS and Apple Silicon are unsupported. Install from "
    "https://github.com/NVlabs/RoboLab and run this backend there; the RoboCasa "
    "backend is the CPU-runnable substitute."
)


class RoboLabUnavailable(RuntimeError):
    """The RoboLab stack is not installed or not supported on this machine."""


def require_robolab():
    """Import RoboLab or explain exactly which requirement is unmet."""
    try:
        import robolab  # noqa: F401
    except ImportError as error:
        raise RoboLabUnavailable(f"{error}. {_REQUIREMENTS}") from error
    return robolab


@dataclass
class RoboLabEnvironment:
    """A RoboLab-120 task as a harness environment.

    Untested against a live instance; see the module docstring.
    """

    task: str = "RubiksCube"
    representation: str = "joint_velocity"
    env_id: int = 0
    views: tuple = ("over_shoulder",)
    history: int = 0
    env: object = None
    progress: ProgressTracker = field(default_factory=ProgressTracker)
    prompt: str = ""
    _frames: list = field(default_factory=list)
    _last_raw: object = None
    _last_success: bool = False

    def __post_init__(self) -> None:
        if self.representation not in SUPPORTED_REPRESENTATIONS:
            raise UnsupportedRepresentation(
                f"the RoboLab backend cannot express {self.representation!r}; DROID's "
                f"registration is joint-space, so it supports "
                f"{', '.join(SUPPORTED_REPRESENTATIONS)}. `eef_delta` is native to the "
                "RoboCasa backend instead.")
        if any(view not in CAMERA_ROLES for view in self.views) or not self.views:
            raise ValueError(f"views must be a non-empty subset of {tuple(CAMERA_ROLES)}")
        if type(self.history) is not int or not 0 <= self.history <= 8:
            raise ValueError("history must be an int from 0 through 8")

    # --- the harness Protocol ------------------------------------------

    def reset(self, request):
        robolab = require_robolab()
        self.close()
        # RoboLab's own entry point owns scene construction; this is the one
        # line that will need checking against a live install, because the
        # published contract documents the policy side rather than this one.
        self.env = robolab.make_env(self.task, seed=request.seed)  # type: ignore[attr-defined]
        raw = self.env.reset()
        self.progress = ProgressTracker()
        self._frames = []
        self._last_success = False
        self.prompt = str(getattr(self.env, "instruction", "") or self.task)
        return self._observation(raw)

    def step(self, action):
        from .method_harness import Transition

        row = np.asarray(action, dtype=np.float32)
        if row.shape != (EXPOSED_DOF,):
            raise ValueError(f"RoboLab rows must have shape ({EXPOSED_DOF},), got {row.shape}")
        raw, reward, done, info = self.env.step(row)
        self._last_raw = raw
        self._last_success = bool(info.get("success", False)) if isinstance(info, dict) else False
        self.progress.succeeded = self.progress.succeeded or self._last_success
        return Transition(observation=self._observation(raw), reward=float(reward),
                          terminated=bool(done) or self._last_success)

    def verify(self):
        from .method_harness import Verification

        return Verification(
            success=self._last_success, reward=float(self.progress.score) / 100.0,
            message=("RoboLab task predicate satisfied" if self._last_success else
                     f"predicate unmet; partial score {self.progress.score}/100"))

    def close(self) -> None:
        environment, self.env = self.env, None
        if environment is not None and hasattr(environment, "close"):
            environment.close()

    # --- observation ----------------------------------------------------

    def _observation(self, raw):
        """Select one environment from the batch and convert to numpy.

        RoboLab batches across parallel environments and hands back torch
        tensors; the harness is single-environment and numpy, which is what
        `_extract_observation` exists to bridge.
        """
        images = raw["image_obs"]
        frame = [np.asarray(_to_numpy(images[CAMERA_ROLES[view]])[self.env_id], dtype=np.uint8)
                 for view in self.views]
        self._frames.append(frame)
        window = self._frames[-(self.history + 1):] if self.history else self._frames[-1:]
        del self._frames[:-(self.history + 1)]
        proprio = raw["proprio_obs"]
        state = np.concatenate([
            _to_numpy(proprio["arm_joint_pos"])[self.env_id].astype(np.float32),
            _to_numpy(proprio["gripper_pos"])[self.env_id].astype(np.float32),
        ])
        return policy_observation(frames=window, state=state, prompt=self.prompt)

    def describe(self) -> dict:
        return {"backend": "robolab", "task": self.task, "robot": "DROID",
                "representation": self.representation, "exposed_dof": EXPOSED_DOF,
                "observation": {"views": list(self.views), "history": self.history},
                "fixed_base": True, "verified_against_live_instance": False}


def _to_numpy(tensor):
    """Accept a torch tensor or anything array-like, without importing torch."""
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    return np.asarray(tensor)


def inference_client_adapter(method, *, representation="joint_velocity"):
    """Serve a local `MethodAdapter` through RoboLab's `InferenceClient`.

    For the case where RoboLab drives and the method is the server, rather than
    the harness driving RoboLab. Built lazily so importing this module never
    requires the RoboLab stack, and kept here so a method written against the
    local contract does not have to be rewritten to run on a GPU box.
    """
    robolab = require_robolab()
    base = robolab.eval.InferenceClient  # type: ignore[attr-defined]

    class _LocalMethodClient(base):
        """Bridge: RoboLab's four hooks onto one `predict` call."""

        def _extract_observation(self, raw_obs, *, env_id=0):
            images = raw_obs["image_obs"]
            frame = [np.asarray(_to_numpy(images[name])[env_id], dtype=np.uint8)
                     for name in CAMERA_ROLES.values() if name in images]
            proprio = raw_obs["proprio_obs"]
            state = np.concatenate([
                _to_numpy(proprio["arm_joint_pos"])[env_id].astype(np.float32),
                _to_numpy(proprio["gripper_pos"])[env_id].astype(np.float32),
            ])
            return {"frames": [frame], "state": state}

        def _pack_request(self, extracted_obs, instruction):
            return policy_observation(frames=extracted_obs["frames"],
                                      state=extracted_obs["state"],
                                      prompt=str(instruction or "complete the task"))

        def _query_server(self, request):
            # No server: the method is local. The hook is kept so the control
            # flow matches RoboLab's, which is what makes this swappable for a
            # genuinely remote policy.
            return method.predict(request)

        def _unpack_response(self, response):
            actions = np.asarray(response.actions, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != EXPOSED_DOF:
                raise ValueError(
                    f"RoboLab expects a (horizon, {EXPOSED_DOF}) chunk, got {actions.shape}")
            return actions

    return _LocalMethodClient
