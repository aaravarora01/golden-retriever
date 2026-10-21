"""Action representations: the study's first ablation axis.

The Astranomy study design ablates *how* a policy expresses an action before it
ablates anything else — absolute, delta or relative end-effector poses, the same
three for joint positions, and B-spline coefficients. Everything downstream
(horizon, observation content, prompt geometry) is measured under one of these,
so the representation has to be an explicit, named choice rather than whatever
the simulator happens to accept.

It is also the axis where the two backends genuinely differ, which is what makes
the GPU port meaningful rather than cosmetic. RoboCasa's kitchen tasks run a
`HybridMobileBase` controller whose arm segment is an operational-space *delta*
pose, so `eef_delta` is native there and `eef_absolute` is one subtraction away.
RoboLab's DROID registration takes 7 joint velocities plus a gripper command, so
`joint_velocity` is native there. A representation a backend cannot express is
refused by name at setup rather than silently approximated, because an
approximated representation would be reported under a label it does not match.

Bounds are part of the representation, not an afterthought. The study caps
per-step motion — "translation targets are limited to 0.02 m per row and
rotation changes to 0.1 rad per row" — and a policy that exceeds a cap has its
action clipped and the clipping recorded, so a policy that constantly saturates
is visible instead of looking merely slow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The study's per-row motion caps, in metres and radians.
MAX_TRANSLATION_PER_STEP = 0.02
MAX_ROTATION_PER_STEP = 0.1

#: Every representation the study names. A backend declares which it supports.
REPRESENTATIONS = (
    "eef_delta",      # (dx, dy, dz, drx, dry, drz, gripper)
    "eef_absolute",   # (x, y, z, rx, ry, rz, gripper) in the robot root frame
    "joint_velocity", # (dq0..dq6, gripper)
    "joint_position", # (q0..q6, gripper)
    "eef_delta_base", # (dx, dy, dz, drx, dry, drz, gripper, base_x, base_y, base_yaw)
    "joint_delta",    # (dq0..dq6, gripper), each normalised to [-1, 1]
)

#: Width of one action row per representation, gripper command included.
REPRESENTATION_DOF = {
    "eef_delta": 7,
    "eef_absolute": 7,
    "joint_velocity": 8,
    "joint_position": 8,
    "eef_delta_base": 10,
    "joint_delta": 8,
}


class UnsupportedRepresentation(ValueError):
    """A backend was asked for a representation it cannot express."""


@dataclass
class ClipLedger:
    """How often a policy's actions had to be clipped, and by how much.

    Recorded rather than silently applied: a policy whose every action
    saturates the per-step cap is behaving very differently from one that never
    touches it, and a success rate alone cannot tell them apart.
    """

    rows: int = 0
    clipped_rows: int = 0
    max_translation_excess: float = 0.0
    max_rotation_excess: float = 0.0

    def note(self, translation_norm: float, rotation_norm: float) -> None:
        self.rows += 1
        translation_excess = max(0.0, translation_norm - MAX_TRANSLATION_PER_STEP)
        rotation_excess = max(0.0, rotation_norm - MAX_ROTATION_PER_STEP)
        if translation_excess > 0.0 or rotation_excess > 0.0:
            self.clipped_rows += 1
        self.max_translation_excess = max(self.max_translation_excess, translation_excess)
        self.max_rotation_excess = max(self.max_rotation_excess, rotation_excess)

    def to_dict(self) -> dict:
        return {"rows": self.rows, "clipped_rows": self.clipped_rows,
                "clipped_fraction": round(self.clipped_rows / self.rows, 4) if self.rows else None,
                "max_translation_excess_m": round(self.max_translation_excess, 6),
                "max_rotation_excess_rad": round(self.max_rotation_excess, 6)}


def _clip_norm(vector: np.ndarray, limit: float) -> tuple[np.ndarray, float]:
    """Scale a vector to at most `limit`, returning it and its original norm."""
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm == 0.0:
        return vector, norm
    return vector * (limit / norm), norm


@dataclass
class EefDeltaCodec:
    """Translate a policy's representation into an end-effector delta row.

    The delta is what the simulator's operational-space controller consumes, so
    every end-effector representation funnels through here and the per-step caps
    are applied in exactly one place.
    """

    representation: str
    ledger: ClipLedger = field(default_factory=ClipLedger)

    def __post_init__(self) -> None:
        if self.representation not in ("eef_delta", "eef_absolute", "eef_delta_base"):
            raise UnsupportedRepresentation(
                f"{self.representation!r} is not an end-effector representation")

    def to_delta(self, row, *, current_position=None, current_rotation=None) -> np.ndarray:
        """Return `(dx, dy, dz, drx, dry, drz, gripper)` within the caps."""
        values = np.asarray(row, dtype=np.float32)
        if values.shape != (REPRESENTATION_DOF[self.representation],):
            raise ValueError(
                f"{self.representation} rows must have shape "
                f"({REPRESENTATION_DOF[self.representation]},), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("action row contains a non-finite value")
        base = values[7:10] if self.representation == "eef_delta_base" else np.zeros(3, np.float32)
        if self.representation == "eef_absolute":
            if current_position is None or current_rotation is None:
                raise ValueError("an absolute representation needs the current pose")
            translation = values[:3] - np.asarray(current_position, dtype=np.float32)
            rotation = values[3:6] - np.asarray(current_rotation, dtype=np.float32)
        else:
            translation, rotation = values[:3], values[3:6]
        translation, translation_norm = _clip_norm(translation, MAX_TRANSLATION_PER_STEP)
        rotation, rotation_norm = _clip_norm(rotation, MAX_ROTATION_PER_STEP)
        self.ledger.note(translation_norm, rotation_norm)
        # Always 10 wide: the arm delta, the gripper, then the base triple,
        # which is zero for the arm-only representations. One shape downstream
        # means the environment never has to branch on the representation.
        return np.concatenate([translation, rotation, values[6:7], base]).astype(np.float32)


def normalised_dof(representation: str) -> int:
    if representation not in REPRESENTATION_DOF:
        raise UnsupportedRepresentation(
            f"unknown representation {representation!r}; "
            f"available: {', '.join(REPRESENTATIONS)}")
    return REPRESENTATION_DOF[representation]


#: Radians of joint motion one unit of a normalised `joint_delta` commands. The
#: `SafetyEnvelope` bounds every action element to 1.0, and a joint target is in
#: radians up to about 3, so a joint representation has to be a *delta* on a
#: declared scale rather than an absolute target. The scale is a little above
#: the planner's own step so a policy can always command a full step.
JOINT_DELTA_SCALE = 0.05


def joint_targets(current, row, *, scale: float = JOINT_DELTA_SCALE):
    """Turn a normalised joint-delta row into absolute joint targets.

    Returns `(targets, gripper)`. Clipping to the normalised bound happens here
    rather than being assumed of the caller, so a policy that over-commands is
    limited in one place and the limit is the same one the envelope enforces.
    """
    values = np.asarray(row, dtype=np.float32)
    if values.shape != (REPRESENTATION_DOF["joint_delta"],):
        raise ValueError(
            f"joint_delta rows must have shape ({REPRESENTATION_DOF['joint_delta']},), "
            f"got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("action row contains a non-finite value")
    if type(scale) is not float or not 0.0 < scale <= 1.0:
        raise ValueError("scale must be a float in (0, 1]")
    deltas = np.clip(values[:7], -1.0, 1.0) * scale
    return np.asarray(current, dtype=np.float64) + deltas, float(values[7])
