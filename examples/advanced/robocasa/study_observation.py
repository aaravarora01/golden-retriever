"""Build one `PolicyObservation` from several camera views and past frames.

`PolicyObservation` carries a single image, and the study's prompt-geometry and
observation-history axes both need more than one — two camera views, and up to
a few past frames. Rather than widen a payload that `openpi_policy` also uses,
the frames are tiled into one image: time runs down the rows, oldest first, and
views run across the columns. That is also how a multi-image prompt reaches a
model in practice, so the tiling is the representation rather than a workaround.

What is *not* here matters as much. The state vector carries the robot's own
proprioception only — joint positions, gripper opening, end-effector pose. No
object pose, no success flag, nothing the scorer reads. The existing Lift demo
in this folder does put privileged state into its observation, which is fine for
a scripted heuristic and would invalidate a policy study, so this lane does not
follow it.
"""

from __future__ import annotations

import numpy as np

from examples.advanced.openpi_policy.common import PolicyObservation

#: Guard against a tiling that would silently produce an enormous image.
MAX_TILE_PIXELS = 4_000_000


def tile_frames(frames) -> np.ndarray:
    """Tile `[[view, ...], ...]` into one image: rows are time, columns views.

    Every frame must share a shape, which the environment guarantees by
    configuring one image size for all cameras. A single view and no history
    reduces to the frame itself, so the common case costs nothing.
    """
    if not frames or not frames[0]:
        raise ValueError("at least one frame with one view is required")
    widths = {len(row) for row in frames}
    if len(widths) != 1:
        raise ValueError("every frame must carry the same number of views")
    shapes = {np.asarray(image).shape for row in frames for image in row}
    if len(shapes) != 1:
        raise ValueError(f"every view must share one shape, got {sorted(shapes)}")
    if len(frames) == 1 and len(frames[0]) == 1:
        return np.ascontiguousarray(np.asarray(frames[0][0], dtype=np.uint8))
    height, width = next(iter(shapes))[:2]
    if height * width * len(frames) * len(frames[0]) > MAX_TILE_PIXELS:
        raise ValueError("tiled observation exceeds the pixel bound")
    rows = [np.concatenate([np.asarray(image, dtype=np.uint8) for image in row], axis=1)
            for row in frames]
    return np.ascontiguousarray(np.concatenate(rows, axis=0))


def policy_observation(*, frames, state, prompt: str) -> PolicyObservation:
    """Assemble the observation a policy is allowed to see."""
    values = np.asarray(state, dtype=np.float32)
    if values.ndim != 1 or not values.size:
        raise ValueError("state must be a non-empty 1-D vector")
    if not np.isfinite(values).all():
        raise ValueError("state contains a non-finite value")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("a policy observation needs a task instruction")
    return PolicyObservation(image=tile_frames(frames), state=values, prompt=prompt)
