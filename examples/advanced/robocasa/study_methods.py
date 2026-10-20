"""Reference methods: the baselines that give real numbers before any policy.

A measurement harness has to be shown to measure before anything interesting is
plugged into it, and the cheapest way to do that is with policies whose
behaviour is known in advance.

`StillMethod` holds the arm still. Its success rate must be zero and its partial
score must be the floor, so any nonzero result from it means the scorer is
crediting something that did not happen. `RandomMethod` samples bounded actions
from a seeded generator; it should occasionally reach or nudge the object and
essentially never succeed, which separates "the scorer never fires" from "the
scorer only fires on success". Both are reproducible: the random one is seeded
per trial, so its episodes replay.

Both also exercise the horizon: a method returns a chunk of `horizon` future
actions, the harness dispatches them one control step at a time, and the number
of chunks is the model-call count the study reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from examples.advanced.openpi_policy.common import ActionChunk, PolicyObservation

from .study_representations import normalised_dof


@dataclass
class StillMethod:
    """Command no motion. The floor every other method is read against."""

    representation: str = "eef_delta"
    horizon: int = 1
    source: str = "scripted"
    calls: int = 0

    def predict(self, observation: PolicyObservation) -> ActionChunk:
        self.calls += 1
        dof = normalised_dof(self.representation)
        actions = np.zeros((self.horizon, dof), dtype=np.float32)
        # An open gripper is the resting command, not a closed one: a closed
        # gripper would be a grasp attempt and this method attempts nothing.
        actions[:, -1] = -1.0
        return ActionChunk(actions=actions, horizon=self.horizon, dof=dof,
                           source=self.source)


@dataclass
class RandomMethod:
    """Bounded random motion from a seeded generator, so episodes replay."""

    representation: str = "eef_delta"
    horizon: int = 1
    scale: float = 0.5
    seed: int = 0
    source: str = "scripted"
    calls: int = 0
    rng: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.scale <= 1.0:
            raise ValueError("scale must be in (0, 1]")
        self.reset(self.seed)

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.calls = 0

    def predict(self, observation: PolicyObservation) -> ActionChunk:
        self.calls += 1
        dof = normalised_dof(self.representation)
        actions = self.rng.uniform(-self.scale, self.scale,
                                   size=(self.horizon, dof)).astype(np.float32)
        return ActionChunk(actions=actions, horizon=self.horizon, dof=dof,
                           source=self.source)
