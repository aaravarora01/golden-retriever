"""Partial-credit scoring, because the simulator only reports pass or fail.

The study reports a `Score` giving "partial credit for task progress (0-100 per
episode)" alongside binary success. RoboCasa does not provide one: its kitchen
tasks expose `_check_success()` and, with `reward_shaping` off, a reward of zero
until that flips. A benchmark that reported only success would make every
unsuccessful policy look identical, which is exactly the distinction a partial
score exists to draw — a policy that reached and grasped the object is not the
same as one that never moved.

So the score is a ladder over the stages a pick-and-place necessarily passes
through, each rung worth 25 points. The rungs are *monotone*: a rung counts only
if every rung below it was reached, so a trace that somehow registers a lift
without a grasp scores as never having grasped rather than collecting a rung out
of order. The top rung is the simulator's own success predicate, so a full score
is never awarded by this module's own reasoning.

Two honesty notes. The lower rungs read privileged simulator state — object pose
and gripper contact — which is fine for a *scorer* and forbidden for a policy;
nothing here is ever placed in a `PolicyObservation`. And the thresholds are
geometric judgements, not ground truth: `REACH_RADIUS` is the distance at which
a gripper is plausibly addressing an object, and `LIFT_HEIGHT` is a rise clearly
beyond settling noise. They are named constants so a reader can disagree with
them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Distance from gripper to object that counts as addressing it, in metres.
REACH_RADIUS = 0.10
#: Rise above the object's starting height that counts as a lift, in metres.
LIFT_HEIGHT = 0.05

#: Monotone stages of a pick-and-place, each worth an equal share of 100.
PROGRESS_RUNGS = ("reached", "grasped", "lifted", "succeeded")
POINTS_PER_RUNG = 100 // len(PROGRESS_RUNGS)


@dataclass
class ProgressTracker:
    """Accumulate the highest rung reached during one episode.

    Latching matters: an episode that grasps the object and then drops it has
    still demonstrably grasped it, and a score that fell back to zero would
    report less than the episode showed.
    """

    initial_object_height: float | None = None
    reached: bool = False
    grasped: bool = False
    lifted: bool = False
    succeeded: bool = False

    def observe(self, *, gripper_position, object_position, grasped: bool,
                success: bool) -> None:
        gripper = np.asarray(gripper_position, dtype=float)
        obj = np.asarray(object_position, dtype=float)
        if self.initial_object_height is None:
            self.initial_object_height = float(obj[2])
        if float(np.linalg.norm(gripper - obj)) <= REACH_RADIUS:
            self.reached = True
        if grasped:
            self.reached = True
            self.grasped = True
        if self.grasped and float(obj[2]) - self.initial_object_height >= LIFT_HEIGHT:
            self.lifted = True
        if success:
            self.succeeded = True

    @property
    def rungs(self) -> dict:
        """Rung-by-rung, enforcing that each implies the one below it."""
        reached = dict(zip(PROGRESS_RUNGS,
                           (self.reached, self.grasped, self.lifted, self.succeeded)))
        highest = 0
        for index, rung in enumerate(PROGRESS_RUNGS, start=1):
            if not reached[rung]:
                break
            highest = index
        return {rung: index < highest for index, rung in enumerate(PROGRESS_RUNGS)}

    @property
    def score(self) -> int:
        """Partial credit out of 100, on the study's scale."""
        return sum(self.rungs.values()) * POINTS_PER_RUNG

    def to_dict(self) -> dict:
        return {"score": self.score, "rungs": self.rungs,
                "reach_radius_m": REACH_RADIUS, "lift_height_m": LIFT_HEIGHT}
