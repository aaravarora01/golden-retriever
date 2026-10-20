"""The committed pick-and-place suite, and why it is the size it is.

The study runs 120 tasks for its headline numbers and a **30-task subset** for
every ablation. This suite is eighteen tasks, which is the ablation scale, and
each entry was obtained by constructing the environment on this machine rather
than read off a list — the goal text is RoboCasa's own, and `initial_reach_m` is
the measured distance from the end-effector to the object at reset.

Of RoboSuite's 38 registered `PickPlace*` and `Place*` environments:

* **18 are here.** Single-object RoboCasa kitchen tasks that construct, carry a
  language goal, and expose one `obj` the progress ladder can score.
* **9 are constructible but unscored.** The multi-object `Place*` composites
  (`PlaceVegetablesEvenly`, `PlaceDishesBySink`, …) have no single `obj` key, so
  the ladder in `study_progress` cannot say which object was grasped. They are
  listed in `MULTI_OBJECT_TASKS` rather than dropped silently, because the gap is
  the scorer's and not the simulator's.
* **6 are not RoboCasa tasks at all.** `PickPlace`, `PickPlaceBread`,
  `PickPlaceCan`, `PickPlaceCereal`, `PickPlaceMilk` and `PickPlaceSingle` are
  base RoboSuite bin-picking environments; they reject `layout_ids` because they
  have no kitchen to lay out.
* **5 need object registries this machine does not have.**
  `PickPlaceCounterToOven`, `PlaceBeveragesTogether`, `PlaceBreakfastItemsAway`,
  `PlaceEqualIceCubes` and `PlaceIceInCup` raise *"a cannot be empty unless no
  samples are taken"* when sampling a category absent from the lightweight
  `objs_lw` pack. Downloading `objs_objaverse` would add them; that download is
  ~30 GB and stalled partway through on this machine once already, so the suite
  is defined against what is actually present.

`initial_reach_m` is recorded because it is the number that bounds what an
arm-only policy can do: every task here starts between 0.305 m and 0.878 m from
the object, and a Cartesian controller with the study's per-step cap stalls
around 0.29 m. A policy that does not command the base cannot reach any of them,
which is why the field is part of the committed data rather than a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Episode seeds, as the study uses them.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
#: Kitchen layout and style, pinned so an episode is reproducible. Both are
#: 1-based: 0 is not a `LayoutType` and raises inside RoboCasa's scene registry.
LAYOUT_ID = 1
STYLE_ID = 1


@dataclass(frozen=True)
class PnPTask:
    """One committed task, with what was measured at reset."""

    task: str
    goal: str
    initial_reach_m: float
    horizon: int = 1000

    def __post_init__(self) -> None:
        if not self.task or not self.task.startswith(("PickPlace", "Place")):
            raise ValueError("task must be a RoboCasa pick-and-place environment name")
        if not self.goal.strip():
            raise ValueError(f"{self.task} must carry its language goal")
        if not 0.0 < self.initial_reach_m < 3.0:
            raise ValueError(f"{self.task} initial reach is implausible")

    @property
    def arm_reachable(self) -> bool:
        """Whether an arm-only policy could plausibly reach the object.

        False for every committed task: the measured stall distance for a capped
        Cartesian controller is about 0.29 m and the nearest object starts at
        0.305 m. Kept as a property rather than a constant so the claim is
        checkable against the recorded measurement instead of asserted.
        """
        return self.initial_reach_m <= ARM_STALL_DISTANCE_M


#: Measured distance at which a capped Cartesian controller stops closing, with
#: no joint at a limit. See STUDY.md for how this was obtained.
ARM_STALL_DISTANCE_M = 0.29

PNP_TASKS = (
    PnPTask("PickPlaceToasterOvenToCounter", "Pick the bread from the tray and place it on the plate.", 0.305),
    PnPTask("PickPlaceCoffee", "Pick the mug from under the coffee machine dispenser and place it on the counter.", 0.333),
    PnPTask("PickPlaceCabinetToCounter", "Pick the sugar cube from the cabinet and place it on the counter.", 0.334),
    PnPTask("PickPlaceFridgeShelfToDrawer", "Pick the mushroom from the fridge shelf and place it in the fridge drawer.", 0.365),
    PnPTask("PickPlaceCounterToDrawer", "Pick the ladle from the counter and place it in the drawer.", 0.429),
    PnPTask("PickPlaceCounterToStandMixer", "Place the cheese in the stand mixer bowl.", 0.439),
    PnPTask("PickPlaceToasterToCounter", "Place the toasted item on a plate.", 0.444),
    PnPTask("PickPlaceCounterToBlender", "Pick the banana from the counter and place it in the blender.", 0.479),
    PnPTask("PickPlaceMicrowaveToCounter", "Pick the marshmallow from the microwave and place it on the counter.", 0.479),
    PnPTask("PickPlaceCounterToCabinet", "Pick the sugar cube from the counter and place it in the cabinet.", 0.512),
    PnPTask("PickPlaceStoveToCounter", "Pick the shrimp from the pan and place it on the plate.", 0.528),
    PnPTask("PickPlaceFridgeDrawerToShelf", "Pick the carrot from the fridge drawer and place it on the fridge shelf.", 0.590),
    PnPTask("PickPlaceSinkToCounter", "Pick the broccoli from the sink and place it on the counter.", 0.605),
    PnPTask("PickPlaceDrawerToCounter", "Pick the tongs from the drawer and place it on the counter.", 0.628),
    PnPTask("PickPlaceCounterToStove", "Pick the eggplant from the plate and place it in the pan.", 0.649),
    PnPTask("PickPlaceCounterToMicrowave", "Pick the marshmallow from the counter and place it in the microwave.", 0.659),
    PnPTask("PickPlaceCounterToToasterOven", "Place the bread on the tray of the toaster oven.", 0.718),
    PnPTask("PickPlaceCounterToSink", "Pick the pear from the counter and place it in the sink.", 0.878),
)

#: Constructible, but the progress ladder cannot score them: no single `obj`.
MULTI_OBJECT_TASKS = (
    "PlaceDishesBySink", "PlaceFoodInBowls", "PlaceLidToBoil", "PlaceMeatInMarinade",
    "PlaceMicrowaveSafeItem", "PlaceOnDishRack", "PlaceStraw", "PlaceVegetablesEvenly",
    "PlaceVeggiesInDrawer",
)

#: Base RoboSuite bin-picking environments, not RoboCasa kitchens. `PickPlace`
#: is abstract and raises `NotImplementedError`; the rest reject `layout_ids`
#: because they have no kitchen to lay out.
NON_KITCHEN_TASKS = (
    "PickPlace", "PickPlaceBread", "PickPlaceCan", "PickPlaceCereal", "PickPlaceMilk",
    "PickPlaceSingle",
)

#: Need object categories absent from the lightweight asset pack. Each of these
#: raises "a cannot be empty unless no samples are taken" when RoboCasa samples
#: a category the `objs_lw` pack does not contain.
MISSING_ASSET_TASKS = (
    "PickPlaceCounterToOven", "PlaceBeveragesTogether", "PlaceBreakfastItemsAway",
    "PlaceEqualIceCubes", "PlaceIceInCup",
)

_BY_NAME = {task.task: task for task in PNP_TASKS}
if len(_BY_NAME) != len(PNP_TASKS):  # pragma: no cover - guards a committed table
    raise AssertionError("the committed suite contains a duplicate task")


def get_task(name: str) -> PnPTask:
    """Return a committed task or fail with the available names."""
    try:
        return _BY_NAME[name]
    except KeyError as error:
        if name in MULTI_OBJECT_TASKS:
            raise ValueError(
                f"{name} constructs but has no single `obj`, so the progress ladder "
                "cannot score it; see MULTI_OBJECT_TASKS") from error
        if name in NON_KITCHEN_TASKS:
            raise ValueError(
                f"{name} is a base RoboSuite bin-picking environment, not a RoboCasa "
                "kitchen task, and rejects layout_ids") from error
        if name in MISSING_ASSET_TASKS:
            raise ValueError(
                f"{name} needs object categories absent from the lightweight asset "
                "pack; download objs_objaverse to enable it") from error
        raise ValueError(
            f"unknown task {name!r}; available: {', '.join(sorted(_BY_NAME))}") from error


def list_tasks() -> tuple:
    return tuple(sorted(_BY_NAME))


def ablation_subset(count: int = 8) -> tuple:
    """A spread across the reach range, for sweeps too slow to run all 18.

    Evenly spaced through the suite ordered by initial reach, so a subset is not
    accidentally all-near or all-far — reach is the variable that decides
    whether a policy can engage the object at all.
    """
    if type(count) is not int or not 1 <= count <= len(PNP_TASKS):
        raise ValueError(f"count must be an int from 1 through {len(PNP_TASKS)}")
    ordered = sorted(PNP_TASKS, key=lambda task: task.initial_reach_m)
    step = len(ordered) / count
    return tuple(ordered[int(index * step)].task for index in range(count))


# --- the task-specification axis -----------------------------------------
#
# The study ablates how the task is specified: default language, vague language,
# or a video demonstration. The first two are available here.
#
# `default` is RoboCasa's own goal string, which names the object and both
# fixtures: "Pick the sugar cube from the counter and place it in the cabinet."
#
# `vague` is derived from the environment name rather than authored per task, so
# no instruction here is invented text. `PickPlaceCounterToCabinet` yields "Move
# the object from the counter to the cabinet." — it keeps the fixtures and drops
# the object identity, which is the information a vague instruction withholds. A
# name that does not split into source and destination falls back to a generic
# instruction, and the fallback is reported rather than hidden, because a task
# whose vague form carries no fixtures is a different condition from one that
# does.
#
# A video-demonstration condition is not implemented: it needs recorded
# demonstrations, and the dataset links for this RoboCasa revision are dead (see
# STUDY.md).

SPECIFICATION_STYLES = ("default", "vague")

_FIXTURE_WORDS = {
    "Counter": "counter", "Cabinet": "cabinet", "Sink": "sink", "Microwave": "microwave",
    "Stove": "stove", "Drawer": "drawer", "Blender": "blender", "StandMixer": "stand mixer",
    "Toaster": "toaster", "ToasterOven": "toaster oven", "FridgeShelf": "fridge shelf",
    "FridgeDrawer": "fridge drawer", "Oven": "oven", "Shelf": "shelf", "Pot": "pot",
}


def _split_fixtures(name: str):
    """Split `PickPlace<Source>To<Destination>` into its two fixture words."""
    if not name.startswith("PickPlace") or "To" not in name:
        return None
    stem = name[len("PickPlace"):]
    for index in range(len(stem) - 1, 0, -1):
        if stem[index:index + 2] == "To":
            source, destination = stem[:index], stem[index + 2:]
            if source in _FIXTURE_WORDS and destination in _FIXTURE_WORDS:
                return _FIXTURE_WORDS[source], _FIXTURE_WORDS[destination]
    return None


def specification(task, style: str = "default") -> dict:
    """The instruction a policy receives, and whether it is the derived one.

    Returns the text plus `derived`, so a run records which condition produced
    it rather than leaving a reader to infer the condition from the wording.
    """
    if style not in SPECIFICATION_STYLES:
        raise ValueError(f"style must be one of {SPECIFICATION_STYLES}")
    entry = task if isinstance(task, PnPTask) else get_task(task)
    if style == "default":
        return {"text": entry.goal, "style": "default", "derived": False,
                "names_object": True}
    fixtures = _split_fixtures(entry.task)
    if fixtures is None:
        return {"text": "Move the object to where it belongs.", "style": "vague",
                "derived": True, "names_object": False, "fallback": True}
    source, destination = fixtures
    return {"text": f"Move the object from the {source} to the {destination}.",
            "style": "vague", "derived": True, "names_object": False, "fallback": False}
