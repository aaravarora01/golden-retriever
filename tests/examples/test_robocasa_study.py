"""Simulator-free tests for the RoboCasa policy-study lane.

Everything here runs without RoboCasa installed: the environment adapter defers
its simulator imports, so its configuration, representation handling and scoring
are testable on their own. The parts that need physics are exercised by the
`study-pnp-baselines` pixi task, which is where real episodes belong.

The properties pinned are the ones that would quietly corrupt a result rather
than break a run — a score credited for progress that did not happen, a clipped
action applied without being recorded, privileged state reaching a policy, or a
representation approximated under a label it does not match.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.advanced.robocasa.study_methods import RandomMethod, StillMethod
from examples.advanced.robocasa.study_observation import policy_observation, tile_frames
from examples.advanced.robocasa.study_progress import (
    LIFT_HEIGHT, POINTS_PER_RUNG, PROGRESS_RUNGS, REACH_RADIUS, ProgressTracker)
from examples.advanced.robocasa.study_representations import (
    MAX_ROTATION_PER_STEP, MAX_TRANSLATION_PER_STEP, REPRESENTATION_DOF, EefDeltaCodec,
    UnsupportedRepresentation, normalised_dof)
from examples.advanced.robocasa.study_robocasa_env import (
    CAMERA_ROLES, ObservationConfig, RoboCasaPnPEnvironment)


def _frame(value=0, size=8):
    return np.full((size, size, 3), value, dtype=np.uint8)


# --- representations -----------------------------------------------------


def test_every_named_representation_has_a_width():
    from examples.advanced.robocasa.study_representations import REPRESENTATIONS

    assert set(REPRESENTATIONS) == set(REPRESENTATION_DOF)
    assert all(width > 0 for width in REPRESENTATION_DOF.values())


def test_an_unknown_representation_names_the_available_ones():
    with pytest.raises(UnsupportedRepresentation, match="available: eef_delta"):
        normalised_dof("b_spline")


def test_a_joint_representation_is_not_an_end_effector_codec():
    """RoboCasa's operational-space controller cannot express it, and
    approximating it would report it under a label it does not match."""
    with pytest.raises(UnsupportedRepresentation):
        EefDeltaCodec("joint_velocity")


def test_the_translation_cap_is_applied_and_recorded():
    codec = EefDeltaCodec("eef_delta")
    row = codec.to_delta(np.array([1.0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))
    assert float(np.linalg.norm(row[:3])) == pytest.approx(MAX_TRANSLATION_PER_STEP)
    ledger = codec.ledger.to_dict()
    assert ledger["clipped_rows"] == 1 and ledger["clipped_fraction"] == 1.0
    assert ledger["max_translation_excess_m"] == pytest.approx(1.0 - MAX_TRANSLATION_PER_STEP)


def test_the_rotation_cap_is_applied_and_recorded():
    codec = EefDeltaCodec("eef_delta")
    row = codec.to_delta(np.array([0, 0, 0, 1.0, 0, 0, -1.0], dtype=np.float32))
    assert float(np.linalg.norm(row[3:6])) == pytest.approx(MAX_ROTATION_PER_STEP)
    assert codec.ledger.to_dict()["max_rotation_excess_rad"] > 0


def test_an_action_inside_the_cap_is_not_recorded_as_clipped():
    """Otherwise the ledger could not distinguish a policy that saturates from
    one that never approaches the limit."""
    codec = EefDeltaCodec("eef_delta")
    codec.to_delta(np.array([0.001, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))
    assert codec.ledger.to_dict()["clipped_rows"] == 0


def test_an_absolute_representation_needs_the_current_pose():
    codec = EefDeltaCodec("eef_absolute")
    with pytest.raises(ValueError, match="current pose"):
        codec.to_delta(np.zeros(7, dtype=np.float32))


def test_an_absolute_target_becomes_a_capped_delta_toward_it():
    codec = EefDeltaCodec("eef_absolute")
    row = codec.to_delta(np.array([0.5, 0, 0, 0, 0, 0, -1.0], dtype=np.float32),
                         current_position=np.zeros(3), current_rotation=np.zeros(3))
    assert row[0] == pytest.approx(MAX_TRANSLATION_PER_STEP)


def test_every_codec_emits_one_shape_so_the_environment_never_branches():
    for representation in ("eef_delta", "eef_delta_base"):
        codec = EefDeltaCodec(representation)
        row = codec.to_delta(np.zeros(REPRESENTATION_DOF[representation], dtype=np.float32))
        assert row.shape == (10,)


def test_the_base_segment_passes_through_only_for_the_mobile_representation():
    mobile = EefDeltaCodec("eef_delta_base").to_delta(
        np.array([0, 0, 0, 0, 0, 0, -1, 0.4, -0.2, 0.1], dtype=np.float32))
    assert mobile[7:10] == pytest.approx([0.4, -0.2, 0.1])
    arm_only = EefDeltaCodec("eef_delta").to_delta(np.zeros(7, dtype=np.float32))
    assert arm_only[7:10] == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.parametrize("row", (
    np.zeros(6, dtype=np.float32), np.zeros(8, dtype=np.float32),
    np.array([np.nan, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    np.array([np.inf, 0, 0, 0, 0, 0, 0], dtype=np.float32),
))
def test_an_unusable_action_row_is_rejected(row):
    with pytest.raises(ValueError):
        EefDeltaCodec("eef_delta").to_delta(row)


# --- the partial score ---------------------------------------------------


def test_no_progress_scores_zero():
    assert ProgressTracker().score == 0


def test_the_ladder_is_monotone_so_a_rung_cannot_be_skipped():
    """A lift reported without a grasp is a trace that never grasped, not
    partial credit for lifting."""
    tracker = ProgressTracker()
    tracker.observe(gripper_position=[9, 9, 9], object_position=[0, 0, 0],
                    grasped=False, success=False)
    tracker.initial_object_height = 0.0
    tracker.observe(gripper_position=[9, 9, 9], object_position=[0, 0, 1.0],
                    grasped=False, success=False)
    assert tracker.score == 0
    assert not any(tracker.rungs.values())


def test_reaching_scores_one_rung():
    tracker = ProgressTracker()
    tracker.observe(gripper_position=[0, 0, REACH_RADIUS / 2], object_position=[0, 0, 0],
                    grasped=False, success=False)
    assert tracker.rungs["reached"] and tracker.score == POINTS_PER_RUNG


def test_a_grasp_implies_a_reach_even_from_a_distance_reading():
    tracker = ProgressTracker()
    tracker.observe(gripper_position=[5, 5, 5], object_position=[0, 0, 0],
                    grasped=True, success=False)
    assert tracker.rungs["reached"] and tracker.rungs["grasped"]
    assert tracker.score == 2 * POINTS_PER_RUNG


def test_a_lift_requires_a_grasp_and_a_real_rise():
    tracker = ProgressTracker()
    tracker.observe(gripper_position=[0, 0, 0], object_position=[0, 0, 0],
                    grasped=True, success=False)
    tracker.observe(gripper_position=[0, 0, 0], object_position=[0, 0, LIFT_HEIGHT / 2],
                    grasped=True, success=False)
    assert not tracker.rungs["lifted"]
    tracker.observe(gripper_position=[0, 0, 0], object_position=[0, 0, LIFT_HEIGHT * 1.5],
                    grasped=True, success=False)
    assert tracker.rungs["lifted"] and tracker.score == 3 * POINTS_PER_RUNG


def test_the_top_rung_is_the_simulator_predicate_and_completes_the_score():
    tracker = ProgressTracker()
    tracker.observe(gripper_position=[0, 0, 0], object_position=[0, 0, 0],
                    grasped=True, success=False)
    tracker.observe(gripper_position=[0, 0, 0], object_position=[0, 0, LIFT_HEIGHT * 2],
                    grasped=True, success=True)
    assert tracker.score == 100
    assert all(tracker.rungs.values())


def test_a_rung_latches_so_a_dropped_object_keeps_its_credit():
    """An episode that grasped and then dropped has still demonstrably
    grasped; a score that fell back would report less than happened."""
    tracker = ProgressTracker()
    tracker.observe(gripper_position=[0, 0, 0], object_position=[0, 0, 0],
                    grasped=True, success=False)
    tracker.observe(gripper_position=[9, 9, 9], object_position=[0, 0, 0],
                    grasped=False, success=False)
    assert tracker.rungs["grasped"]


def test_the_score_scale_matches_the_study():
    assert POINTS_PER_RUNG * len(PROGRESS_RUNGS) == 100


# --- observations --------------------------------------------------------


def test_a_single_view_with_no_history_is_the_frame_itself():
    assert tile_frames([[_frame(3)]]).shape == (8, 8, 3)


def test_views_tile_across_and_history_tiles_down():
    tiled = tile_frames([[_frame(1), _frame(2)], [_frame(3), _frame(4)]])
    assert tiled.shape == (16, 16, 3)
    # Oldest row first, so a policy reads the frames in time order.
    assert tiled[0, 0, 0] == 1 and tiled[8, 0, 0] == 3


@pytest.mark.parametrize("frames", ([], [[]], [[_frame()], [_frame(), _frame()]]))
def test_an_unusable_frame_set_is_rejected(frames):
    with pytest.raises(ValueError):
        tile_frames(frames)


def test_mismatched_view_shapes_are_rejected():
    with pytest.raises(ValueError, match="share one shape"):
        tile_frames([[_frame(size=8), _frame(size=16)]])


def test_an_observation_needs_an_instruction():
    with pytest.raises(ValueError, match="task instruction"):
        policy_observation(frames=[[_frame()]], state=np.zeros(4), prompt="  ")


@pytest.mark.parametrize("state", (np.zeros(0), np.zeros((2, 2)),
                                   np.array([np.nan]), np.array([np.inf])))
def test_an_unusable_state_is_rejected(state):
    with pytest.raises(ValueError):
        policy_observation(frames=[[_frame()]], state=state, prompt="do it")


def test_an_observation_carries_only_what_a_policy_may_see():
    """The scorer reads the object's pose; the policy must not. The Lift demo
    in this folder does put privileged state in its observation, which is fine
    for a heuristic and would invalidate a policy study."""
    observation = policy_observation(frames=[[_frame()]], state=np.arange(11, dtype=np.float32),
                                     prompt="Pick the cube and place it in the cabinet")
    assert set(vars(observation)) == {"image", "state", "prompt"}
    assert observation.state.shape == (11,)


# --- environment configuration (no simulator needed) --------------------


def test_the_camera_roles_mirror_robolabs_registration():
    assert set(CAMERA_ROLES) == {"over_shoulder", "wrist"}
    assert CAMERA_ROLES["wrist"] == "robot0_eye_in_hand"


@pytest.mark.parametrize("kwargs", (
    {"views": ()}, {"views": ("nose_cam",)}, {"views": ("wrist", "wrist")},
    {"history": -1}, {"history": 9}, {"history": 1.0},
    {"image_size": 16}, {"image_size": 1024}, {"image_size": 64.0},
))
def test_an_unusable_observation_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ObservationConfig(**kwargs)


def test_the_observation_config_names_real_cameras():
    config = ObservationConfig(views=("over_shoulder", "wrist"), history=2)
    assert config.camera_names == ["robot0_agentview_left", "robot0_eye_in_hand"]
    assert config.to_dict()["history"] == 2


def test_a_representation_the_backend_cannot_express_is_refused_by_name():
    with pytest.raises(UnsupportedRepresentation, match="RoboLab backend"):
        RoboCasaPnPEnvironment(representation="joint_velocity")


@pytest.mark.parametrize("layout,style", ((0, 1), (1, 0), (-1, 1)))
def test_a_zero_based_layout_is_rejected_before_the_scene_registry_sees_it(layout, style):
    """`layout_ids=0` raises a `KeyError` deep inside RoboCasa, because the
    concrete layouts start at 1 and the negatives are group selectors."""
    with pytest.raises(ValueError):
        RoboCasaPnPEnvironment(layout_id=layout, style_id=style)


def test_the_backend_describes_what_a_run_should_record():
    description = RoboCasaPnPEnvironment(command_base=True).describe()
    assert description["backend"] == "robocasa"
    assert description["command_base"] is True
    assert description["frozen_segments"] == ["torso"]
    assert description["observation"]["views"] == ["over_shoulder"]


def test_freezing_the_base_is_recorded_as_a_frozen_segment():
    """A run that froze the base was measuring an arm against a task that needs
    a mobile manipulator, so the evidence has to say so."""
    assert RoboCasaPnPEnvironment().describe()["frozen_segments"] == ["base", "torso"]


# --- the baselines -------------------------------------------------------


def test_the_still_method_commands_no_motion_and_an_open_gripper():
    chunk = StillMethod(horizon=4).predict(None)
    assert chunk.actions.shape == (4, 7) and chunk.horizon == 4
    assert np.all(chunk.actions[:, :6] == 0.0)
    assert np.all(chunk.actions[:, 6] == -1.0)
    assert chunk.source == "scripted"


def test_the_random_method_replays_from_its_seed():
    first = RandomMethod(horizon=3, seed=11).predict(None).actions
    second = RandomMethod(horizon=3, seed=11).predict(None).actions
    assert np.array_equal(first, second)
    assert not np.array_equal(first, RandomMethod(horizon=3, seed=12).predict(None).actions)


def test_the_random_method_stays_inside_the_safety_envelope():
    from examples.advanced.robocasa.method_harness import SafetyEnvelope

    chunk = RandomMethod(horizon=6, seed=2).predict(None)
    assert SafetyEnvelope().validate(chunk).actions.shape == (6, 7)


def test_a_method_counts_its_own_calls():
    """One call per dispatched chunk is the study's model-call metric."""
    method = StillMethod(horizon=2)
    for _ in range(3):
        method.predict(None)
    assert method.calls == 3


def test_a_mobile_method_emits_the_wider_row():
    chunk = RandomMethod(horizon=2, seed=1, representation="eef_delta_base").predict(None)
    assert chunk.actions.shape == (2, 10) and chunk.dof == 10


@pytest.mark.parametrize("scale", (0.0, -0.1, 1.5))
def test_an_unusable_random_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        RandomMethod(scale=scale)


# --- the RoboLab port ----------------------------------------------------


def test_the_robolab_backend_satisfies_the_same_protocol_as_robocasa():
    """The port is an adapter swap only if both satisfy one contract."""
    from examples.advanced.robocasa.study_robolab_env import RoboLabEnvironment

    for name in ("reset", "step", "verify", "close", "describe"):
        assert callable(getattr(RoboLabEnvironment, name))
        assert callable(getattr(RoboCasaPnPEnvironment, name))


def test_a_missing_robolab_stack_says_which_requirement_is_unmet():
    """Not a bare ImportError three frames deep: the reason this cannot run
    here is a hardware and OS requirement, and the message says so."""
    from examples.advanced.robocasa.study_robolab_env import RoboLabUnavailable, require_robolab

    with pytest.raises(RoboLabUnavailable, match="RTX GPU"):
        require_robolab()
    with pytest.raises(RoboLabUnavailable, match="Apple Silicon"):
        require_robolab()


def test_importing_the_robolab_backend_needs_no_simulator():
    """It is imported by the sweep on every platform, so its module body must
    not touch Isaac."""
    import examples.advanced.robocasa.study_robolab_env as module

    assert module.EXPOSED_DOF == 8


def test_the_robolab_backend_declares_a_joint_action_space():
    """DROID is fixed-base and joint-space; RoboCasa is mobile and
    operational-space. The backends sit at two points on the study's
    action-representation axis, which is the port's whole value."""
    from examples.advanced.robocasa.study_robolab_env import (
        SUPPORTED_REPRESENTATIONS, RoboLabEnvironment)

    assert SUPPORTED_REPRESENTATIONS == ("joint_velocity", "joint_position")
    description = RoboLabEnvironment().describe()
    assert description["fixed_base"] is True and description["exposed_dof"] == 8
    assert description["verified_against_live_instance"] is False


def test_an_end_effector_representation_is_refused_by_the_robolab_backend():
    from examples.advanced.robocasa.study_robolab_env import RoboLabEnvironment

    with pytest.raises(UnsupportedRepresentation, match="RoboCasa backend"):
        RoboLabEnvironment(representation="eef_delta")


def test_the_robolab_camera_roles_match_droids_registration():
    from examples.advanced.robocasa.study_robolab_env import CAMERA_ROLES as ROBOLAB_ROLES

    assert ROBOLAB_ROLES["over_shoulder"] == "over_shoulder_left_camera"
    assert ROBOLAB_ROLES["wrist"] == "wrist_cam"
    # The same role names as the RoboCasa backend, which is what lets one
    # observation config describe either.
    assert set(ROBOLAB_ROLES) == set(CAMERA_ROLES)


# --- the sweep -----------------------------------------------------------


class _FakeEnvironment:
    """Stands in for a backend so the sweep is testable without physics."""

    def __init__(self, task="T", representation="eef_delta", *, succeed_at=None):
        from examples.advanced.robocasa.study_progress import ProgressTracker
        from examples.advanced.robocasa.study_representations import EefDeltaCodec

        self.task = task
        self.representation = representation
        self.progress = ProgressTracker()
        self.codec = EefDeltaCodec(representation)
        self.steps = 0
        self.succeed_at = succeed_at

    def _observation(self):
        return policy_observation(frames=[[_frame()]], state=np.zeros(3), prompt="go")

    def reset(self, request):
        self.steps = 0
        return self._observation()

    def step(self, action):
        from examples.advanced.robocasa.method_harness import Transition

        self.steps += 1
        self.codec.to_delta(np.asarray(action, dtype=np.float32))
        done = self.succeed_at is not None and self.steps >= self.succeed_at
        if done:
            self.progress.observe(gripper_position=[0, 0, 0], object_position=[0, 0, 0],
                                  grasped=True, success=True)
            self.progress.observe(gripper_position=[0, 0, 0], object_position=[0, 0, 1.0],
                                  grasped=True, success=True)
        return Transition(observation=self._observation(), reward=0.0, terminated=done)

    def verify(self):
        from examples.advanced.robocasa.method_harness import Verification

        return Verification(success=self.progress.succeeded)

    def close(self):
        pass

    def describe(self):
        return {"backend": "fake", "task": self.task}


def _sweep(**kwargs):
    from examples.advanced.robocasa.study_sweep import Sweep

    return Sweep(build_environment=lambda task, representation: _FakeEnvironment(
        task=task, representation=representation, **kwargs), **{
            "tasks": ("T1",), "seeds": (0, 1)})


def test_a_sweep_runs_every_cell_and_summarises_per_method():
    from examples.advanced.robocasa.study_sweep import MethodSpec

    specs = (MethodSpec(method_id="still", horizon=4,
                        build=lambda env: StillMethod(horizon=4)),
             MethodSpec(method_id="random", horizon=4,
                        build=lambda env: RandomMethod(horizon=4, seed=0)))
    summary = _sweep().run(specs)
    assert summary["protocol"]["episodes"] == 2
    assert len(summary["episodes"]) == 4
    assert set(summary["summary"]) == {"still", "random"}
    assert summary["errors"] == 0


def test_the_sweep_counts_model_calls_from_the_harness_not_the_method():
    """A method cannot misreport its own call count if the count comes from the
    harness's dispatch events."""
    from examples.advanced.robocasa.study_sweep import MethodSpec

    summary = _sweep().run((MethodSpec(method_id="m", horizon=5,
                                       build=lambda env: StillMethod(horizon=5)),))
    row = summary["episodes"][0]
    assert row["model_calls"] * 5 >= row["control_steps"] > 0
    assert row["horizon"] == 5


def test_control_steps_and_model_calls_are_separate_counts():
    from examples.advanced.robocasa.study_sweep import MethodSpec

    narrow = _sweep().run((MethodSpec(method_id="h1", horizon=1,
                                      build=lambda env: StillMethod(horizon=1)),))
    wide = _sweep().run((MethodSpec(method_id="h8", horizon=8,
                                    build=lambda env: StillMethod(horizon=8)),))
    # Same control steps, fewer calls: that is what a horizon buys, and why
    # the two must never be summed or substituted.
    assert narrow["summary"]["h1"]["control_steps"] == wide["summary"]["h8"]["control_steps"]
    assert wide["summary"]["h8"]["model_calls"] < narrow["summary"]["h1"]["model_calls"]


def test_a_fresh_method_is_built_for_every_trial():
    """Reusing one would leak an episode's phase or generator into the next."""
    from examples.advanced.robocasa.study_sweep import MethodSpec

    built = []
    specs = (MethodSpec(method_id="m", horizon=2,
                        build=lambda env: built.append(1) or StillMethod(horizon=2)),)
    _sweep().run(specs)
    assert len(built) == 2


def test_a_successful_cell_is_scored_and_reported():
    from examples.advanced.robocasa.study_sweep import MethodSpec

    summary = _sweep(succeed_at=3).run((MethodSpec(method_id="m", horizon=1,
                                                   build=lambda env: StillMethod(horizon=1)),))
    assert summary["summary"]["m"]["success_rate"] == 1.0
    assert summary["summary"]["m"]["mean_score"] == 100.0


def test_a_sweep_rejects_duplicate_methods_and_an_empty_set():
    from examples.advanced.robocasa.study_sweep import MethodSpec

    spec = MethodSpec(method_id="same", build=lambda env: StillMethod())
    with pytest.raises(ValueError, match="duplicate method_id"):
        _sweep().run((spec, spec))
    with pytest.raises(ValueError, match="at least one method"):
        _sweep().run(())


def test_a_method_spec_needs_a_factory_not_an_instance():
    from examples.advanced.robocasa.study_sweep import MethodSpec

    with pytest.raises(TypeError, match="callable"):
        MethodSpec(method_id="m", build=StillMethod())


def test_the_rendered_summary_is_sorted_json():
    from examples.advanced.robocasa.study_sweep import MethodSpec, render

    summary = _sweep().run((MethodSpec(method_id="m", horizon=2,
                                       build=lambda env: StillMethod(horizon=2)),))
    text = render(summary)
    assert json_loads_keys_sorted(text)
    assert "episodes" not in json_loads(render(summary, summary_only=True))


def json_loads(text):
    import json

    return json.loads(text)


def json_loads_keys_sorted(text):
    import json

    data = json.loads(text)
    return list(data) == sorted(data)


def test_the_default_seeds_are_the_studys_protocol():
    from examples.advanced.robocasa.study_sweep import DEFAULT_SEEDS

    assert DEFAULT_SEEDS == (0, 1, 2, 3, 4)


# --- the committed task suite -------------------------------------------


def test_the_suite_accounts_for_every_registered_environment():
    """38 registered `PickPlace*`/`Place*` environments, each in exactly one
    bucket, so nothing was dropped silently."""
    from examples.advanced.robocasa.study_tasks import (
        MISSING_ASSET_TASKS, MULTI_OBJECT_TASKS, NON_KITCHEN_TASKS, PNP_TASKS)

    buckets = (tuple(task.task for task in PNP_TASKS), MULTI_OBJECT_TASKS,
               NON_KITCHEN_TASKS, MISSING_ASSET_TASKS)
    names = [name for bucket in buckets for name in bucket]
    assert len(names) == 38
    assert len(set(names)) == 38, "a task appears in two buckets"
    assert len(PNP_TASKS) == 18


def test_every_committed_task_carries_its_measured_reach_and_goal():
    from examples.advanced.robocasa.study_tasks import PNP_TASKS

    for task in PNP_TASKS:
        assert task.goal.strip() and task.goal.endswith(".")
        assert 0.3 <= task.initial_reach_m <= 0.9


def test_no_committed_task_is_arm_reachable():
    """The measurement that bounds the whole lane: a capped Cartesian
    controller stalls near 0.29 m and the nearest object starts at 0.305 m, so
    a policy that does not command the base cannot engage any task."""
    from examples.advanced.robocasa.study_tasks import ARM_STALL_DISTANCE_M, PNP_TASKS

    assert not any(task.arm_reachable for task in PNP_TASKS)
    assert min(task.initial_reach_m for task in PNP_TASKS) > ARM_STALL_DISTANCE_M


def test_an_unscorable_task_explains_itself_rather_than_404ing():
    from examples.advanced.robocasa.study_tasks import get_task

    with pytest.raises(ValueError, match="no single `obj`"):
        get_task("PlaceVegetablesEvenly")
    with pytest.raises(ValueError, match="bin-picking"):
        get_task("PickPlaceCan")
    with pytest.raises(ValueError, match="objs_objaverse"):
        get_task("PickPlaceCounterToOven")
    with pytest.raises(ValueError, match="available:"):
        get_task("NoSuchTask")


def test_the_ablation_subset_spreads_across_the_reach_range():
    """A subset that was accidentally all-near or all-far would measure
    something other than the suite, since reach decides whether a policy can
    engage the object at all."""
    from examples.advanced.robocasa.study_tasks import ablation_subset, get_task

    subset = ablation_subset(6)
    assert len(subset) == len(set(subset)) == 6
    reaches = [get_task(name).initial_reach_m for name in subset]
    assert max(reaches) - min(reaches) > 0.3


@pytest.mark.parametrize("count", (0, -1, 19, 1.0, True))
def test_an_unusable_subset_size_is_rejected(count):
    from examples.advanced.robocasa.study_tasks import ablation_subset

    with pytest.raises(ValueError):
        ablation_subset(count)


def test_the_layout_is_pinned_and_one_based():
    from examples.advanced.robocasa.study_tasks import LAYOUT_ID, STYLE_ID

    assert LAYOUT_ID >= 1 and STYLE_ID >= 1


def test_the_seeds_are_the_studys_protocol():
    from examples.advanced.robocasa.study_tasks import DEFAULT_SEEDS

    assert DEFAULT_SEEDS == (0, 1, 2, 3, 4)


@pytest.mark.parametrize("bad", (
    {"task": "NotATask", "goal": "g.", "initial_reach_m": 0.5},
    {"task": "PickPlaceX", "goal": "   ", "initial_reach_m": 0.5},
    {"task": "PickPlaceX", "goal": "g.", "initial_reach_m": 0.0},
    {"task": "PickPlaceX", "goal": "g.", "initial_reach_m": 5.0},
))
def test_an_unusable_task_entry_is_rejected(bad):
    from examples.advanced.robocasa.study_tasks import PnPTask

    with pytest.raises(ValueError):
        PnPTask(**bad)


# --- telemetry and transcript replay ------------------------------------


def _observation(fill=0, prompt="pick the cube", state=5):
    return policy_observation(frames=[[_frame(fill)]],
                              state=np.arange(state, dtype=np.float32), prompt=prompt)


def _record(**overrides):
    from examples.advanced.robocasa.study_telemetry import CallRecord

    base = {"task": "T", "method_id": "m", "seed": 0, "call_index": 0,
            "prompt_sha256": "a" * 64, "latency_s": 0.5, "parse_ok": True}
    return CallRecord(**{**base, **overrides})


def test_the_prompt_digest_covers_the_image_not_just_the_instruction():
    """An action chunk depends on what the policy saw, so a transcript keyed on
    the instruction alone would return the same answer at every step."""
    from examples.advanced.robocasa.study_transcript import assemble_prompt, prompt_digest

    same = prompt_digest(assemble_prompt(_observation(0)))
    assert same == prompt_digest(assemble_prompt(_observation(0)))
    assert same != prompt_digest(assemble_prompt(_observation(1)))
    assert same != prompt_digest(assemble_prompt(_observation(0, prompt="other"))) 
    assert same != prompt_digest(assemble_prompt(_observation(0, state=6)))


def test_the_image_is_hashed_not_embedded():
    """A 128x128x3 frame is 49 kB; a transcript carrying them would be
    unusable, and a digest over the bytes is exact."""
    from examples.advanced.robocasa.study_transcript import assemble_prompt

    prompt = assemble_prompt(policy_observation(
        frames=[[_frame(size=128)]], state=np.zeros(5), prompt="go"))
    assert len(prompt) < 400
    assert "image_sha256" in prompt


def test_a_configuration_changes_the_digest_so_two_cannot_collide():
    from examples.advanced.robocasa.study_transcript import assemble_prompt, prompt_digest

    plain = prompt_digest(assemble_prompt(_observation()))
    configured = prompt_digest(assemble_prompt(_observation(), config={"variant": "terse"}))
    assert plain != configured


def test_a_recorded_call_replays_exactly():
    from examples.advanced.robocasa.study_transcript import TranscriptMethod, TranscriptRecorder

    recorder = TranscriptRecorder(method=StillMethod(horizon=3))
    original = recorder.predict(_observation())
    replayed = TranscriptMethod(store=recorder.store, dof=7).predict(_observation())
    assert np.allclose(replayed.actions, original.actions)
    assert replayed.horizon == original.horizon == 3


def test_a_recorder_stores_only_what_it_knows():
    """Inventing token or cost figures would fabricate the numbers a study
    measures, and a measured latency would make the file differ every time."""
    from examples.advanced.robocasa.study_transcript import TranscriptRecorder

    recorder = TranscriptRecorder(method=StillMethod(horizon=2))
    recorder.predict(_observation())
    entry = next(iter(recorder.store.entries.values()))
    assert set(entry) == {"response", "model_id"}


def test_a_recorder_can_attach_a_providers_reported_cost():
    from examples.advanced.robocasa.study_transcript import TranscriptRecorder

    recorder = TranscriptRecorder(
        method=StillMethod(horizon=2), model_id="fake:model",
        report_cost=lambda: {"prompt_tokens": 90, "usd": 0.003, "latency_s": 0.4})
    recorder.predict(_observation())
    entry = next(iter(recorder.store.entries.values()))
    assert entry["usd"] == 0.003 and entry["model_id"] == "fake:model"
    assert recorder.ledger.telemetry[0]["recorded_latency_s"] == 0.4


def test_a_transcript_miss_is_refused_and_measured():
    """No fallback: answering from a scripted policy would mix scripted
    behaviour into a replayed arm's results."""
    from examples.advanced.robocasa.study_transcript import (
        InvalidPolicyOutput, TranscriptMethod, TranscriptRecorder)

    recorder = TranscriptRecorder(method=StillMethod(horizon=2))
    recorder.predict(_observation(0))
    method = TranscriptMethod(store=recorder.store, dof=7)
    with pytest.raises(InvalidPolicyOutput, match="no recorded call"):
        method.predict(_observation(1))
    row = method.ledger.telemetry[-1]
    assert row["transcript_hit"] is False and row["parse_ok"] is False


@pytest.mark.parametrize("text,expected", (
    ("", "empty"), ("   ", "empty"), (None, "empty"),
    ("no json here", "no JSON array"),
    ("[]", "non-empty list"),
    ("[[1,2]]", "must each have 7"),
    ("[[0,0,0,0,0,0,0],[0,0,0]]", "must each have 7"),
    ("[[0,0,0,0,0,0,9]]", "normalised action bound"),
    ('[[0,0,0,0,0,0,"x"]]', "non-numeric"),
    ("[[0,0,0,0,0,0,null]]", "non-finite"),
))
def test_an_unusable_chunk_response_is_refused(text, expected):
    from examples.advanced.robocasa.study_transcript import (
        InvalidPolicyOutput, parse_chunk_response)

    with pytest.raises(InvalidPolicyOutput, match=expected):
        parse_chunk_response(text, dof=7)


def test_a_packaged_response_still_parses():
    from examples.advanced.robocasa.study_transcript import parse_chunk_response

    fenced = "Here you go:\n```json\n[[0.1,0,0,0,0,0,-1]]\n```\nDone."
    assert parse_chunk_response(fenced, dof=7).shape == (1, 7)


def test_an_overlong_chunk_is_refused_before_the_envelope_sees_it():
    from examples.advanced.robocasa.study_transcript import (
        MAX_HORIZON, InvalidPolicyOutput, parse_chunk_response)

    rows = json_loads(f"[{','.join(['[0,0,0,0,0,0,0]'] * (MAX_HORIZON + 1))}]")
    import json as _json

    with pytest.raises(InvalidPolicyOutput, match="exceeds the bound"):
        parse_chunk_response(_json.dumps(rows), dof=7)


def test_the_store_rejects_a_malformed_document(tmp_path):
    from examples.advanced.robocasa.study_transcript import TranscriptStore

    with pytest.raises(ValueError, match="not a sha256 digest"):
        TranscriptStore({"short": {"response": "[]"}})
    with pytest.raises(ValueError, match="no response"):
        TranscriptStore({"a" * 64: {"nope": 1}})
    with pytest.raises(FileNotFoundError):
        TranscriptStore.from_path(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 99, "calls": {}}')
    with pytest.raises(ValueError, match="unsupported transcript schema"):
        TranscriptStore.from_path(bad)


def test_a_transcript_round_trips_through_a_file(tmp_path):
    from examples.advanced.robocasa.study_transcript import TranscriptRecorder, TranscriptStore

    recorder = TranscriptRecorder(method=StillMethod(horizon=2))
    recorder.predict(_observation())
    path = recorder.store.write(tmp_path / "t.json")
    loaded = TranscriptStore.from_path(path)
    assert len(loaded) == 1 and len(loaded.source_sha256) == 64


def test_an_unset_transcript_path_fails_rather_than_degrading(monkeypatch):
    from examples.advanced.robocasa.study_transcript import (
        TRANSCRIPT_ENV, EnvTranscriptMethod)

    monkeypatch.delenv(TRANSCRIPT_ENV, raising=False)
    with pytest.raises(ValueError, match=TRANSCRIPT_ENV):
        EnvTranscriptMethod()


def test_the_ledger_keeps_the_deterministic_and_measured_halves_apart():
    """A caller that mixed them would produce a record that can never be
    compared across runs."""
    from examples.advanced.robocasa.study_telemetry import NONDETERMINISTIC_FIELDS
    from examples.advanced.robocasa.study_transcript import TranscriptRecorder

    recorder = TranscriptRecorder(method=StillMethod(horizon=2))
    recorder.predict(_observation())
    canonical, = recorder.ledger.canonical
    assert set(canonical) == {"call_index", "prompt_sha256", "horizon"}
    assert all(field not in canonical for field in NONDETERMINISTIC_FIELDS)
    assert "latency_s" in recorder.ledger.telemetry[0]


@pytest.mark.parametrize("field,value", (
    ("task", ""), ("method_id", ""),
    ("call_index", -1), ("call_index", 1.0),
    ("prompt_sha256", "short"), ("latency_s", -0.1), ("latency_s", 1),
    ("prompt_tokens", -1), ("usd", -0.01), ("usd", 1), ("horizon", -1),
))
def test_an_unusable_call_record_is_rejected(field, value):
    with pytest.raises(ValueError):
        _record(**{field: value})


def test_the_sidecar_round_trips_and_joins(tmp_path):
    from examples.advanced.robocasa.study_telemetry import (
        SIDECAR_NAME, read_sidecar, verify_join, write_sidecar)

    write_sidecar(tmp_path, [_record(call_index=0), _record(call_index=1)])
    assert (tmp_path / SIDECAR_NAME).is_file()
    rows = read_sidecar(tmp_path)
    ledger = [{"call_index": i, "prompt_sha256": "a" * 64} for i in (0, 1)]
    assert verify_join(ledger, rows)["joined"]


def test_a_digest_mismatch_breaks_the_join():
    """So a sidecar cannot be silently attached to another run's results."""
    from examples.advanced.robocasa.study_telemetry import verify_join

    verdict = verify_join([{"call_index": 0, "prompt_sha256": "b" * 64}],
                          [_record().to_dict()])
    assert not verdict["joined"]
    assert "prompt digest disagrees" in " ".join(verdict["errors"])


def test_a_gap_in_call_indices_breaks_the_join():
    from examples.advanced.robocasa.study_telemetry import verify_join

    rows = [_record(call_index=0).to_dict(), _record(call_index=2).to_dict()]
    ledger = [{"call_index": 0, "prompt_sha256": "a" * 64},
              {"call_index": 2, "prompt_sha256": "a" * 64}]
    assert "complete ordered sequence" in " ".join(verify_join(ledger, rows)["errors"])


def test_an_absent_sidecar_reads_as_empty(tmp_path):
    from examples.advanced.robocasa.study_telemetry import read_sidecar

    assert read_sidecar(tmp_path) == []


def test_absent_cost_is_reported_absent_rather_than_zero():
    """Summing a missing count as zero quietly understates the cost a study
    exists to measure."""
    from examples.advanced.robocasa.study_telemetry import projected_cost

    summary = projected_cost([_record().to_dict()])
    assert summary["prompt_tokens"] is None
    assert summary["prompt_tokens_reported_by"] == 0


def test_reported_cost_is_totalled_with_its_denominator():
    from examples.advanced.robocasa.study_telemetry import projected_cost

    rows = [_record(call_index=0, prompt_tokens=10, usd=0.002).to_dict(),
            _record(call_index=1).to_dict()]
    summary = projected_cost(rows)
    assert summary["prompt_tokens"] == 10 and summary["prompt_tokens_reported_by"] == 1
    assert summary["usd"] == 0.002 and summary["calls"] == 2


def test_the_two_latencies_are_separate_fields():
    """A replayed call's measured latency is a dictionary lookup; reporting it
    as model latency would understate it by orders of magnitude."""
    row = _record(latency_s=0.001, recorded_latency_s=2.4).to_dict()
    assert row["latency_s"] != row["recorded_latency_s"]


# --- the model boundary and the three policy forms ----------------------


def _config(**overrides):
    from examples.advanced.robocasa.study_model import ModelConfig

    return ModelConfig(**{"model_id": "test:model", **overrides})


def _answers(*texts):
    from examples.advanced.robocasa.study_model import OfflineModelClient

    return OfflineModelClient(answers=list(texts))


def _direct(client, **kwargs):
    from examples.advanced.robocasa.study_model import DirectModelMethod

    return DirectModelMethod(client=client, config=kwargs.pop("config", _config()),
                             dof=7, **kwargs)


@pytest.mark.parametrize("field,value", (
    ("model_id", ""), ("model_id", "  "), ("model_id", 7),
    ("prompt_variant", "nope"), ("reasoning_effort", "maximum"),
    ("horizon_band", "medium"),
    ("temperature", -0.1), ("temperature", 2.1), ("temperature", 0),
    ("timeout_seconds", 0.0), ("timeout_seconds", 60),
    ("max_response_bytes", 0), ("max_output_tokens", -1),
))
def test_an_unusable_model_config_is_rejected(field, value):
    with pytest.raises(ValueError):
        _config(**{field: value})


def test_the_horizon_bands_are_the_studys():
    from examples.advanced.robocasa.study_model import HORIZON_BANDS

    assert HORIZON_BANDS == {"short": (3, 15), "long": (16, 32)}
    assert _config(horizon_band="long").horizon_range == (16, 32)


def test_the_token_cap_is_the_studys():
    assert _config().max_output_tokens == 8_192


@pytest.mark.parametrize("variant", ("terse", "explained", "with_format"))
def test_every_prompt_variant_renders_with_the_action_width(variant):
    text = _config(prompt_variant=variant).instructions(dof=7)
    assert text.strip() and "{" not in text


@pytest.mark.parametrize("field,value", (
    ("prompt_variant", "terse"), ("model_id", "other:model"),
    ("reasoning_effort", "high"), ("horizon_band", "long"),
))
def test_each_prompt_axis_changes_the_digest(field, value):
    """Wired: the configuration is part of the hashed prompt, so two
    configurations cannot collide on one transcript entry."""
    from dataclasses import replace

    from examples.advanced.robocasa.study_model import assemble_model_prompt
    from examples.advanced.robocasa.study_transcript import prompt_digest

    observation = _observation()
    base = prompt_digest(assemble_model_prompt(observation, _config(), dof=7))
    altered = prompt_digest(assemble_model_prompt(
        observation, replace(_config(), **{field: value}), dof=7))
    assert base != altered


def test_carried_history_changes_the_digest():
    """Which is what makes Direct and Agentic distinguishable requests rather
    than the same call under two names."""
    from examples.advanced.robocasa.study_model import assemble_model_prompt
    from examples.advanced.robocasa.study_transcript import prompt_digest

    observation = _observation()
    without = prompt_digest(assemble_model_prompt(observation, _config(), dof=7))
    with_history = prompt_digest(assemble_model_prompt(
        observation, _config(), dof=7, history=({"call": 0, "horizon": 5},)))
    assert without != with_history


# --- every expected failure becomes a refusal ---------------------------


@pytest.mark.parametrize("status", ("timeout", "oversize", "refused", "transport_error"))
def test_a_non_ok_status_is_a_refusal(status):
    from examples.advanced.robocasa.study_model import ModelCallStatus, ModelResponse
    from examples.advanced.robocasa.study_transcript import InvalidPolicyOutput

    method = _direct(_answers(ModelResponse(status=ModelCallStatus(status), detail="why")))
    with pytest.raises(InvalidPolicyOutput, match=f"ended {status}"):
        method.predict(_observation())
    assert method.telemetry()[0]["parse_ok"] is False


def test_an_unparseable_answer_is_a_refusal():
    from examples.advanced.robocasa.study_transcript import InvalidPolicyOutput

    with pytest.raises(InvalidPolicyOutput):
        _direct(_answers("I would rather not.")).predict(_observation())


def test_a_chunk_outside_the_configured_band_is_a_refusal():
    """The horizon band is part of the configuration, so a model that ignores
    it has produced an answer the run did not ask for."""
    import json

    from examples.advanced.robocasa.study_transcript import InvalidPolicyOutput

    one_row = json.dumps([[0.0] * 7])
    with pytest.raises(InvalidPolicyOutput, match="outside the configured short band"):
        _direct(_answers(one_row)).predict(_observation())


def test_a_client_that_raises_is_a_refusal_naming_the_transport():
    from examples.advanced.robocasa.study_model import OfflineModelClient
    from examples.advanced.robocasa.study_transcript import InvalidPolicyOutput

    class _Exploding(OfflineModelClient):
        def complete(self, prompt, config):
            raise ConnectionResetError("peer hung up")

    with pytest.raises(InvalidPolicyOutput, match="transport_error"):
        _direct(_Exploding()).predict(_observation())


def test_an_oversize_answer_is_a_refusal():
    import json

    from examples.advanced.robocasa.study_transcript import InvalidPolicyOutput

    rows = json.dumps([[0.0] * 7] * 5)
    method = _direct(_answers(rows), config=_config(max_response_bytes=10))
    with pytest.raises(InvalidPolicyOutput, match="exceeds its configured bound"):
        method.predict(_observation())


def test_the_base_protocol_refuses_to_be_used_directly():
    from examples.advanced.robocasa.study_model import ModelClient

    with pytest.raises(NotImplementedError):
        ModelClient().complete("x", _config())


def test_a_non_client_reference_is_rejected():
    from examples.advanced.robocasa.study_model import load_model_client

    with pytest.raises(TypeError, match="complete"):
        load_model_client("examples.advanced.robocasa.study_model:PROMPT_VARIANTS")
    with pytest.raises(ValueError, match="module:symbol"):
        load_model_client("examples.advanced.robocasa.study_model.OfflineModelClient")


# --- the three forms ----------------------------------------------------


def test_all_three_forms_produce_a_chunk_with_no_key():
    from examples.advanced.robocasa.study_model import offline_forms

    forms = offline_forms(lambda: StillMethod(horizon=5))
    assert set(forms) == {"direct", "agentic", "orchestrated"}
    for method in forms.values():
        chunk = method.predict(_observation())
        assert chunk.horizon == 5 and chunk.dof == 7


def test_the_three_forms_agree_against_the_offline_client():
    """They answer from the same wrapped method, so a difference would mean the
    harness alters behaviour rather than the form being measured."""
    from examples.advanced.robocasa.study_model import offline_forms

    forms = offline_forms(lambda: StillMethod(horizon=5))
    chunks = [method.predict(_observation()).actions for method in forms.values()]
    assert all(np.allclose(chunks[0], other) for other in chunks[1:])


def test_the_agentic_form_carries_history_across_calls():
    from examples.advanced.robocasa.study_model import AgenticModelMethod, OfflineModelClient

    method = AgenticModelMethod(
        client=OfflineModelClient(method=StillMethod(horizon=5)), config=_config(),
        dof=7, memory=2)
    for _ in range(3):
        method.predict(_observation())
    # Three calls, and the prompts differ because the carried window grew.
    prompts = method.client.calls
    assert len(prompts) == 3 and len(set(prompts)) == 3


def test_the_agentic_form_at_zero_memory_matches_direct():
    """Informationally identical by construction, so a reported difference
    between them at memory 0 would be noise."""
    from examples.advanced.robocasa.study_model import (
        AgenticModelMethod, DirectModelMethod, OfflineModelClient)

    direct = DirectModelMethod(
        client=OfflineModelClient(method=StillMethod(horizon=5)), config=_config(), dof=7)
    agentic = AgenticModelMethod(
        client=OfflineModelClient(method=StillMethod(horizon=5)), config=_config(),
        dof=7, memory=0)
    direct.predict(_observation())
    agentic.predict(_observation())
    assert direct.client.calls == agentic.client.calls


@pytest.mark.parametrize("memory", (-1, 17, 1.0, None))
def test_an_unusable_memory_depth_is_rejected(memory):
    from examples.advanced.robocasa.study_model import AgenticModelMethod, OfflineModelClient

    with pytest.raises(ValueError):
        AgenticModelMethod(client=OfflineModelClient(), config=_config(), memory=memory)


def test_the_orchestrated_form_records_every_review():
    """The intervention rate has to be readable without re-running anything."""
    from examples.advanced.robocasa.study_model import offline_forms

    method = offline_forms(lambda: StillMethod(horizon=5))["orchestrated"]
    method.predict(_observation())
    review, = method.reviews
    assert review["proposed_steps"] == 5 and review["committed_steps"] == 5
    assert review["intervened"] is False


def test_the_orchestrated_form_falls_back_to_the_proposal_on_a_refusal():
    """A reviewer that cannot answer must not fail the episode: the base
    policy's proposal already exists and is what it was reviewing."""
    from examples.advanced.robocasa.study_model import OfflineModelClient, OrchestratedModelMethod

    method = OrchestratedModelMethod(
        client=OfflineModelClient(answers=["nonsense"]), config=_config(), dof=7,
        base=StillMethod(horizon=5))
    chunk = method.predict(_observation())
    assert chunk.horizon == 5
    review, = method.reviews
    assert review["intervened"] is False and review["reason"]


def test_the_orchestrated_form_records_an_intervention():
    import json

    from examples.advanced.robocasa.study_model import OfflineModelClient, OrchestratedModelMethod

    shorter = json.dumps([[0.0] * 7] * 3)
    method = OrchestratedModelMethod(
        client=OfflineModelClient(answers=[shorter]), config=_config(), dof=7,
        base=StillMethod(horizon=8))
    chunk = method.predict(_observation())
    assert chunk.horizon == 3
    review, = method.reviews
    assert review["proposed_steps"] == 8 and review["committed_steps"] == 3
    assert review["intervened"] is True


def test_the_orchestrated_proposal_length_is_the_studys():
    from examples.advanced.robocasa.study_model import ORCHESTRATED_PROPOSAL_STEPS

    assert ORCHESTRATED_PROPOSAL_STEPS == 15


def test_an_orchestrated_method_needs_a_base():
    from examples.advanced.robocasa.study_model import OfflineModelClient, OrchestratedModelMethod

    with pytest.raises(TypeError, match="base with predict"):
        OrchestratedModelMethod(client=OfflineModelClient(), config=_config())


# --- the ledger -----------------------------------------------------------


def test_the_model_ledger_names_the_configuration_and_nothing_measured():
    from examples.advanced.robocasa.study_model import OfflineModelClient
    from examples.advanced.robocasa.study_telemetry import NONDETERMINISTIC_FIELDS

    method = _direct(OfflineModelClient(method=StillMethod(horizon=5)))
    method.predict(_observation())
    entry, = method.canonical_ledger()
    assert entry["model_config"] == _config().to_dict()
    assert all(field not in entry for field in NONDETERMINISTIC_FIELDS)
    assert "latency_s" in method.telemetry()[0]


def test_a_call_can_be_banked_for_free_replay():
    """So one paid sweep replays offline afterwards."""
    from examples.advanced.robocasa.study_model import OfflineModelClient
    from examples.advanced.robocasa.study_transcript import TranscriptStore

    store = TranscriptStore({})
    method = _direct(OfflineModelClient(method=StillMethod(horizon=5)), record_into=store)
    method.predict(_observation())
    assert len(store) == 1


def test_a_refused_call_is_not_banked():
    """A transcript of answers that do not parse would replay as refusals and
    read as a model failure rather than a recording failure."""
    from examples.advanced.robocasa.study_transcript import InvalidPolicyOutput, TranscriptStore

    store = TranscriptStore({})
    method = _direct(_answers("nonsense"), record_into=store)
    with pytest.raises(InvalidPolicyOutput):
        method.predict(_observation())
    assert len(store) == 0


def test_the_prompt_axes_are_wired_but_not_live():
    """Asserting both halves so neither can be read as a finding: the digest
    changes with the configuration, and the offline client's answer does not."""
    from dataclasses import replace

    from examples.advanced.robocasa.study_model import (
        PROMPT_VARIANTS, OfflineModelClient, assemble_model_prompt)
    from examples.advanced.robocasa.study_transcript import prompt_digest

    observation = _observation()
    configs = [replace(_config(), prompt_variant=name) for name in sorted(PROMPT_VARIANTS)]
    digests = {prompt_digest(assemble_model_prompt(observation, config, dof=7))
               for config in configs}
    assert len(digests) == len(configs), "the axis does not reach the model boundary"

    answers = set()
    for config in configs:
        method = _direct(OfflineModelClient(method=StillMethod(horizon=5)), config=config)
        answers.add(method.predict(observation).actions.tobytes())
    assert len(answers) == 1, "the offline client is supposed to ignore the configuration"
