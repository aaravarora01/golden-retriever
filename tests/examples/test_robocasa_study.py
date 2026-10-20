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
