"""Simulator-backed tests for the arm IK, skipped without RoboCasa.

These need a live MuJoCo model, so they are separated from the simulator-free
suite in `test_robocasa_study.py`. What they pin is the claim the module is
committed for: the object is reachable and the solver finds a clear posture for
it, which is what retires the earlier theory that these tasks need the base to
move. They also pin the three joint-control facts that are not in any
documentation.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

ROBOCASA = importlib.util.find_spec("robocasa") is not None
pytestmark = pytest.mark.skipif(not ROBOCASA, reason="needs the robocasa environment")

TASK = "PickPlaceCounterToCabinet"


@pytest.fixture(scope="module")
def arm_env():
    import robocasa  # noqa: F401 - registers the task suite
    import robosuite

    from examples.advanced.robocasa.study_ik import ArmModel, joint_position_controller_config

    env = robosuite.make(TASK, robots="PandaOmron",
                         controller_configs=joint_position_controller_config(),
                         has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, seed=0, layout_ids=[1], style_ids=[1])
    env.reset()
    yield env, ArmModel(env=env)
    env.close()


def _object_position(env):
    import mujoco

    model, data = env.sim.model._model, env.sim.data._data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, env.objects["obj"].root_body)
    return np.asarray(data.xpos[body]).copy()


def test_the_controller_config_asks_for_absolute_joint_targets():
    """The default drives the arm in operational space, which means trusting a
    solve that stalls 0.288 m short."""
    from examples.advanced.robocasa.study_ik import joint_position_controller_config

    config = joint_position_controller_config()
    arm = config["body_parts"]["right"]
    assert arm["type"] == "JOINT_POSITION" and arm["input_type"] == "absolute"
    # `output_max` clips the absolute target, so it must span the joint ranges.
    assert arm["output_max"] >= 3.0 and arm["output_min"] <= -3.0


def test_the_arm_model_resolves_against_a_live_model(arm_env):
    _, arm = arm_env
    assert arm.site >= 0
    assert len(arm.dof) == len(arm.qadr) == 7
    assert arm.limits.shape == (7, 2)
    assert np.all(arm.limits[:, 0] < arm.limits[:, 1])


def test_the_jacobian_is_full_rank_at_the_start_posture(arm_env):
    """Full rank means the arm has the freedom to move the hand in any
    direction from here, which is why the controller's stall was never a
    kinematic limit."""
    _, arm = arm_env
    jacobian = arm.jacobian()
    assert jacobian.shape == (3, 7)
    assert np.linalg.matrix_rank(jacobian) == 3


def test_one_damped_step_predicts_the_motion_it_asks_for(arm_env):
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import damped_step

    target = _object_position(env)
    increment, residual = damped_step(arm, target)
    assert residual > 0.3, "the object should start out of easy reach"
    predicted = arm.jacobian() @ increment
    desired = target - arm.hand()
    # The step is capped, so compare directions rather than magnitudes.
    cosine = float(predicted @ desired / (np.linalg.norm(predicted) * np.linalg.norm(desired)))
    assert cosine > 0.99


def test_the_object_is_reachable_without_moving_the_base(arm_env):
    """The headline measurement. A clear posture exists for a point just above
    the object from the robot's existing base position, so the earlier theory
    that these tasks require base navigation was wrong."""
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import solve_posture

    solution = solve_posture(arm, _object_position(env) + np.array([0.0, 0.0, 0.09]))
    assert solution.reached, solution.to_dict()
    assert solution.clear, solution.to_dict()
    assert solution.residual_m < 0.01


def test_solving_restores_the_simulation_state(arm_env):
    """It is a query, not a move: a solve that left the arm elsewhere would
    corrupt the episode it was called from."""
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import solve_posture

    before = arm.joints().copy()
    hand_before = arm.hand().copy()
    solve_posture(arm, _object_position(env) + np.array([0.0, 0.0, 0.09]), restarts=4)
    assert np.allclose(arm.joints(), before, atol=1e-9)
    assert np.allclose(arm.hand(), hand_before, atol=1e-9)


def test_forward_kinematics_confirms_the_solution(arm_env):
    """The solver's own residual could be self-consistent and wrong; this
    checks it against the model independently."""
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import _hand_at, solve_posture

    target = _object_position(env) + np.array([0.0, 0.0, 0.09])
    solution = solve_posture(arm, target)
    assert np.linalg.norm(_hand_at(arm, solution.joints) - target) < 0.01


def test_the_solution_respects_the_joint_limits(arm_env):
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import solve_posture

    solution = solve_posture(arm, _object_position(env) + np.array([0.0, 0.0, 0.09]))
    assert np.all(solution.joints >= arm.limits[:, 0])
    assert np.all(solution.joints <= arm.limits[:, 1])


def test_obstructions_ignore_contact_with_the_target_object(arm_env):
    """Touching the jar is the point; touching a cabinet is what blocks a
    path, and only the second should count."""
    _, arm = arm_env
    assert arm.object_geoms, "the target object's geoms should be identified"
    assert not (arm.object_geoms & arm.robot_geoms)
    names = {name for name in arm.object_geoms}
    for first, second in arm.obstructions():
        import mujoco

        model = arm.env.sim.model._model
        ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, first),
               mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, second)}
        assert not (ids & names), f"{first}/{second} is the target object, not an obstruction"


@pytest.mark.parametrize("bad", ({"restarts": 0}, {"restarts": 501},
                                  {"iterations": 0}, {"iterations": 2001}))
def test_an_unusable_solver_budget_is_rejected(arm_env, bad):
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import solve_posture

    with pytest.raises(ValueError):
        solve_posture(arm, _object_position(env), **bad)


def test_the_droop_compensator_accumulates_and_saturates():
    """A saturated correction is the signal that a joint is blocked rather
    than drooping, which is how the two failures are told apart."""
    from examples.advanced.robocasa.study_ik import DROOP_LIMIT, DroopCompensator

    compensator = DroopCompensator()
    compensator.reset(7)
    want, stuck = np.ones(7), np.zeros(7)
    assert not compensator.wound_up
    for _ in range(200):
        command = compensator.command(want, stuck)
    assert compensator.wound_up
    assert np.all(np.abs(command - want) <= DROOP_LIMIT + 1e-9)


def test_the_droop_compensator_is_inert_when_tracking_is_exact():
    from examples.advanced.robocasa.study_ik import DroopCompensator

    compensator = DroopCompensator()
    compensator.reset(7)
    want = np.full(7, 0.3)
    for _ in range(10):
        command = compensator.command(want, want)
    assert np.allclose(command, want)
    assert not compensator.wound_up
