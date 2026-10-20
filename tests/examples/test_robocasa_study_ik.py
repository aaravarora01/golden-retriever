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


# --- motion planning -----------------------------------------------------


def test_the_direct_path_to_the_goal_is_blocked(arm_env):
    """The premise the planner exists for: both endpoints are clear and the
    straight line between them is not."""
    import mujoco

    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import solve_posture

    model, data = env.sim.model._model, env.sim.data._data
    goal = solve_posture(arm, _object_position(env) + np.array([0.0, 0.0, 0.09]))
    saved = arm.joints().copy()

    def clashes(joints):
        for address, value in zip(arm.qadr, joints):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        return len(arm.obstructions())

    try:
        assert clashes(goal.joints) == 0, "the goal posture should be clear"
        worst = max(clashes(saved + (goal.joints - saved) * (i / 20)) for i in range(21))
        assert worst > 0, "if the straight line were clear, no planner would be needed"
    finally:
        clashes(saved)


def test_the_planner_finds_a_collision_free_path(arm_env):
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import plan_path, solve_posture

    goal = solve_posture(arm, _object_position(env) + np.array([0.0, 0.0, 0.09]))
    path = plan_path(arm, goal.joints, seed=0)
    assert path.found, path.to_dict()
    assert len(path.waypoints) >= 2
    # Smoothing should collapse an RRT's wandering, because a wandering path
    # costs control steps, which is one of the study's reported metrics.
    assert path.smoothed_from > len(path.waypoints)


def test_every_planned_segment_is_collision_free(arm_env):
    """The property that makes the path worth having, checked independently of
    the planner's own bookkeeping."""
    import mujoco

    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import plan_path, solve_posture

    model, data = env.sim.model._model, env.sim.data._data
    goal = solve_posture(arm, _object_position(env) + np.array([0.0, 0.0, 0.09]))
    path = plan_path(arm, goal.joints, seed=0)
    saved = arm.joints().copy()

    def clashes(joints):
        for address, value in zip(arm.qadr, joints):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        return len(arm.obstructions())

    try:
        # The start posture self-collides at reset, so the first waypoint is
        # exempt; every segment after it must be clear.
        for first, second in zip(path.waypoints[1:], path.waypoints[2:]):
            for index in range(1, 9):
                assert clashes(first + (second - first) * (index / 8)) == 0
    finally:
        clashes(saved)


def test_planning_restores_the_simulation_state(arm_env):
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import plan_path, solve_posture

    goal = solve_posture(arm, _object_position(env) + np.array([0.0, 0.0, 0.09]))
    before, hand_before = arm.joints().copy(), arm.hand().copy()
    plan_path(arm, goal.joints, iterations=300)
    assert np.allclose(arm.joints(), before, atol=1e-9)
    assert np.allclose(arm.hand(), hand_before, atol=1e-9)


def test_the_planner_reports_failure_rather_than_looping(arm_env):
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import plan_path, solve_posture

    goal = solve_posture(arm, _object_position(env) + np.array([0.0, 0.0, 0.09]))
    path = plan_path(arm, goal.joints, iterations=2, seed=7)
    assert path.found is False and path.waypoints == []


def test_a_goal_in_collision_is_refused_before_planning(arm_env):
    """Planning to an unreachable posture would burn the whole budget."""
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import plan_path

    buried = arm.limits[:, 0] + 1e-3
    path = plan_path(arm, buried, iterations=50)
    assert path.found is False and path.iterations == 0


@pytest.mark.parametrize("bad", ({"iterations": 0}, {"step": 0.0}, {"step": 2.0},
                                  {"step": 1}))
def test_an_unusable_planner_budget_is_rejected(arm_env, bad):
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import plan_path

    with pytest.raises(ValueError):
        plan_path(arm, arm.joints(), **bad)


# --- following a planned path --------------------------------------------


def _commander(env, arm, grip=-1.0):
    """Step the simulation with a joint target, leaving other segments alone."""
    def command(target):
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[0:7] = arm.clamp(target)
        action[7] = grip
        action[-1] = -1.0
        env.step(action)
    return command


def test_following_a_planned_path_reaches_the_goal_without_collisions(arm_env):
    """The result this module exists for. Commanding waypoints directly leaves
    the later ones 1.8-2.1 rad short and collides; gating the commanded point
    on the arm keeping up follows the line the planner verified."""
    env, arm = arm_env
    env.reset()
    arm = type(arm)(env=env)
    from examples.advanced.robocasa.study_ik import follow_path, plan_path, solve_posture

    target = _object_position(env) + np.array([0.0, 0.0, 0.09])
    goal = solve_posture(arm, target)
    assert goal.clear
    path = plan_path(arm, goal.joints, seed=0)
    assert path.found
    # The reset posture self-collides, so the bar is that execution introduces
    # no *new* contact, not that there is none.
    at_start = len(arm.obstructions())
    report = follow_path(arm, path.waypoints, command=_commander(env, arm))
    assert report.worst_obstructions <= at_start, report.to_dict()
    assert not report.stalled, report.to_dict()
    assert report.joint_error < 0.05, report.to_dict()
    # The payoff: the gripper is at the planned point, from 0.512 m at reset.
    assert float(np.linalg.norm(target - arm.hand())) < 0.01


def test_a_report_says_whether_the_verified_path_was_actually_followed(arm_env):
    """`worst_obstructions` is the honest measure: a non-zero value means the
    executed motion left the checked line, whatever the final error says."""
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import FollowReport

    assert FollowReport(worst_obstructions=0, stalled=False).to_dict()["clean"] is True
    assert FollowReport(worst_obstructions=2, stalled=False).to_dict()["clean"] is False
    assert FollowReport(worst_obstructions=0, stalled=True).to_dict()["clean"] is False


def test_following_needs_at_least_a_start_and_a_target(arm_env):
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import follow_path

    with pytest.raises(ValueError, match="start and a target"):
        follow_path(arm, [arm.joints()], command=_commander(env, arm))


@pytest.mark.parametrize("bad", ({"increment": 0.0}, {"increment": 1.0},
                                  {"tolerance": 0.0}, {"tolerance": 2.0},
                                  {"increment": 1}))
def test_an_unusable_follow_setting_is_rejected(arm_env, bad):
    env, arm = arm_env
    from examples.advanced.robocasa.study_ik import follow_path

    with pytest.raises(ValueError):
        follow_path(arm, [arm.joints(), arm.joints() + 0.1],
                    command=_commander(env, arm), **bad)


def test_the_controller_tracks_a_large_clear_joint_move(arm_env):
    """Pins the corrected finding: this is not a drooping controller. A 1.0 rad
    move on a joint whose path is clear arrives, because gravity compensation is
    already on. What fails is a move that drives the arm into contact."""
    env, arm = arm_env
    env.reset()
    arm = type(arm)(env=env)
    command = _commander(env, arm)
    start = arm.joints().copy()
    target = start.copy()
    target[3] = np.clip(start[3] + 1.0, arm.limits[3, 0] + 0.05, arm.limits[3, 1] - 0.05)
    for _ in range(400):
        command(target)
    assert abs(float(arm.joints()[3] - target[3])) < 0.02


def test_an_absolute_joint_command_is_not_clipped_by_the_action_spec(arm_env):
    """Corrects an earlier claim in this module: the spec advertises +/-1 and
    the controller still receives the value verbatim."""
    env, arm = arm_env
    env.reset()
    arm = type(arm)(env=env)
    low, high = env.action_spec
    assert high[0] <= 1.0 + 1e-9, "the spec does advertise a narrow range"
    controller = env.robots[0].composite_controller.part_controllers["right"]
    target = arm.joints().copy()
    target[0] = 2.5
    _commander(env, arm)(target)
    assert float(controller.goal_qpos[0]) == pytest.approx(2.5, abs=1e-6)
