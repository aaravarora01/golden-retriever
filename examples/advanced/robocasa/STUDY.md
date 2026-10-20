# Policy study on RoboCasa pick-and-place

Reproduces the *measurement design* of NVIDIA SRL's
[Astranomy](https://research.nvidia.com/labs/srl/projects/astranomy/#study-design)
study on RoboCasa's pick-and-place suite, and puts the backend behind a Protocol
so the same sweep runs against [RoboLab-120](https://github.com/NVlabs/RoboLab)
on a GPU machine without being rewritten.

```bash
pixi run -e robocasa study-pnp-baselines   # per-method summary
pixi run -e robocasa study-pnp-sweep       # with every episode row
pixi run -e robocasa study-pnp-vague       # the task-specification axis
pixi run -e robocasa test-robocasa-study   # simulator-free, 188 tests
pixi run -e robocasa test-robocasa-study-ik # simulator-backed, 54 tests
```

## What this is not

The study evaluates **RoboLab-120 on the DROID robot**, and *"all simulations ran
on NVIDIA L40s GPUs"*. RoboLab needs an NVIDIA RTX GPU with 48 GB+ VRAM
recommended, Ubuntu 22.04+ and Isaac Sim 5.0/5.1; there is no CPU fallback and no
Apple Silicon support. RoboCasa is therefore a **substitute, not the benchmark**:
different simulator, robot and task set, so absolute numbers are not comparable
with published ones and nothing here claims they are.

**No model is called anywhere in this lane**, so there is no policy result yet —
only the measurement spine and the baselines a policy would be read against.

## Built on what was already here

The lane extends `method_harness`, this folder's existing contract, rather than
sitting beside it: `PolicyObservation` in, a `SafetyEnvelope`-validated
`ActionChunk` out, one control step dispatched per chunk row. That dispatch loop
*is* the study's action-chunking mechanism, and its `chunk_dispatched` event is
the model-call count, so neither had to be invented.

| Module | Role |
| --- | --- |
| `study_robocasa_env` | RoboCasa PnP behind the harness Protocol — runs on CPU |
| `study_robolab_env` | RoboLab-120 behind the same Protocol — the GPU port |
| `study_representations` | The action-representation axis and the per-step caps |
| `study_observation` | Views and history tiled into one `PolicyObservation` |
| `study_progress` | The 0–100 partial score RoboCasa does not provide |
| `study_methods` | `StillMethod` and `RandomMethod`, the two baselines |
| `study_scripted` | A privileged waypoint reference (blocked; see below) |
| `study_sweep` | task × seed × method, with the three counts kept apart |
| `study_tasks` | The 18-task committed suite and the specification axis |
| `study_telemetry` | Latency, tokens and cost, beside the results not in them |
| `study_transcript` | Record and replay calls keyed by prompt digest |
| `study_model` | The model call boundary and the three policy forms |
| `study_ik` | IK, posture solving, droop compensation, RRT-Connect planning |

## Axis coverage

| Study axis | Here | Status |
| --- | --- | --- |
| Action representation | `eef_delta`, `eef_absolute`, `eef_delta_base` on RoboCasa; `joint_velocity`, `joint_position` on RoboLab | Live; the two backends sit at different points on it |
| Action horizon | `horizon` on a method; chunk rows per call | Live — trades control steps against model calls |
| Observation content | `ObservationConfig.views` | Live |
| Observation history | `ObservationConfig.history` | Live |
| Task specification | `default` vs derived `vague` | Live |
| Prompt content / model / effort / horizon band | `ModelConfig`, hashed into the prompt | **Wired, not live** — needs a model to move behaviour |
| Direct / Agentic / Orchestrated | `study_model`'s three methods | Mechanisms live; a result needs a model |
| 120 tasks × 5 episodes, seeds 0–4 | 18 committed tasks × seeds 0–4 | Protocol matches; this is the ablation-suite scale |
| Video demonstration | — | Needs demonstrations; the dataset links are dead |

Metrics follow the study: success rate, a 0–100 `Score`, per-call latency, and
**model calls per episode**. Control steps, model calls and episodes are three
separate counts and are never summed — the study reports the second, and a
longer horizon reduces it while leaving the first unchanged.

## Measured facts, not assumptions

Each of these was found by running the simulator and each one bounds what can be
measured.

**RoboCasa registration is an import side effect.** The 38 pick-and-place
environments appear in RoboSuite's registry only once `robocasa` is imported;
without it `robosuite.make` reports the task as unknown among 19 base
environments.

**Layout and style ids are 1-based and must be lists.** `layout_ids=0` raises a
`KeyError` inside the scene registry — 0 is not a `LayoutType`, the negatives are
group selectors, and concrete layouts start at 1.

**The action vector is 12-wide and mostly not the arm.** `PandaOmron` under
`HybridMobileBase` splits as arm `0:6` (an operational-space delta), gripper
`6:7`, base `7:10`, torso `10:11`, with a trailing `base_mode` flag: above zero
drives the base and updates the arm goal from its *desired* pose, at or below
zero drives the arm from its *achieved* pose.

**Plain `Panda` is not an option.** RoboCasa's kitchens expect the mobile-base
camera rig and fail on a missing `robot0_agentview_center`.

**`WHOLE_BODY_IK` changes the representation, not just the solver.** Through the
flat action vector it takes 7 *absolute joint* targets plus a gripper, which is
`joint_position`; its `ik_input_ref_frame` is `mobilebase0_base`, so nothing
there is a world-frame pose target.

**Freezing the base makes these tasks unsolvable.** Across the eighteen
committed tasks the object starts 0.305–0.878 m from the end-effector. A Cartesian
controller driven straight at it stalls — 0.337 m under OSC with an aggressive
gain, 0.288 m with the per-step cap applied correctly — with no joint at a limit.
The cause is orientation rather than reach: `init_robot_base_ref` places the base
facing the *target* fixture, so the object sits outside the arm's workspace, and
the base cannot drive toward the counter because it is already against it.
Closing the gap needs base yaw plus translation. `eef_delta_base` and
`command_base` exist so a run is not measuring an arm against a mobile-manipulator
task, and `describe()` records `frozen_segments` either way, because a run that
froze the base measured something other than what it claimed.

## Baseline results

Numbers live in the PR description rather than here, but the shape is the point: both baselines
report a 0.00 success rate and a 0 score with no rung reached, 200 control steps
against 40 model calls, and the random method saturates the per-step cap on every
row while the still method never touches it. That is what the baselines are for —
the floor is a real floor, the scorer credits nothing that did not happen, and the
clip ledger separates a policy that constantly saturates from one that does not.

## The model seam

`study_model` is the boundary a frontier model plugs into, shaped after
`fast_downward.py` in the sibling drawer harness: a serializable config with
explicit bounds, a status enum rather than exceptions for expected outcomes, and
the response captured whole. A non-ok status, an oversize answer, an unparseable
answer, a chunk outside the configured horizon band and a transport exception
are one thing from the harness's side — no dispatchable chunk — so each becomes
a refusal that is recorded before it is raised.

No provider is shipped, so the default test path needs no key.
`OfflineModelClient` answers from a wrapped method, which is what lets the whole
path run in CI. Supplying a provider is one class:

```python
class MyProvider(ModelClient):
    def __init__(self):
        self.client = ...   # construct the SDK client here, never at import time

    def complete(self, prompt, config):
        try:
            reply = ...     # one call, honouring config.timeout_seconds
        except TimeoutError:
            return ModelResponse(status=ModelCallStatus.TIMEOUT, detail="...")
        return ModelResponse(text=reply.text, model_id=config.model_id,
                             prompt_tokens=reply.usage.input_tokens,
                             completion_tokens=reply.usage.output_tokens,
                             usd=..., latency_s=...)
```

The three forms differ as the study defines them. **Direct** is stateless.
**Agentic** carries a window of its own prior calls, which is what the
context-history axis ablates; at memory 0 it is informationally Direct, which is
stated in the contract and pinned by a test rather than reported as a
difference. **Orchestrated** reviews a base method's fixed 15-step proposal and
either accepts it or substitutes a shorter segment, recording every review so
the intervention rate is readable; a reviewer that cannot answer falls back to
the proposal rather than failing the episode.

**The prompt-shaped axes are wired, not live.** The prompt variant, model id,
reasoning effort and horizon band all enter the hashed prompt, so each provably
changes the digest and two configurations cannot collide on one transcript
entry. Whether a model's behaviour changes with them is unknowable without a
model, and the offline client ignores the configuration by design — so a sweep
over variants against it returns identical rows, which is a tautology rather
than a null result. The tests assert both halves on purpose.

`record_into` banks successful calls so one paid sweep replays offline for free
thereafter. A refused call is not banked: a transcript of answers that do not
parse would replay as refusals and read as a model failure rather than a
recording failure.

## What is incomplete, and why

**No policy reaches a nonzero success rate, and the obstacle is control rather
than measurement.** `study_scripted` reaches, descends, grasps, lifts and
releases from privileged object pose under the study's per-step caps, but it
stalls in `approach_above`. Base navigation was then built and probed, and these
are the measurements:

* Yawing the base to face the object **works** — bearing 0.43 → −0.026 rad, the
  object ends dead ahead in the base frame.
* Driving forward **does not close range** — 0.534 m → 0.515 m. The base is
  already against the counter, and `[1,0,0]` moves the end-effector +0.024 m in
  twenty steps while `[-1,0,0]` moves it 0.684 m: forward is blocked.
* `OSC_POSE` stalls at 0.337 m with an aggressive gain and 0.288 m with the cap
  applied correctly, no joint at a limit.
* `WHOLE_BODY_IK`'s 7-wide `right` segment is an IK target the solver converts,
  not joint pass-through — commanding element 0 alone moves all seven joints —
  and all three documented layouts (position with zero rotation, position with
  axis-angle, position with quaternion) reach 0.32 m transiently then diverge to
  about 1.44 m.

Jacobian IK was then built (`study_ik`), and it moved the diagnosis twice.

**The object is reachable from the robot's existing base position.** Brute-force
sampling of arm postures finds one within 0.052 m without moving the base, and
`solve_posture` refines that to a residual of 0.0008 m with zero obstructions,
confirmed by independent forward kinematics. The arm's position Jacobian is full
rank at the start posture, so the controller's stall was never a kinematic
limit. **The base-navigation theory above is therefore wrong**, and it is left in
place because it was the working hypothesis for several of the measurements.

**What blocks a reference policy is motion planning, not reaching.** The start
posture and the goal posture are both collision-free; the straight line between
them is not. The arm self-collides at reset (`link5` against the gripper hand)
and the gripper wedges under `cab_1_main_group_bottom` along the way, and
commanding the goal directly leaves the shoulder pitch and elbow 0.400 and 0.598
rad short. That is physical blocking rather than control error: integral
correction drives the other five joints to 0.002 rad and those two do not move,
which is exactly what `DroopCompensator.wound_up` reports.

A collision-checked planner was then built. `plan_path` is bidirectional
RRT-Connect in joint space, with every extension verified by interpolated
collision checks, and it **works**: it finds a clear path in a few thousand
iterations and smoothing collapses it from ~74 waypoints to 5. Single-waypoint
routing had failed — 400 random intermediate postures produced no clear two-leg
path — but that was straight-line segments clipping obstacles rather than a
cramped space: 54% of sampled joint space is collision-free, and 95% of a small
neighbourhood around the start posture is. The planner is bidirectional because
the free space at the goal is much tighter, 35% against 95%.

**Trajectory tracking is now solved, and the reach is.** `follow_path` commands
densely along each planned segment and advances the commanded point only as the
arm keeps up. With it the gripper reaches a point 9 cm above the object to
within **0.7 mm with no new contact**, from 0.512 m at reset.

Isolating that corrected two claims an earlier version of this file made.
Gravity compensation was **already on** — robosuite's joint controller computes
`mass_matrix @ (kp·e + kd·ė) + qfrc_bias` and tracks a clear 1.0 rad single-joint
move to 0.0008 rad — so there was no droop to fix. And `output_max` does **not**
clip an absolute target: it scales the delta path only, and `env.step` does not
clip to `action_spec` either, so a commanded 2.5 rad arrives as 2.5 even though
the spec advertises ±1. What looked like a drooping controller was a *blocked*
joint: commanding the shoulder +1.0 rad drives the arm into contact and the
torque then saturates against the actuator limit while the joint does not
arrive.

The real fault was subtler. The planner verifies the straight line between
waypoints, but *commanding* a waypoint lets the arm find its own dynamic route
there, which is not the line that was checked. Ramping relative to the achieved
pose stalls, because the commanded point never gets ahead of a lagging arm; a
fixed time schedule runs away from it, leaving the arm up to 0.8 rad behind and
colliding with things the kinematic check called clear. Gating the advance on
the arm keeping up is what makes the executed motion follow the verified path.

**Orientation is solved, and the fix was not the one expected.**

| target | selection | restarts | position error | tilt from vertical | clear |
| --- | --- | --- | --- | --- | --- |
| 8.5 cm above | none | 30 | 0.0001 m | 49.1° | yes |
| 8.5 cm above | top-down | 200 | 0.0003 m | **13.6°** | yes |
| 2.5 cm above | none | 30 | 0.0001 m | 56.1° | **no** |
| 2.5 cm above | top-down | 200 | 0.0020 m | **11.9°** | yes |

With seven joints and a three-axis position target the solution manifold is
four-dimensional, and across restarts the gripper orientation spans 13° to 159°.
So a solve that ignores orientation returns whichever posture it landed on —
measured at 49–56° off vertical — while near-vertical postures were in the
search all along. Scoring orientation when *selecting* among position-accurate
restarts finds them, at a cost of under two millimetres of position, and at
grasp height it also turns a colliding solution into a clear one.

Steering orientation *inside* the solve was tried first and all three variants
were worse. A weighted second task costs position outright: 0.39 m against
0.0008 m. Projecting into the nullspace of the position task does not help
either — the exact projector leaks nothing (`‖Jp·N‖ = 0` with a true
pseudoinverse, against 0.003 per step with a damped one), but the clamped joint
motion no longer preserves position, and clipping the two tasks separately did
not recover it. Selection among restarts needs none of that machinery.

The restart budget bounds what orientation is findable: 60 restarts settled for
34.4° where 200 reached 13.6°, so `ORIENTATION_RESTARTS` is wider than the
default that suffices for position alone. Clear solutions also thin out near the
object — 27 of 200 at the approach height against 4 of 200 at grasp height.

**The pick works.** Approach, descend, close, lift: the object leaves the counter
by 14.5 cm and is still held at the top, three runs out of three,
deterministically. That is **reached, grasped and lifted — three of the four
progress rungs, a Score of 75**, where every arm previously scored zero. The
task's own `success` predicate is still false because it is pick-and-*place*
into the cabinet and nothing transports or releases yet.

Three things stood between the reach and the pick, and the first was a bug in
this lane's own planner. **Bidirectional path reconstruction depended on which
tree grew last**: tree 0 is rooted at the start and tree 1 at the goal, and the
assembly reasoned about which had just grown instead of naming the connecting
node on each side, so on odd iterations it returned a path that did not end at
the goal. `follow_path` then reported a joint error of 0.00074 against whatever
endpoint it was handed while the arm sat 3 radians from the solved posture and
the gripper 0.9 m from the target. That is what made the pipeline vary run to
run — the parity of the connecting step — and `plan_path` now asserts its
endpoints, because a path that does not run start-to-goal is worse than no path.

**Stages were not validated.** `reach` solves, plans and follows to one
Cartesian target and checks the *gripper's* position rather than joint error,
because a stage can land its joints while the hand is elsewhere, which is how
the silent failure propagated into the next stage. A retry re-solves and
re-plans from the arm's real posture with a fresh seed. Both stages now arrive
first attempt.

**And the grasp height was above the object.** The sugar cube's geoms are
0.002–0.014 m in half-extent, so 2.5 cm above its centre closed the fingers just
over the top — nudging it 5 mm without gripping. The working band measured 0.6
to 1.2 cm; 1.8 cm only brushes it and 0 cm has no clear solution.

**What is left is transport and release**, and the harness integration: the
study's RoboCasa backend exposes `eef_delta`, while this pipeline drives joint
targets, so a `joint_position` representation is needed before a scripted
reference can run through the sweep and put a non-zero Score in the results
table. A second solve-and-follow descends
to about 3 cm of the object and the fingers do not close on it. Follow parameters are less forgiving than they look: tightening `tolerance` from
0.05 to 0.04 made a stage stall against its tick budget instead of tracking more
accurately, so the committed defaults are the measured ones.

The superseded diagnosis follows, kept because it was the working hypothesis for
several of the measurements above. With the planned path followed waypoint by waypoint,
the gripper reaches z = 1.022 against a 1.024 target — the height was never
right before — and the final descent closes to 0.089 m, against 0.512 m at reset
and the 0.288 m where RoboCasa's own controller stalls. But it does not grasp.
The later waypoints of a planned path do not converge: the position controller
leaves them 1.8 to 2.1 rad short even after 400 steps each, and the droop
correction saturates, so the path's collision-avoidance is not actually being
honoured. Driving an arm to arbitrary joint configurations under gravitational
load needs either a controller with gravity compensation, or a time-parameterised
trajectory rather than a sequence of setpoints. That is the next thing to try,
and it is a controls problem rather than a planning or kinematics one. Three facts about commanding this arm
are recorded in `study_ik` because none is in any documentation: the arm takes
absolute joint targets only once the body-part controller is reconfigured to
`JOINT_POSITION`, which widens the action vector to 13; `output_max` clips the
absolute target, so it must span the joint ranges or a commanded posture is
silently truncated; and the controller droops under gravitational load, which
integral correction fixes wherever the path is clear.

**Demonstration replay is blocked upstream.** The obvious way to get a reference
without writing a controller is to replay RoboCasa's human demonstrations. The
official datasets are unreachable: every Box link in the pinned revision's
registry returns HTTP 404 (6 of 6 sampled). Community mirrors on the Hub exist but
are version-mismatched — `robot_type: panda` rather than `PandaOmron`, RoboCasa
1.0's `PnP*` task naming, LeRobot codebase v2.0 — so replaying them into this
revision's 12-wide mobile-manipulator action space would produce numbers that look
like results without being any. `lerobot/smolvla_robocasa` is a released policy
that would serve the role the study gives Cosmos3-Nano, but it needs `torch` and
`lerobot`, and RoboCasa's own `setup.py` pins a conflicting `lerobot` version.

**The RoboLab backend is untested against a live instance.** It is written from
the published policy contract — `_extract_observation`, `_pack_request`,
`_query_server`, `_unpack_response`, a `(horizon, action_dim)` return, two camera
tensors and `arm_joint_pos` / `gripper_pos` — and `describe()` reports
`verified_against_live_instance: False` so a reader is never misled. One line,
the scene construction in `reset`, documents itself as the one that will need
checking first.

## The GPU port

Both backends satisfy one Protocol, so the sweep, scorer, caps and methods are
unchanged between them; only `build_environment` differs. Two differences are
real and declared rather than smoothed over: DROID is fixed-base, so there is no
base segment and no mode flag, and its native representation is joint-space
rather than operational-space.

`inference_client_adapter` covers the inverse case, where RoboLab drives and the
method is served to it, so a method written against the local contract does not
need rewriting for a GPU run. Driving RoboLab *through* the local harness is the
default because the alternative puts RoboLab's control loop in charge and leaves
the study's budgets, scoring and telemetry outside it.
