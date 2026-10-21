"""Entry point for the RoboCasa pick-and-place policy study.

Prints one sorted JSON object and returns a meaningful exit code, following the
other runnable demos in this folder.

    pixi run -e robocasa study-pnp-baselines

The default sweep is the two baselines the harness has to be read against, over
the study's five fixed seeds. It is expected to report a zero success rate; the
numbers that matter from it are the control-step and model-call counts, the
partial-score floor, and the clipped-action fraction.
"""

from __future__ import annotations

import argparse

from .study_methods import RandomMethod, StillMethod
from .study_reference import ScriptedPickMethod
from .study_sweep import MethodSpec, Sweep, render
from .study_tasks import (
    DEFAULT_SEEDS, ablation_subset, get_task, list_tasks, vla_comparable)


def baseline_specs(horizon: int, representation: str) -> tuple:
    return (
        MethodSpec(method_id="still", horizon=horizon, representation=representation,
                   label="no motion; the floor every method is read against",
                   build=lambda env, h=horizon, r=representation: StillMethod(
                       horizon=h, representation=r)),
        MethodSpec(method_id="random", horizon=horizon, representation=representation,
                   label="bounded seeded random motion",
                   build=lambda env, h=horizon, r=representation: RandomMethod(
                       horizon=h, representation=r, seed=0)),
    )


def reference_specs(horizon: int) -> tuple:
    """The scripted reference, which is privileged and drives joint deltas.

    It reads the object's pose from the simulator, so it is a ceiling to read
    policies against rather than an entry in the comparison. Its representation
    is fixed: the pipeline behind it commands joint targets, and asking for an
    end-effector representation would silently reinterpret its rows.
    """
    return (
        MethodSpec(method_id="scripted-pick", horizon=horizon, representation="joint_delta",
                   label="privileged reach, grasp and lift; the ceiling, not a policy",
                   build=lambda env, h=horizon: ScriptedPickMethod(environment=env, horizon=h)),
    )


def vla_specs(horizon: int) -> tuple:
    """A released RoboCasa VLA checkpoint, in its own representation.

    Its observation and action spaces are fixed by the checkpoint — three views
    at 256 px, a 16-wide state, a 12-wide action in its own ordering — so none
    of them follow `--representation`.
    """
    from .study_vla import SmolVlaMethod

    return (
        MethodSpec(method_id="smolvla", horizon=horizon, representation="lerobot_12",
                   label="lerobot/smolvla_robocasa; a trained policy, not privileged",
                   build=lambda env, h=horizon: SmolVlaMethod(horizon=h)),
    )


def vla_observation():
    from .study_robocasa_env import ObservationConfig
    from .study_vla import LEROBOT_IMAGE_SIZE, LEROBOT_VIEWS

    return ObservationConfig(views=LEROBOT_VIEWS, history=0,
                             image_size=LEROBOT_IMAGE_SIZE)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="study-pnp")
    parser.add_argument("--task", action="append", dest="tasks", default=None,
                        help="Committed task name; repeatable. Defaults to a reach-spread subset.")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Run all 18 committed tasks instead of the subset.")
    parser.add_argument("--subset", type=int, default=4,
                        help="Size of the reach-spread subset when no task is named.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Episode seeds. The study uses 0-4.")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Actions proposed per model call.")
    parser.add_argument("--representation", default="eef_delta",
                        help="Action representation; eef_delta_base commands the mobile base.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--specification", choices=("default", "vague"), default="default",
                        help="Task-specification axis: RoboCasa's goal, or the derived "
                             "instruction that withholds the object's identity.")
    parser.add_argument("--vla-tasks", action="store_true",
                        help="Run only the tasks the released VLA checkpoint also knows.")
    parser.add_argument("--methods", choices=("baselines", "reference", "vla", "both"),
                        default="baselines",
                        help="Which arms to run. The reference is privileged and always "
                             "uses joint_delta, whatever --representation says.")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)

    from .study_robocasa_env import RoboCasaPnPEnvironment

    def build(task, representation):
        if representation == "lerobot_12":
            # The checkpoint fixes its own observation and state format.
            return RoboCasaPnPEnvironment(
                task=task, representation=representation,
                observation=vla_observation(), state_mode="lerobot",
                specification_style=args.specification)
        return RoboCasaPnPEnvironment(
            task=task, representation=representation,
            command_base=representation == "eef_delta_base",
            specification_style=args.specification)

    try:
        if args.tasks:
            tasks = tuple(get_task(name).task for name in args.tasks)
        elif args.vla_tasks:
            tasks = vla_comparable()
        elif args.all_tasks:
            tasks = list_tasks()
        else:
            tasks = ablation_subset(args.subset)
        specs = ()
        if args.methods in ("baselines", "both"):
            specs += baseline_specs(args.horizon, args.representation)
        if args.methods in ("reference", "both"):
            specs += reference_specs(args.horizon)
        if args.methods == "vla":
            specs += vla_specs(args.horizon)
        # A model's output is admitted only when a model arm is actually in
        # the sweep, so the default stays scripted-only.
        sources = ("scripted", "model") if args.methods == "vla" else ("scripted",)
        sweep = Sweep(build_environment=build, tasks=tasks,
                      seeds=tuple(args.seeds), max_steps=args.max_steps,
                      allowed_sources=sources)
        summary = sweep.run(specs)
    except (ValueError, TypeError, OSError) as error:
        print(render({"error": f"{type(error).__name__}: {error}"}))
        return 1
    print(render(summary, summary_only=args.summary_only))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
