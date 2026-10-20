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
from .study_sweep import DEFAULT_SEEDS, MethodSpec, Sweep, render

#: Tasks whose initial reach was measured; see STUDY.md for why none is
#: arm-reachable without commanding the base.
DEFAULT_TASKS = ("PickPlaceCounterToCabinet", "PickPlaceCabinetToCounter")


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="study-pnp")
    parser.add_argument("--task", action="append", dest="tasks", default=None,
                        help="RoboCasa task; repeatable. Defaults to two measured ones.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Episode seeds. The study uses 0-4.")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Actions proposed per model call.")
    parser.add_argument("--representation", default="eef_delta",
                        help="Action representation; eef_delta_base commands the mobile base.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)

    from .study_robocasa_env import RoboCasaPnPEnvironment

    def build(task, representation):
        return RoboCasaPnPEnvironment(
            task=task, representation=representation,
            command_base=representation == "eef_delta_base")

    sweep = Sweep(build_environment=build,
                  tasks=tuple(args.tasks or DEFAULT_TASKS),
                  seeds=tuple(args.seeds), max_steps=args.max_steps)
    try:
        summary = sweep.run(baseline_specs(args.horizon, args.representation))
    except (ValueError, TypeError, OSError) as error:
        print(render({"error": f"{type(error).__name__}: {error}"}))
        return 1
    print(render(summary, summary_only=args.summary_only))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
