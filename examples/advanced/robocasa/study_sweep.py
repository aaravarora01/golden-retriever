"""Sweep task x episode x method, and report what the study reports.

One JSON object to stdout and a meaningful exit code, following this folder's
other entry points. The sweep layers on `MethodHarness` rather than replacing
it: the harness already dispatches one control step per chunk row and emits a
`chunk_dispatched` event per `predict`, which is where the two step counts come
from.

Three counts are kept apart because they are different quantities, and the study
reports the third. **Control steps** are rows actually dispatched to the
simulator. **Model calls** are `predict` invocations, one per chunk — the metric
the study names, and the one a longer horizon reduces. **Episodes** are trials.
Nothing here sums them.

Latency is measured but kept out of any canonical record, for the reason the
sibling drawer harness documents: a measured duration differs on every run, so a
record containing one can never be compared byte for byte. It is reported per
episode as a summary and belongs in a sidecar if it is ever joined to evidence.

The protocol follows the study's: tasks x 5 episodes at fixed seeds 0-4 by
default, since that is what the published numbers use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter

#: The study's episode protocol: five episodes per task at fixed seeds.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class MethodSpec:
    """A named method plus how to build a fresh one for each trial.

    A factory rather than an instance: a method may carry per-episode state —
    a scripted phase, a random generator, a call ledger — and reusing one across
    trials would leak an episode into the next.
    """

    method_id: str
    build: object
    representation: str = "eef_delta"
    horizon: int = 1
    label: str = ""

    def __post_init__(self) -> None:
        if not self.method_id or self.method_id != self.method_id.strip():
            raise ValueError("method_id must be a non-empty, unpadded string")
        if not callable(self.build):
            raise TypeError("build must be callable and return a fresh method")


@dataclass
class EpisodeResult:
    """One trial's measurements, in the study's vocabulary."""

    task: str
    method_id: str
    seed: int
    success: bool
    status: str
    score: int
    rungs: dict
    control_steps: int
    model_calls: int
    horizon: int
    representation: str
    latency_s: float
    clip: dict
    backend: dict
    error: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class Sweep:
    """Run every (task, seed, method) cell and summarise per method."""

    build_environment: object
    tasks: tuple = ("PickPlaceCounterToCabinet",)
    seeds: tuple = DEFAULT_SEEDS
    max_steps: int = 400
    results: list = field(default_factory=list)

    def run(self, methods) -> dict:
        from .method_harness import MethodHarness, SafetyEnvelope, TrialRequest

        specs = list(methods)
        if not specs:
            raise ValueError("a sweep needs at least one method")
        if len({spec.method_id for spec in specs}) != len(specs):
            raise ValueError("methods contain a duplicate method_id")
        harness = MethodHarness(safety=SafetyEnvelope())
        for task in self.tasks:
            for seed in self.seeds:
                for spec in specs:
                    self.results.append(self._cell(harness, task, seed, spec, TrialRequest))
        return self.summarise()

    def _cell(self, harness, task, seed, spec, TrialRequest):
        environment = self.build_environment(task=task, representation=spec.representation)
        method = spec.build(environment)
        started = perf_counter()
        report = harness.run(
            TrialRequest(method_id=spec.method_id, task=task, episode=seed, seed=seed,
                         max_steps=self.max_steps),
            environment=environment, method=method, close_environment=False)
        latency = perf_counter() - started
        # One `chunk_dispatched` event per `predict`, which is the study's
        # model-call count and is read from the harness rather than from the
        # method, so a method cannot misreport it.
        calls = sum(1 for event in report.events if event.kind == "chunk_dispatched")
        progress = getattr(environment, "progress", None)
        codec = getattr(environment, "codec", None)
        try:
            environment.close()
        except Exception:  # noqa: BLE001 - cleanup failure must not lose the result
            pass
        return EpisodeResult(
            task=task, method_id=spec.method_id, seed=seed,
            success=bool(report.success), status=report.status,
            score=int(progress.score) if progress else 0,
            rungs=dict(progress.rungs) if progress else {},
            control_steps=int(report.steps), model_calls=calls, horizon=spec.horizon,
            representation=spec.representation, latency_s=round(latency, 4),
            clip=codec.ledger.to_dict() if codec else {},
            backend=environment.describe(), error=report.error)

    def summarise(self) -> dict:
        per_method: dict = {}
        for result in self.results:
            bucket = per_method.setdefault(result.method_id, [])
            bucket.append(result)
        summary = {}
        for method_id, rows in sorted(per_method.items()):
            count = len(rows)
            summary[method_id] = {
                "episodes": count,
                "successes": sum(row.success for row in rows),
                # The study's two headline numbers.
                "success_rate": round(sum(row.success for row in rows) / count, 4),
                "mean_score": round(sum(row.score for row in rows) / count, 2),
                # Three counts, reported separately and never summed.
                "control_steps": sum(row.control_steps for row in rows),
                "model_calls": sum(row.model_calls for row in rows),
                "mean_control_steps": round(sum(row.control_steps for row in rows) / count, 2),
                "mean_model_calls": round(sum(row.model_calls for row in rows) / count, 2),
                "mean_latency_s": round(sum(row.latency_s for row in rows) / count, 3),
                "rungs_reached": {
                    rung: sum(bool(row.rungs.get(rung)) for row in rows)
                    for rung in (rows[0].rungs or {})},
                "clipped_fraction": _mean_clip(rows),
                "errors": sum(1 for row in rows if row.error),
            }
        return {"schema_version": 1,
                "protocol": {"tasks": list(self.tasks), "seeds": list(self.seeds),
                             "max_steps": self.max_steps,
                             "episodes": len(self.tasks) * len(self.seeds)},
                "episodes": [row.to_dict() for row in self.results],
                "summary": summary,
                # A sweep of baselines is expected to score zero, so "passed"
                # would be a meaningless verdict. Errors are the thing a run
                # can fail on.
                "errors": sum(1 for row in self.results if row.error)}


def _mean_clip(rows):
    fractions = [row.clip.get("clipped_fraction") for row in rows
                 if row.clip.get("clipped_fraction") is not None]
    return round(sum(fractions) / len(fractions), 4) if fractions else None


def render(summary: dict, *, summary_only: bool = False) -> str:
    payload = ({key: value for key, value in summary.items() if key != "episodes"}
               if summary_only else summary)
    return json.dumps(payload, sort_keys=True, allow_nan=False)
