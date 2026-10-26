"""Check a policy against its environment before a long run.

Every expensive mistake in this study was an interface disagreement that
produced valid, parseable, useless behaviour:

* a prompt stating `[-1, 1]` against a codec whose cap was 0.02 m — 75 to 93%
  of rows clipped, zero successes on eighteen tasks, 75 minutes
* a prompt whose example was 7 wide beside a stated width of 10 — the model
  answered 7, which is correct behaviour when an example and its description
  disagree
* a policy handed an image hash instead of pixels, which looks exactly like a
  policy that cannot see because it is one
* a state in the world frame where the checkpoint was trained on the base frame
* a normaliser that loaded empty, which mean-std normalisation turns into
  plausible nonsense rather than an error

None of these raise. All of them are cheap to detect in a handful of steps, and
that is what this does: run a few control steps and assert the things that must
be true of the pair, loudly, before committing hours.

It is a *check*, not a measurement. It says nothing about whether a policy is
any good — only that the policy and the environment agree about what an action
is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Fraction of clipped rows above which a run is reporting the clipper's
#: behaviour rather than the policy's. The measured failure sat at 0.75 to 0.93;
#: a policy told its real limits sat at 0.
SATURATION_LIMIT = 0.5
#: Steps to probe. Enough to exercise a chunk boundary and the clip ledger,
#: few enough to cost cents against a paid model.
PROBE_STEPS = 12


@dataclass
class Finding:
    """One check, and why it matters if it fails."""

    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class Preflight:
    """Run the checks and collect findings."""

    findings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(finding.passed for finding in self.findings)

    def note(self, name: str, passed: bool, detail: str = "") -> None:
        self.findings.append(Finding(name=name, passed=bool(passed), detail=detail))

    def to_dict(self) -> dict:
        return {"ok": self.ok,
                "failed": [f.name for f in self.findings if not f.passed],
                "findings": [f.to_dict() for f in self.findings]}


def check_action_width(preflight: Preflight, *, representation: str, dof: int) -> None:
    """The policy's declared width must be the representation's width."""
    from .study_representations import REPRESENTATION_DOF

    expected = REPRESENTATION_DOF.get(representation)
    preflight.note(
        "action width agrees with the representation",
        expected == dof,
        f"{representation} is {expected} wide, the policy declares {dof}")


def check_prompt_contract(preflight: Preflight, *, config, dof: int) -> None:
    """The prompt must state this width, and this codec's real limits."""
    import json
    import re

    from .study_representations import MAX_ROTATION_PER_STEP, MAX_TRANSLATION_PER_STEP

    try:
        text = config.instructions(dof=dof)
    except Exception as error:  # noqa: BLE001 - an unusable prompt is the finding
        preflight.note("prompt renders", False, f"{type(error).__name__}: {error}")
        return
    preflight.note("prompt renders", True)

    found = re.search(r"\[\[.*?\]\]", text, re.S)
    if not found:
        preflight.note("prompt shows an example", False, "no example found")
    else:
        widths = {len(row) for row in json.loads(found.group(0))}
        preflight.note(
            "prompt example matches the stated width",
            widths == {dof},
            f"example rows are {sorted(widths)} wide, stated {dof}")

    states_scale = (str(MAX_TRANSLATION_PER_STEP) in text
                    and str(MAX_ROTATION_PER_STEP) in text)
    preflight.note(
        "prompt states the codec's per-step limits",
        states_scale,
        f"expected {MAX_TRANSLATION_PER_STEP} m and {MAX_ROTATION_PER_STEP} rad "
        "to appear; a prompt that gives only the envelope bound teaches a "
        "magnitude that is clipped away")


def check_observation(preflight: Preflight, *, observation) -> None:
    """The policy must be handed pixels and a finite state, not placeholders."""
    image = getattr(observation, "image", None)
    preflight.note(
        "observation carries an image",
        image is not None and np.asarray(image).ndim == 3,
        f"image is {None if image is None else np.asarray(image).shape}")
    if image is not None:
        frame = np.asarray(image)
        preflight.note(
            "image is not blank",
            float(frame.std()) > 1.0,
            f"pixel std {float(frame.std()):.2f}; a constant frame means the "
            "renderer produced nothing")
    state = np.asarray(getattr(observation, "state", []))
    preflight.note(
        "state is finite and non-empty",
        state.size > 0 and bool(np.isfinite(state).all()),
        f"state shape {state.shape}")
    prompt = getattr(observation, "prompt", "")
    preflight.note("observation carries a task instruction",
                   bool(isinstance(prompt, str) and prompt.strip()), repr(prompt)[:80])


def check_saturation(preflight: Preflight, *, clip: dict) -> None:
    """Most actions reaching their cap means the clipper is the policy."""
    fraction = clip.get("clipped_fraction")
    if fraction is None:
        preflight.note("clip ledger reports a saturation fraction", False,
                       "no clipped_fraction; saturation cannot be seen")
        return
    preflight.note(
        "actions are not mostly clipped",
        float(fraction) <= SATURATION_LIMIT,
        f"clipped_fraction {float(fraction):.3f} against a limit of "
        f"{SATURATION_LIMIT}; excess translation "
        f"{clip.get('max_translation_excess_m')} m")


def check_calls(preflight: Preflight, *, method, steps: int) -> None:
    """A policy must have been asked, and must mostly have answered."""
    described = {}
    describe = getattr(method, "describe", None)
    if callable(describe):
        value = describe()
        if isinstance(value, dict):
            described = value
    # Where the count lives differs by method: a Flow adapter counts on itself,
    # a model method counts on its client, and a ledger counts either way. Look
    # in all three, because a check that reads zero from the wrong attribute
    # reports a working policy as one that was never called.
    def first_count(*candidates):
        for value in candidates:
            if value:
                return int(value)
        return 0

    client = getattr(method, "client", None)
    ledger = getattr(method, "ledger", None)
    calls = first_count(described.get("calls"), getattr(method, "calls", 0),
                        getattr(client, "calls", 0),
                        len(getattr(ledger, "canonical", ()) or ()))
    refusals = first_count(described.get("refusals"), getattr(method, "refusals", 0),
                           getattr(client, "refusals", 0))
    preflight.note("the policy was called", calls > 0, f"{calls} calls in {steps} steps")
    if calls:
        preflight.note(
            "most calls were answered",
            refusals / calls <= 0.5,
            f"{refusals} of {calls} refused; a refused call becomes a held "
            "no-motion action and is indistinguishable from a bad policy")


def run(*, environment, method, task: str, representation: str, dof: int,
        config=None, steps: int = PROBE_STEPS) -> Preflight:
    """Probe the pair for `steps` control steps and report every finding."""
    from .method_harness import TrialRequest

    preflight = Preflight()
    check_action_width(preflight, representation=representation, dof=dof)
    if config is not None:
        check_prompt_contract(preflight, config=config, dof=dof)

    observation = environment.reset(
        TrialRequest(method_id="preflight", task=task, seed=0, max_steps=steps))
    check_observation(preflight, observation=observation)

    dispatched = 0
    try:
        while dispatched < steps:
            chunk = method.predict(observation)
            rows = np.asarray(chunk.actions, dtype=np.float32)
            if dispatched == 0:
                preflight.note(
                    "chunk width matches the representation",
                    rows.ndim == 2 and rows.shape[1] == dof,
                    f"chunk is {rows.shape}, expected (*, {dof})")
            for row in rows:
                if dispatched >= steps:
                    break
                observation = environment.step(row).observation
                dispatched += 1
    except Exception as error:  # noqa: BLE001 - a crash in the probe is the finding
        preflight.note("the pair runs", False, f"{type(error).__name__}: {error}"[:220])
    else:
        preflight.note("the pair runs", True, f"{dispatched} control steps")

    codec = getattr(environment, "codec", None)
    if codec is not None:
        check_saturation(preflight, clip=codec.ledger.to_dict())
    check_calls(preflight, method=method, steps=dispatched)
    return preflight
