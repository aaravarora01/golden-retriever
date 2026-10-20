"""The model-as-policy boundary, and the study's three policy forms.

This is the seam a frontier model plugs into. It is complete and tested offline;
what it does not do is call anything, because a live call needs an API key and
spends money. The shape follows `fast_downward.py` in the sibling drawer
harness, which is the established pattern here for calling something the repo
does not control: a serializable config with explicit bounds, a status enum
rather than exceptions for expected outcomes, and the response captured whole so
a failure becomes evidence instead of a traceback.

**No provider is implemented.** `ModelClient` is the protocol an adapter
satisfies, `load_model_client` resolves one by reference, and
`OfflineModelClient` answers from a wrapped method so the entire path — prompt
assembly, the call boundary, chunk parsing, refusal, the cost sidecar — runs in
CI with no key and no network.

The three policy forms are expressed as the study defines them, and the
differences here are real rather than cosmetic:

* **Direct** — stateless. Each call sees the current observation only, and the
  chunk length is the study's `3-15` or `16-32` horizon band.
* **Agentic** — carries its own history across calls: the last few
  observations' digests and the chunks it proposed. That history is what the
  study's context-history axis ablates.
* **Orchestrated** — reviews a base method's fixed 15-step proposal and either
  accepts it or intervenes with a shorter segment, recording which. The study
  uses Cosmos3-Nano-Policy as the base; here it is whatever method is handed in,
  which is the scripted reference until a trained one is available.

One honest limit, stated so a results table cannot imply otherwise: the prompt
axes are **wired, not live**. The configuration enters the prompt that gets
hashed, so a different prompt variant or reasoning effort is provably a different
request and cannot collide in one transcript. Whether a *model's behaviour*
changes with them is unknowable without a model, and the offline client ignores
the configuration by design — so a sweep over variants against it returns
identical rows, which is a tautology rather than a null result.
"""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from examples.advanced.openpi_policy.common import ActionChunk

from .study_transcript import (
    CallLedger, InvalidPolicyOutput, assemble_prompt, parse_chunk_response, prompt_digest)

#: Prompt variants, committed so a recorded digest stays reproducible rather
#: than depending on wording assembled at call time.
PROMPT_VARIANTS = {
    "terse": "Output the next robot action chunk as JSON. No other text.",
    "explained": (
        "You control a 7-DOF Franka arm on a mobile base in a kitchen. The state "
        "vector is joint positions, gripper opening and end-effector position. "
        "Output the next chunk of actions as a JSON array of rows; each row is an "
        "end-effector delta and a gripper command, every value in [-1, 1]."
    ),
    "with_format": (
        "You control a 7-DOF Franka arm on a mobile base in a kitchen. The state "
        "vector is joint positions, gripper opening and end-effector position.\n"
        "Reply with exactly one JSON array of rows and no other text, for example "
        "[[0.1,0.0,-0.2,0,0,0,-1],[0.1,0,-0.1,0,0,0,-1]]. Each row has {dof} values "
        "in [-1, 1]: an end-effector delta then a gripper command where -1 is open "
        "and +1 is closed. Return between {low} and {high} rows."
    ),
}

REASONING_EFFORTS = ("none", "low", "medium", "high")
#: The study's two horizon bands.
HORIZON_BANDS = {"short": (3, 15), "long": (16, 32)}
#: The Orchestrated form reviews a fixed-length proposal, as the study does.
ORCHESTRATED_PROPOSAL_STEPS = 15


class ModelCallStatus(str, Enum):
    """Why a call ended. Expected outcomes are statuses, not exceptions."""

    OK = "ok"
    TIMEOUT = "timeout"
    OVERSIZE = "oversize"
    REFUSED = "refused"
    TRANSPORT_ERROR = "transport_error"


@dataclass(frozen=True)
class ModelConfig:
    """Serializable settings for one bounded model call.

    Every field enters the prompt that is hashed, so two configurations cannot
    share a transcript entry and a recorded run names the configuration that
    produced it.
    """

    model_id: str
    prompt_variant: str = "with_format"
    reasoning_effort: str = "none"
    horizon_band: str = "short"
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    max_response_bytes: int = 200_000
    max_output_tokens: int = 8_192

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(f"unknown prompt variant {self.prompt_variant!r}; "
                             f"available: {', '.join(sorted(PROMPT_VARIANTS))}")
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")
        if self.horizon_band not in HORIZON_BANDS:
            raise ValueError(f"horizon_band must be one of {tuple(HORIZON_BANDS)}")
        if (type(self.temperature) is not float or not math.isfinite(self.temperature)
                or not 0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be a finite float from 0.0 through 2.0")
        if (type(self.timeout_seconds) is not float or not math.isfinite(self.timeout_seconds)
                or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be a positive finite float")
        for name in ("max_response_bytes", "max_output_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def horizon_range(self) -> tuple:
        return HORIZON_BANDS[self.horizon_band]

    def instructions(self, *, dof: int) -> str:
        low, high = self.horizon_range
        return PROMPT_VARIANTS[self.prompt_variant].format(dof=dof, low=low, high=high)

    def to_dict(self) -> dict:
        return {"model_id": self.model_id, "prompt_variant": self.prompt_variant,
                "reasoning_effort": self.reasoning_effort, "horizon_band": self.horizon_band,
                "temperature": self.temperature, "timeout_seconds": self.timeout_seconds,
                "max_response_bytes": self.max_response_bytes,
                "max_output_tokens": self.max_output_tokens}


@dataclass(frozen=True)
class ModelResponse:
    """One call's outcome, captured whole so a failure is evidence."""

    text: str = ""
    status: ModelCallStatus = ModelCallStatus.OK
    model_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    usd: float | None = None
    latency_s: float | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, ModelCallStatus):
            raise TypeError("status must be a ModelCallStatus")
        if type(self.text) is not str:
            raise TypeError("response text must be a string")
        for name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be None or a non-negative int")
        for name in ("usd", "latency_s"):
            value = getattr(self, name)
            if value is not None and (type(value) is not float
                                       or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be None or a non-negative finite float")

    def reported_cost(self) -> dict:
        return {"model_id": self.model_id, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "usd": self.usd,
                "latency_s": self.latency_s}


class ModelClient:
    """The protocol an adapter satisfies: one bounded call, cost reported.

    An adapter overrides `complete` to send `prompt` to a provider and return a
    `ModelResponse`. It must honour `config.timeout_seconds` and
    `config.max_response_bytes`, and express an expected failure — a timeout, an
    oversize or refused answer, a transport error — as a *status* rather than
    raising, so the failure becomes a recorded refusal instead of a harness
    crash. Reporting the provider's token counts and cost is what makes those
    columns mean anything; omitting them is honest and they are then reported as
    absent rather than as zero.

    Nothing may be opened at import time. Construct the provider client in
    `__init__`, which is where `load_model_client` calls it.
    """

    def complete(self, prompt: str, config: ModelConfig) -> ModelResponse:
        raise NotImplementedError("a model client must implement complete(prompt, config)")


@dataclass
class OfflineModelClient(ModelClient):
    """A deterministic stand-in that answers from a wrapped method.

    It exists so the whole model path runs with no network and no key. It
    deliberately ignores the configuration, so a sweep over prompt variants
    against it produces identical rows — a tautology, not a null result, and the
    reason the prompt axes are described as wired rather than live.
    """

    method: object = None
    model_id: str = "offline:scripted"
    answers: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def complete(self, prompt: str, config: ModelConfig) -> ModelResponse:
        self.calls.append(prompt)
        if self.answers:
            answer = self.answers.pop(0)
            if isinstance(answer, ModelResponse):
                return answer
            return ModelResponse(text=answer, model_id=self.model_id)
        if self.method is None:
            return ModelResponse(status=ModelCallStatus.REFUSED, model_id=self.model_id,
                                 detail="offline client has neither answers nor a method")
        # The observation is not reachable from the prompt, so the wrapped
        # method is called with None; a method that needs the observation should
        # be driven through `record_into` instead.
        chunk = self.method.predict(None)
        rows = np.asarray(chunk.actions, dtype=np.float32).round(6).tolist()
        import json

        return ModelResponse(text=json.dumps(rows), model_id=self.model_id)


def load_model_client(reference: str) -> ModelClient:
    """Construct a client from a serializable ``module:symbol`` reference.

    A client reachable only from live Python cannot be named in saved results,
    so a run using it is not reproducible from its record.
    """
    try:
        module_name, symbol_name = reference.split(":", 1)
    except ValueError as error:
        raise ValueError("model client reference must use 'module:symbol' syntax") from error
    target = getattr(importlib.import_module(module_name), symbol_name)
    client = target() if isinstance(target, type) else target
    if not callable(getattr(client, "complete", None)):
        raise TypeError(f"model client {reference!r} does not define complete(prompt, config)")
    return client


def assemble_model_prompt(observation, config: ModelConfig, *, dof: int,
                          history=()) -> str:
    """Render the exact text hashed and sent for one call.

    The instruction, the configuration and any carried history all enter the
    hashed payload, so the Direct and Agentic forms of one configuration are
    distinguishable requests rather than colliding on one transcript entry.
    """
    payload = {"instructions": config.instructions(dof=dof), "dof": dof}
    if history:
        payload["history"] = list(history)
    return assemble_prompt(observation, config={**config.to_dict(), **payload})


@dataclass
class _ModelMethod:
    """Shared call machinery for every form: one bounded call, then a chunk."""

    client: object
    config: ModelConfig
    dof: int = 7
    source: str = "scripted"
    record_into: object = None
    ledger: CallLedger = field(default_factory=CallLedger)

    def __post_init__(self) -> None:
        if not callable(getattr(self.client, "complete", None)):
            raise TypeError("a model method requires a client with complete(prompt, config)")
        if not isinstance(self.config, ModelConfig):
            raise TypeError("a model method requires a ModelConfig")

    def canonical_ledger(self) -> list:
        """Deterministic per-call record, safe to put beside results."""
        return [dict(entry, model_config=self.config.to_dict())
                for entry in self.ledger.canonical]

    def telemetry(self) -> list:
        return [dict(row) for row in self.ledger.telemetry]

    def _call(self, observation, *, history=()):
        prompt = assemble_model_prompt(observation, self.config, dof=self.dof, history=history)
        digest = prompt_digest(prompt)
        started = time.perf_counter()
        try:
            response = self.client.complete(prompt, self.config)
        except Exception as error:  # noqa: BLE001 - a transport failure is the call's outcome
            response = ModelResponse(status=ModelCallStatus.TRANSPORT_ERROR,
                                     detail=f"{type(error).__name__}: {error}")
        latency = time.perf_counter() - started
        actions, failure = None, None
        if response.status is not ModelCallStatus.OK:
            failure = InvalidPolicyOutput(
                f"model call ended {response.status.value}: {response.detail}"[:200])
        elif len(response.text.encode("utf-8")) > self.config.max_response_bytes:
            failure = InvalidPolicyOutput("model response exceeds its configured bound")
        else:
            try:
                actions = parse_chunk_response(response.text, dof=self.dof)
            except InvalidPolicyOutput as error:
                failure = error
        low, high = self.config.horizon_range
        if failure is None and not low <= actions.shape[0] <= high:
            failure = InvalidPolicyOutput(
                f"chunk of {actions.shape[0]} rows is outside the configured "
                f"{self.config.horizon_band} band of {low}-{high}")
            actions = None
        entry = response.reported_cost()
        if failure is None and self.record_into is not None:
            import json

            self.record_into.entries[digest] = {
                "response": json.dumps(np.asarray(actions).round(6).tolist()),
                **{key: value for key, value in entry.items() if value is not None}}
        self.ledger.note(digest=digest, latency=latency, parse_ok=failure is None,
                         hit=None, horizon=None if actions is None else int(actions.shape[0]),
                         entry=entry)
        if failure is not None:
            raise failure
        return ActionChunk(actions=actions, horizon=int(actions.shape[0]),
                           dof=int(actions.shape[1]), source=self.source)


@dataclass
class DirectModelMethod(_ModelMethod):
    """Stateless: each call sees the current observation only."""

    form: str = "direct"

    def predict(self, observation):
        return self._call(observation)


@dataclass
class AgenticModelMethod(_ModelMethod):
    """Carries its own history across calls, which is what the axis ablates.

    `memory` is the number of previous calls described in the prompt. At 0 it is
    informationally the Direct form and must behave identically, which is
    stated here and pinned by a test rather than reported as a difference.
    """

    form: str = "agentic"
    memory: int = 3
    _history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.memory) is not int or not 0 <= self.memory <= 16:
            raise ValueError("memory must be an int from 0 through 16")

    def reset(self) -> None:
        self._history = []
        self.ledger.reset()

    def predict(self, observation):
        window = self._history[-self.memory:] if self.memory else ()
        chunk = self._call(observation, history=tuple(window))
        self._history.append({"call": len(self._history),
                              "horizon": int(chunk.horizon)})
        return chunk


@dataclass
class OrchestratedModelMethod(_ModelMethod):
    """Review a base method's fixed-length proposal and possibly intervene.

    The study has the model review a 15-step proposal from a separate trained
    policy and either accept it or substitute a shorter segment. `base` is
    whatever method is supplied — the scripted reference, until a trained one is
    available here — and every review is recorded so the intervention rate is
    readable without re-running anything.
    """

    form: str = "orchestrated"
    base: object = None
    proposal_steps: int = ORCHESTRATED_PROPOSAL_STEPS
    reviews: list = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not callable(getattr(self.base, "predict", None)):
            raise TypeError("an orchestrated method requires a base with predict(observation)")
        if type(self.proposal_steps) is not int or not 1 <= self.proposal_steps <= 64:
            raise ValueError("proposal_steps must be an int from 1 through 64")

    def reset(self) -> None:
        self.reviews = []
        self.ledger.reset()

    def predict(self, observation):
        proposal = self.base.predict(observation)
        proposed = int(proposal.horizon)
        try:
            # The reviewer is asked for a replacement; accepting the proposal is
            # what happens when it cannot produce a usable one, so a model that
            # fails to answer does not fail the episode.
            reviewed = self._call(observation, history=({"proposed_steps": proposed},))
        except InvalidPolicyOutput as error:
            self.reviews.append({"call": len(self.reviews), "proposed_steps": proposed,
                                 "committed_steps": proposed, "intervened": False,
                                 "reason": str(error)[:120]})
            return proposal
        self.reviews.append({"call": len(self.reviews), "proposed_steps": proposed,
                             "committed_steps": int(reviewed.horizon),
                             "intervened": int(reviewed.horizon) != proposed, "reason": ""})
        return reviewed


def offline_forms(method_factory, *, dof: int = 7, model_id: str = "offline:scripted") -> dict:
    """The three forms against the offline client, for a keyless sweep.

    Every form answers from the same wrapped method, so a difference between
    them in a sweep would mean the harness is altering behaviour rather than the
    form being measured. A test pins that they agree.
    """
    config = ModelConfig(model_id=model_id, horizon_band="short")

    def client():
        return OfflineModelClient(method=method_factory(), model_id=model_id)

    return {
        "direct": DirectModelMethod(client=client(), config=config, dof=dof),
        "agentic": AgenticModelMethod(client=client(), config=config, dof=dof, memory=3),
        "orchestrated": OrchestratedModelMethod(
            client=client(), config=replace(config, horizon_band="short"), dof=dof,
            base=method_factory()),
    }
