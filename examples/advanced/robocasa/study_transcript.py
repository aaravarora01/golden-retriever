"""Replay recorded policy calls keyed by prompt digest; no network, no keys.

This is the model seam's offline half. A paid sweep is recorded once and replays
for free thereafter, and the default test path needs no API key — which is what
lets the whole model path, including its failure modes, run in CI.

**The digest is over the prompt, and the prompt includes the image.** An action
chunk depends on what the policy saw, so a transcript keyed on the instruction
alone would return the same answer at every step of an episode. Images are
hashed rather than embedded: a 128x128x3 frame is 49 KB and a transcript would be
unusable if it carried them, while a digest over the raw bytes is exact. The
consequence worth stating is that the key is only as stable as the renderer — a
different image size, camera set or RoboCasa build produces different frames and
therefore a different digest, so a transcript belongs to the configuration that
recorded it.

**There is no fallback on a miss.** A missing entry raises, the method's caller
records a refusal, and the episode ends. Answering from a scripted policy
instead would mix scripted behaviour into a replayed arm's results, which is the
most effective way to make a benchmark report a number that is not about the
policy it names. An incomplete transcript must look incomplete.

Parsing is forgiving of packaging and strict about content: the JSON array
inside a code fence or prose still parses, and is then held to the same shape
and bounds any method's chunk faces. A malformed response is therefore a
measured refusal rather than a crash, which is what makes an invalid-output rate
reportable.

Ported from `retriever-bklva`'s drawer harness. What changed is the payload: the
drawer policy answered with one action dict, a RoboCasa policy answers with a
`(horizon, dof)` array of floats.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np

TRANSCRIPT_ENV = "ROBOCASA_STUDY_TRANSCRIPT"
TRANSCRIPT_SCHEMA = 1
_MAX_TRANSCRIPT_BYTES = 8_000_000
_MAX_RESPONSE_BYTES = 200_000
#: Matches `SafetyEnvelope` so a transcript cannot smuggle a chunk the harness
#: would reject anyway.
MAX_HORIZON = 64
MAX_DOF = 32


class InvalidPolicyOutput(ValueError):
    """A policy produced something the harness must not dispatch."""

    def __init__(self, reason, output=None):
        super().__init__(reason)
        self.reason = reason
        self.output = output


def assemble_prompt(observation, *, config=None) -> str:
    """Render the exact bytes that get hashed for one call.

    Canonical JSON over the instruction, the proprioceptive state and a digest
    of the image — not the image itself. The configuration, when given, is part
    of the prompt rather than metadata beside it, so two prompt variants or two
    models cannot collide on one transcript entry.
    """
    image = np.ascontiguousarray(np.asarray(observation.image, dtype=np.uint8))
    payload = {
        "prompt": str(observation.prompt),
        "state": [round(float(value), 6) for value in np.asarray(observation.state).ravel()],
        "image_sha256": sha256(image.tobytes()).hexdigest(),
        "image_shape": list(image.shape),
    }
    if config is not None:
        payload["config"] = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))


def prompt_digest(prompt: str) -> str:
    return sha256(prompt.encode("utf-8")).hexdigest()


def parse_chunk_response(text, *, dof: int):
    """Extract and validate one action chunk from a recorded response."""
    if type(text) is not str or not text.strip():
        raise InvalidPolicyOutput("transcript response is empty", text)
    if len(text.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        raise InvalidPolicyOutput("transcript response exceeds the size bound", None)
    candidates = [text.strip()]
    start, end = text.find("["), text.rfind("]")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, RecursionError):
            continue
        return validate_chunk(parsed, dof=dof)
    raise InvalidPolicyOutput("transcript response contains no JSON array", text[:200])


def validate_chunk(parsed, *, dof: int):
    """Hold a parsed response to the shape and bounds a chunk must satisfy."""
    if not isinstance(parsed, list) or not parsed:
        raise InvalidPolicyOutput("a chunk must be a non-empty list of rows", parsed)
    if not all(isinstance(row, list) for row in parsed):
        raise InvalidPolicyOutput("every chunk row must be a list", parsed)
    widths = {len(row) for row in parsed}
    if widths != {dof}:
        raise InvalidPolicyOutput(
            f"chunk rows must each have {dof} values, got widths {sorted(widths)}", parsed)
    if len(parsed) > MAX_HORIZON:
        raise InvalidPolicyOutput(
            f"chunk horizon {len(parsed)} exceeds the bound of {MAX_HORIZON}", None)
    if dof > MAX_DOF:
        raise InvalidPolicyOutput(f"chunk dof {dof} exceeds the bound of {MAX_DOF}", None)
    try:
        actions = np.asarray(parsed, dtype=np.float64)
    except (TypeError, ValueError) as error:
        # A row of the right width can still hold a string or a null. Letting
        # numpy's own error escape would crash the harness on malformed model
        # output, which is the one thing this function exists to prevent.
        raise InvalidPolicyOutput(f"chunk holds a non-numeric value: {error}", None) from error
    if not np.isfinite(actions).all():
        raise InvalidPolicyOutput("chunk contains a non-finite value", None)
    if np.abs(actions).max(initial=0.0) > 1.0:
        raise InvalidPolicyOutput("chunk exceeds the normalised action bound of 1.0", None)
    return actions.astype(np.float32)


class TranscriptStore:
    """A digest-keyed set of recorded calls, loaded once at construction."""

    def __init__(self, entries, *, source_sha256="", source_name=""):
        if not isinstance(entries, dict):
            raise TypeError("transcript entries must be a mapping")
        for digest, entry in entries.items():
            if type(digest) is not str or len(digest) != 64:
                raise ValueError(f"transcript key {digest!r} is not a sha256 digest")
            if not isinstance(entry, dict) or "response" not in entry:
                raise ValueError(f"transcript entry {digest[:12]} has no response")
        self.entries = entries
        self.source_sha256 = source_sha256
        self.source_name = source_name

    def __len__(self):
        return len(self.entries)

    def get(self, digest):
        return self.entries.get(digest)

    @classmethod
    def from_path(cls, path) -> TranscriptStore:
        """Read a transcript file; never called at import time."""
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"no transcript at {file}")
        raw = file.read_bytes()
        if len(raw) > _MAX_TRANSCRIPT_BYTES:
            raise ValueError("transcript exceeds the size bound")
        document = json.loads(raw)
        if not isinstance(document, dict) or document.get("schema_version") != TRANSCRIPT_SCHEMA:
            raise ValueError("unsupported transcript schema")
        return cls(document.get("calls") or {}, source_sha256=sha256(raw).hexdigest(),
                   source_name=file.name)

    def to_document(self) -> dict:
        return {"schema_version": TRANSCRIPT_SCHEMA, "calls": self.entries}

    def write(self, path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_document(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        return destination


@dataclass
class CallLedger:
    """The two halves of a call record, kept apart on purpose.

    `canonical` holds digests and indices, which are the same on every run.
    `telemetry` holds what was measured, which is not. A caller that mixes them
    produces a record that can never be compared across runs.
    """

    canonical: list = field(default_factory=list)
    telemetry: list = field(default_factory=list)

    def reset(self) -> None:
        self.canonical = []
        self.telemetry = []

    def note(self, *, digest, latency, parse_ok, hit=None, horizon=None, entry=None):
        index = len(self.canonical)
        entry = entry or {}
        self.canonical.append({"call_index": index, "prompt_sha256": digest,
                               "horizon": horizon})
        self.telemetry.append({"call_index": index, "prompt_sha256": digest,
                               "latency_s": round(latency, 6), "parse_ok": parse_ok,
                               "transcript_hit": hit, "horizon": horizon,
                               "model_id": entry.get("model_id"),
                               "prompt_tokens": entry.get("prompt_tokens"),
                               "completion_tokens": entry.get("completion_tokens"),
                               "usd": entry.get("usd"),
                               "recorded_latency_s": entry.get("latency_s")})
        return index


@dataclass
class TranscriptRecorder:
    """Wrap a method and record the call it would have made.

    Swap the wrapped method for a model client and this produces a real
    transcript; nothing else changes. Only what the recorder actually knows is
    stored — a response and a `model_id`. Inventing token or cost figures would
    fabricate exactly the numbers a study measures, and storing a measured
    latency would make the file differ on every recording.
    """

    method: object
    representation: str = "eef_delta"
    model_id: str = "oracle:scripted"
    report_cost: object = None
    store: TranscriptStore = field(default_factory=lambda: TranscriptStore({}))
    ledger: CallLedger = field(default_factory=CallLedger)

    def __post_init__(self) -> None:
        if not callable(getattr(self.method, "predict", None)):
            raise TypeError("a recorder needs a method with predict(observation)")

    def predict(self, observation):
        prompt = assemble_prompt(observation)
        digest = prompt_digest(prompt)
        started = time.perf_counter()
        chunk = self.method.predict(observation)
        latency = time.perf_counter() - started
        rows = np.asarray(chunk.actions, dtype=np.float32).round(6).tolist()
        entry = {"response": json.dumps(rows), "model_id": self.model_id}
        if self.report_cost is not None:
            entry.update(self.report_cost())
        self.store.entries[digest] = entry
        self.ledger.note(digest=digest, latency=latency, parse_ok=True, hit=None,
                         horizon=int(chunk.horizon), entry=entry)
        return chunk


@dataclass
class TranscriptMethod:
    """Replay a recorded transcript as a method.

    A miss raises `InvalidPolicyOutput`; see the module docstring for why there
    is no fallback.
    """

    store: TranscriptStore
    dof: int = 7
    source: str = "scripted"
    ledger: CallLedger = field(default_factory=CallLedger)

    def __post_init__(self) -> None:
        if not isinstance(self.store, TranscriptStore):
            raise TypeError("a transcript method requires a TranscriptStore")

    def predict(self, observation):
        from examples.advanced.openpi_policy.common import ActionChunk

        prompt = assemble_prompt(observation)
        digest = prompt_digest(prompt)
        started = time.perf_counter()
        entry = self.store.get(digest)
        actions, failure = None, None
        if entry is not None:
            try:
                actions = parse_chunk_response(entry.get("response"), dof=self.dof)
            except InvalidPolicyOutput as error:
                failure = error
        latency = time.perf_counter() - started
        # Recorded before raising, so a miss or a parse failure is measured
        # rather than lost: that rate is the point of the exercise.
        self.ledger.note(digest=digest, latency=latency,
                         parse_ok=failure is None and entry is not None,
                         hit=entry is not None,
                         horizon=None if actions is None else int(actions.shape[0]),
                         entry=entry)
        if entry is None:
            raise InvalidPolicyOutput(
                f"no recorded call for prompt {digest[:12]} ({len(self.store)} in transcript)")
        if failure is not None:
            raise failure
        return ActionChunk(actions=actions, horizon=int(actions.shape[0]),
                           dof=int(actions.shape[1]), source=self.source)


class EnvTranscriptMethod(TranscriptMethod):
    """Referenceable entry point: replay the transcript the environment names.

    A class rather than a factory function, so a loader that constructs types
    does not hand the factory itself to the harness as the method. Nothing is
    read at import time; an unset or missing path fails loudly rather than
    degrading to a different policy.
    """

    def __init__(self, dof: int = 7):
        path = os.environ.get(TRANSCRIPT_ENV)
        if not path:
            raise ValueError(f"set {TRANSCRIPT_ENV} to a recorded transcript path")
        super().__init__(store=TranscriptStore.from_path(path), dof=dof)
