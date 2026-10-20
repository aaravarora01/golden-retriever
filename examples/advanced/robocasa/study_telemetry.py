"""Per-call latency, token and cost records, kept beside the results.

The study reports inference latency per call, tokens and USD per episode, and
model calls per episode. The first three cannot live in a record that is meant
to be comparable across runs — a measured duration differs every time — so they
go in a sidecar and the canonical sweep output keeps only what is deterministic.

The join is verifiable rather than positional. Each sidecar row carries the
prompt digest of the call it describes, and a policy that keeps a ledger records
the same digest per call index, so `verify_join` can confirm the two describe
the same episode. A positional join would silently survive being attached to a
different run's results, which is exactly the failure worth designing out.

Two timings are kept apart because they are not the same quantity. `latency_s`
is what this process measured; for a replayed transcript that is the cost of a
dictionary lookup. `recorded_latency_s` is what the original call reported.
Reporting the first as model latency would understate it by orders of magnitude.

Absent is not zero. A missing token count is reported as absent, with the number
of rows that carried each field, because summing a missing count as zero quietly
understates the cost a study exists to measure.

Ported from the sibling drawer harness in `retriever-bklva`, where the same
separation is enforced against byte-identical evidence; nothing here is specific
to either simulator.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

SIDECAR_NAME = "telemetry.jsonl"
#: Fields measured at run time, which therefore differ between two runs of one
#: configuration. Everything else in a row is deterministic.
NONDETERMINISTIC_FIELDS = ("latency_s",)


@dataclass(frozen=True)
class CallRecord:
    """One policy call's measured and reported cost."""

    task: str
    method_id: str
    seed: int
    call_index: int
    prompt_sha256: str
    latency_s: float
    parse_ok: bool
    horizon: int | None = None
    transcript_hit: bool | None = None
    model_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    usd: float | None = None
    recorded_latency_s: float | None = None

    def __post_init__(self) -> None:
        if not self.task or not self.method_id:
            raise ValueError("a call record needs a task and a method_id")
        if type(self.call_index) is not int or self.call_index < 0:
            raise ValueError("call_index must be a non-negative int")
        if type(self.prompt_sha256) is not str or len(self.prompt_sha256) != 64:
            raise ValueError("prompt_sha256 must be a 64-character hex digest")
        if type(self.latency_s) is not float or not self.latency_s >= 0:
            raise ValueError("latency_s must be a non-negative float")
        for name in ("prompt_tokens", "completion_tokens", "horizon"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be None or a non-negative int")
        for name in ("usd", "recorded_latency_s"):
            value = getattr(self, name)
            if value is not None and (type(value) is not float or not value >= 0):
                raise ValueError(f"{name} must be None or a non-negative float")

    def to_dict(self) -> dict:
        return asdict(self)


def write_sidecar(directory, records) -> Path:
    """Write `telemetry.jsonl` beside an episode's results."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / SIDECAR_NAME
    with target.open("w", encoding="utf-8") as stream:
        for record in records:
            row = record.to_dict() if isinstance(record, CallRecord) else dict(record)
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return target


def read_sidecar(directory) -> list:
    path = Path(directory) / SIDECAR_NAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_join(ledger, rows) -> dict:
    """Check that a sidecar describes the episode it sits next to."""
    errors = []
    if len(ledger) != len(rows):
        errors.append(f"ledger has {len(ledger)} calls, sidecar has {len(rows)}")
    for index, (entry, row) in enumerate(zip(ledger, rows)):
        if entry.get("call_index") != row.get("call_index"):
            errors.append(f"call_index disagrees at position {index}")
        elif entry.get("prompt_sha256") != row.get("prompt_sha256"):
            errors.append(f"prompt digest disagrees at call {entry.get('call_index')}")
    if [row.get("call_index") for row in rows] != list(range(len(rows))):
        errors.append("sidecar call indices are not a complete ordered sequence")
    return {"joined": not errors, "calls": len(rows), "errors": errors}


def projected_cost(rows) -> dict:
    """Totals over a sidecar, with absent values reported rather than zeroed."""
    summary = {"calls": len(rows), "measured_latency_s": round(
        sum(float(row.get("latency_s") or 0.0) for row in rows), 6)}
    for name in ("prompt_tokens", "completion_tokens", "usd", "recorded_latency_s"):
        present = [row[name] for row in rows if row.get(name) is not None]
        summary[name] = (round(sum(present), 6) if name in ("usd", "recorded_latency_s")
                         else sum(present)) if present else None
        summary[f"{name}_reported_by"] = len(present)
    summary["parse_failures"] = sum(1 for row in rows if row.get("parse_ok") is False)
    summary["transcript_misses"] = sum(1 for row in rows if row.get("transcript_hit") is False)
    return summary
