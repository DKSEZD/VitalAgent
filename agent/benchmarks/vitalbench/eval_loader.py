"""Agent-side loader for VitalBench reactive evaluation JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
import re
from typing import Any, Iterator, Optional

from agent.mhealth.schemas import canonicalize_dataset_name


@dataclass(frozen=True)
class WindowLocator:
    """Identifies a contiguous time window of raw signal for a given patient.

    All five VitalBench datasets use this uniform locator schema. Tools
    consume (dataset, patient_id, window_start_s, window_end_s) directly;
    no per-dataset decoder is needed.
    """

    dataset: str
    patient_id: str
    window_start_s: float
    window_end_s: float
    recording_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", canonicalize_dataset_name(self.dataset))


@dataclass(frozen=True)
class AgentInput:
    """The complete view of a QA sample visible to the agent at eval time.

    The agent sees the natural-language question, the answer space (for
    verify/choose questions), and a window locator pointing at raw signal.
    The agent does not see: the answer, evaluation metadata, expected tools,
    or any GT-side analytical fields.
    """

    question_id: str
    question: str
    window_locator: WindowLocator
    answer_set: Optional[list[str]] = None


@dataclass(frozen=True)
class EvalSpec:
    """The view used by the evaluator to score the agent's answer."""

    question_id: str
    answer: list[str]
    answer_type: str
    evaluation_method: str
    numeric_tolerance: Optional[float]
    answer_set: Optional[list[str]]


@dataclass(frozen=True)
class QASample:
    """All three views from one JSONL row, returned together for convenience.

    Downstream code is expected to:
    - Pass `agent_input` to the agent
    - Pass `eval_spec` to the evaluator
    - Carry `gt_metadata` through to the result record for post-hoc analysis,
      but never use it during agent execution or scoring
    """

    agent_input: AgentInput
    eval_spec: EvalSpec
    gt_metadata: dict[str, Any]


_AGENT_FIELDS: frozenset[str] = frozenset(
    {
        "question_id",
        "question",
        "answer_set",
    }
)

_LOCATOR_FIELDS: frozenset[str] = frozenset(
    {
        "dataset",
        "patient_id",
        "recording_id",
        "segment_id",
        "window_start_s",
        "window_duration_s",
    }
)

_ICENTIA_SEGMENT_RE = re.compile(r"(?:^|_)segment_(\d+)(?:_|$)")


def _raw_recording_id(row: dict[str, Any]) -> str | None:
    recording_id = row.get("recording_id") or row.get("segment_id")
    if recording_id is not None:
        return str(recording_id)
    if str(row.get("dataset")) == "icentia11k":
        match = _ICENTIA_SEGMENT_RE.search(str(row.get("source_state_id") or ""))
        if match:
            return match.group(1).zfill(2)
    return None

_EVAL_FIELDS: frozenset[str] = frozenset(
    {
        "question_id",
        "answer",
        "answer_type",
        "evaluation_method",
        "numeric_tolerance",
        "answer_set",
    }
)


def load_qa_jsonl(path: str | Path) -> Iterator[QASample]:
    """Stream-load QA samples from a JSONL file.

    Yields one QASample per line. Memory-friendly for large benchmark files.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if a row is missing required fields, or if window time
            fields are null.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"QA JSONL not found: {path}")

    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Invalid QA row at {path}:{line_num}: expected JSON object"
                )

            try:
                yield _split_row(row)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to parse row at {path}:{line_num}: {exc}"
                ) from exc


def _split_row(row: dict[str, Any]) -> QASample:
    """Project one JSONL row into agent / eval / gt_metadata views."""

    window_start_s = row.get("window_start_s")
    window_duration_s = row.get("window_duration_s")
    if window_start_s is None or window_duration_s is None:
        raise ValueError(
            f"Sample {row.get('question_id', '<unknown>')!r} is missing "
            "window_start_s or window_duration_s; agent cannot locate signal"
        )

    locator = WindowLocator(
        dataset=canonicalize_dataset_name(row["dataset"]),
        patient_id=row["patient_id"],
        window_start_s=float(window_start_s),
        window_end_s=float(window_start_s) + float(window_duration_s),
        recording_id=_raw_recording_id(row),
    )

    raw_answer_set = row.get("answer_set")
    if raw_answer_set is not None and len(raw_answer_set) == 0:
        normalized_answer_set: Optional[list[str]] = None
    else:
        normalized_answer_set = list(raw_answer_set) if raw_answer_set else None

    agent_input = AgentInput(
        question_id=row["question_id"],
        question=row["question"],
        window_locator=locator,
        answer_set=normalized_answer_set,
    )

    eval_spec = EvalSpec(
        question_id=row["question_id"],
        answer=list(row["answer"]),
        answer_type=row["answer_type"],
        evaluation_method=row["evaluation_method"],
        numeric_tolerance=row.get("numeric_tolerance"),
        answer_set=normalized_answer_set,
    )

    consumed = _AGENT_FIELDS | _LOCATOR_FIELDS | _EVAL_FIELDS
    gt_metadata = {k: v for k, v in row.items() if k not in consumed}

    _assert_no_answer_leak(agent_input)

    return QASample(
        agent_input=agent_input,
        eval_spec=eval_spec,
        gt_metadata=gt_metadata,
    )


def _assert_no_answer_leak(agent_input: AgentInput) -> None:
    """Enforce that AgentInput cannot contain any field named 'answer'."""

    leaked = {f.name for f in fields(agent_input)} & {"answer", "answer_type"}
    if leaked:
        raise AssertionError(
            f"AgentInput must never contain answer fields; found {leaked}"
        )


__all__ = [
    "WindowLocator",
    "AgentInput",
    "EvalSpec",
    "QASample",
    "load_qa_jsonl",
]
