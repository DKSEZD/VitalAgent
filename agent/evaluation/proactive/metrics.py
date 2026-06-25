"""Metrics and matching logic for proactive evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:
    from agent.data.streaming import ECGWindow
    from agent.proactive.pipeline import ProactivePipeline

KNOWN_RHYTHM_LABELS = frozenset({"N", "AF", "Other"})
UNKNOWN_RHYTHM_LABEL = "Unknown"

_RAW_RHYTHM_TO_NORMALIZED = {
    "AFIB": "AF",
    "AFL": "AF",
    "AFLUT": "AF",
    "N": "N",
    "NORMAL": "N",
    "NSR": "N",
    "SINUS": "N",
    "SR": "N",
}

@dataclass(frozen=True)
class TransitionEvent:
    patient_id: str
    segment_id: str
    window_index: int
    offset_s: float
    from_label: str
    to_label: str

    @property
    def pair(self) -> str:
        return f"{self.from_label}->{self.to_label}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"pair": self.pair}


@dataclass(frozen=True)
class WindowRhythmRecord:
    patient_id: str
    segment_id: str
    window_index: int
    offset_s: float
    truth_label: str
    predicted_label: str
    raw_truth_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_truth_labels"] = list(self.raw_truth_labels)
        return payload


@dataclass(frozen=True)
class MatchedTransition:
    ground_truth_event: TransitionEvent
    predicted_event: TransitionEvent
    latency_windows: int
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ground_truth_event": self.ground_truth_event.to_dict(),
            "predicted_event": self.predicted_event.to_dict(),
            "latency_windows": self.latency_windows,
            "latency_seconds": self.latency_seconds,
        }


@dataclass(frozen=True)
class GroundTruthEpisode:
    patient_id: str
    segment_id: str
    episode_id: int
    label: str
    start_window: int
    end_window: int
    start_offset_s: float
    end_offset_s: float

    @property
    def duration_windows(self) -> int:
        return self.end_window - self.start_window + 1

    @property
    def duration_seconds(self) -> float:
        return self.end_offset_s - self.start_offset_s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "duration_windows": self.duration_windows,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class InterventionEvent:
    patient_id: str
    segment_id: str
    window_index: int
    offset_s: float
    urgency: str
    triggered_rules: tuple[str, ...] = ()
    reason: str = ""
    condition_active: bool = False
    rule_layer: dict[str, Any] = field(default_factory=dict)
    judge_layer: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["triggered_rules"] = list(self.triggered_rules)
        return payload


@dataclass(frozen=True)
class MatchedIntervention:
    ground_truth_episode: GroundTruthEpisode
    intervention_event: InterventionEvent
    latency_windows: int
    latency_seconds: float
    duplicate_events: tuple[InterventionEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ground_truth_episode": self.ground_truth_episode.to_dict(),
            "intervention_event": self.intervention_event.to_dict(),
            "latency_windows": self.latency_windows,
            "latency_seconds": self.latency_seconds,
            "duplicate_events": [event.to_dict() for event in self.duplicate_events],
        }


@dataclass
class TransitionEvalReport:
    windows_processed: int = 0
    ground_truth_transition_total: int = 0
    predicted_transition_total: int = 0
    matched_transition_total: int = 0
    false_positive_total: int = 0
    false_negative_total: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    latency_windows_mean: float | None = None
    latency_windows_median: float | None = None
    latency_windows_p90: float | None = None
    latency_windows_max: int | None = None
    latency_seconds_mean: float | None = None
    latency_seconds_median: float | None = None
    latency_seconds_p90: float | None = None
    latency_seconds_max: float | None = None
    per_transition_pair: dict[str, dict[str, Any]] = field(default_factory=dict)
    truth_unknown_window_count: int = 0
    predicted_unknown_window_count: int = 0
    skipped_ground_truth_transition_count: int = 0
    unknown_raw_label_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InterventionReplayReport:
    windows_processed: int = 0
    ground_truth_episode_total: int = 0
    intervention_event_total: int = 0
    matched_episode_total: int = 0
    missed_episode_total: int = 0
    false_alert_total: int = 0
    duplicate_alert_total: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    latency_windows_mean: float | None = None
    latency_windows_median: float | None = None
    latency_windows_p90: float | None = None
    latency_windows_max: int | None = None
    latency_seconds_mean: float | None = None
    latency_seconds_median: float | None = None
    latency_seconds_p90: float | None = None
    latency_seconds_max: float | None = None
    per_label: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_rule: dict[str, dict[str, Any]] = field(default_factory=dict)
    truth_unknown_window_count: int = 0
    unknown_raw_label_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransitionEvalArtifacts:
    report: TransitionEvalReport
    window_records: list[WindowRhythmRecord]
    ground_truth_events: list[TransitionEvent]
    predicted_events: list[TransitionEvent]
    matches: list[MatchedTransition]

    def iter_debug_records(self) -> Iterable[dict[str, Any]]:
        for record in self.window_records:
            yield {"record_type": "window"} | record.to_dict()
        for event in self.ground_truth_events:
            yield {"record_type": "ground_truth_transition"} | event.to_dict()
        for event in self.predicted_events:
            yield {"record_type": "predicted_transition"} | event.to_dict()
        for match in self.matches:
            yield {"record_type": "matched_transition"} | match.to_dict()


@dataclass
class InterventionReplayArtifacts:
    report: InterventionReplayReport
    window_records: list[WindowRhythmRecord]
    ground_truth_episodes: list[GroundTruthEpisode]
    intervention_events: list[InterventionEvent]
    matches: list[MatchedIntervention]
    false_alerts: list[InterventionEvent]
    missed_episodes: list[GroundTruthEpisode] = field(default_factory=list)
    patient_context: dict[str, Any] | None = None

    def iter_debug_records(self) -> Iterable[dict[str, Any]]:
        if self.patient_context is not None:
            yield {"record_type": "patient_context"} | self.patient_context
        for record in self.window_records:
            yield {"record_type": "window"} | record.to_dict()
        for episode in self.ground_truth_episodes:
            yield {"record_type": "ground_truth_episode"} | episode.to_dict()
        for event in self.intervention_events:
            yield {"record_type": "intervention_event"} | event.to_dict()
        for match in self.matches:
            yield {"record_type": "matched_intervention"} | match.to_dict()
        for event in self.false_alerts:
            yield {"record_type": "false_alert"} | event.to_dict()
        for episode in self.missed_episodes:
            yield {"record_type": "missed_episode"} | episode.to_dict()


def normalize_annotation_rhythm(
    raw_label: str,
    unknown_raw_label_counts: Counter[str] | None = None,
) -> str:
    """Normalise one raw Icentia rhythm label into the proactive 4-class space."""
    token = str(raw_label).strip().upper()
    normalised = _RAW_RHYTHM_TO_NORMALIZED.get(token)
    if normalised is not None:
        return normalised
    if unknown_raw_label_counts is not None and token:
        unknown_raw_label_counts[token] += 1
    return "Other"


def derive_window_truth_label(
    window: ECGWindow,
    unknown_raw_label_counts: Counter[str] | None = None,
) -> str:
    """
    Derive one window-level truth label from annotation coverage.

    The label with the largest annotated duration wins. Windows with no
    rhythm spans are treated as ``Unknown`` in v1.
    """
    annotations = window.annotations
    if annotations is None or not annotations.rhythm_spans:
        return UNKNOWN_RHYTHM_LABEL

    duration_by_label: dict[str, int] = {}
    for span in annotations.rhythm_spans:
        raw_label = str(span.rhythm_type)
        label = normalize_annotation_rhythm(raw_label, unknown_raw_label_counts)
        duration = max(0, int(span.end_sample) - int(span.start_sample))
        duration_by_label[label] = duration_by_label.get(label, 0) + duration

    if not duration_by_label:
        return UNKNOWN_RHYTHM_LABEL

    return max(duration_by_label.items(), key=lambda item: item[1])[0]


def extract_ground_truth_transitions(
    window_records: list[WindowRhythmRecord],
) -> tuple[list[TransitionEvent], int]:
    """Extract GT transition events; adjacent pairs touching Unknown are skipped."""
    events: list[TransitionEvent] = []
    skipped = 0

    for previous, current in zip(window_records, window_records[1:], strict=False):
        if (
            previous.patient_id != current.patient_id
            or previous.segment_id != current.segment_id
        ):
            continue
        if (
            previous.truth_label == UNKNOWN_RHYTHM_LABEL
            or current.truth_label == UNKNOWN_RHYTHM_LABEL
        ):
            skipped += 1
            continue
        if previous.truth_label == current.truth_label:
            continue
        events.append(
            TransitionEvent(
                patient_id=current.patient_id,
                segment_id=current.segment_id,
                window_index=current.window_index,
                offset_s=current.offset_s,
                from_label=previous.truth_label,
                to_label=current.truth_label,
            )
        )
    return events, skipped


def extract_predicted_transitions(
    window_records: list[WindowRhythmRecord],
) -> list[TransitionEvent]:
    """Extract predicted transition events from consecutive known predicted labels."""
    events: list[TransitionEvent] = []

    for previous, current in zip(window_records, window_records[1:], strict=False):
        if (
            previous.patient_id != current.patient_id
            or previous.segment_id != current.segment_id
        ):
            continue
        if (
            previous.predicted_label not in KNOWN_RHYTHM_LABELS
            or current.predicted_label not in KNOWN_RHYTHM_LABELS
        ):
            continue
        if previous.predicted_label == current.predicted_label:
            continue
        events.append(
            TransitionEvent(
                patient_id=current.patient_id,
                segment_id=current.segment_id,
                window_index=current.window_index,
                offset_s=current.offset_s,
                from_label=previous.predicted_label,
                to_label=current.predicted_label,
            )
        )
    return events

def extract_predicted_transitions_debounced(
    window_records: list[WindowRhythmRecord],
    min_consecutive: int = 3,
) -> list[TransitionEvent]:
    """Extract predicted transitions with debounce filtering.

    A transition is only emitted when the predicted label has been
    stable (unchanged) for ``min_consecutive`` consecutive windows.
    This suppresses rapid oscillation between AF/Other that inflates
    the false-positive count.

    Parameters
    ----------
    window_records:
        Per-window records (same as ``extract_predicted_transitions``).
    min_consecutive:
        Number of consecutive windows required before a label change
        is confirmed. Default 3 (= 30 s at 10 s windows).
    """
    events: list[TransitionEvent] = []

    confirmed_label: str | None = None
    pending_label: str | None = None
    pending_count: int = 0
    # The window record where the pending streak started
    pending_start_record: WindowRhythmRecord | None = None

    prev_patient_segment: tuple[str, str] | None = None

    for record in window_records:
        current_key = (record.patient_id, record.segment_id)

        # Reset state on segment boundary
        if current_key != prev_patient_segment:
            confirmed_label = None
            pending_label = None
            pending_count = 0
            pending_start_record = None
            prev_patient_segment = current_key

        label = record.predicted_label
        if label not in KNOWN_RHYTHM_LABELS:
            continue

        # Bootstrap: first known label becomes confirmed immediately
        if confirmed_label is None:
            confirmed_label = label
            pending_label = None
            pending_count = 0
            continue

        if label == confirmed_label:
            # Stable — reset any pending streak
            pending_label = None
            pending_count = 0
            pending_start_record = None
        elif label == pending_label:
            # Continue building the pending streak
            pending_count += 1
            if pending_count >= min_consecutive:
                # Confirmed transition
                events.append(
                    TransitionEvent(
                        patient_id=record.patient_id,
                        segment_id=record.segment_id,
                        # Use the window where the streak started
                        window_index=pending_start_record.window_index,
                        offset_s=pending_start_record.offset_s,
                        from_label=confirmed_label,
                        to_label=label,
                    )
                )
                confirmed_label = label
                pending_label = None
                pending_count = 0
                pending_start_record = None
        else:
            # New candidate label — restart pending streak
            pending_label = label
            pending_count = 1
            pending_start_record = record

    return events


def match_transition_events(
    ground_truth_events: list[TransitionEvent],
    predicted_events: list[TransitionEvent],
) -> tuple[list[MatchedTransition], list[TransitionEvent]]:
    """
    Greedily match GT events to the earliest later prediction with the same to_label.

    Matching is performed independently inside each segment and is bounded
    by the next GT transition in the same segment.
    """
    matches: list[MatchedTransition] = []
    unmatched_predicted_events: list[TransitionEvent] = []

    gt_by_segment: dict[tuple[str, str], list[TransitionEvent]] = defaultdict(list)
    pred_by_segment: dict[tuple[str, str], list[TransitionEvent]] = defaultdict(list)

    for event in ground_truth_events:
        gt_by_segment[(event.patient_id, event.segment_id)].append(event)
    for event in predicted_events:
        pred_by_segment[(event.patient_id, event.segment_id)].append(event)

    all_segments = sorted(set(gt_by_segment) | set(pred_by_segment))
    for segment_key in all_segments:
        gt_events = sorted(
            gt_by_segment.get(segment_key, []),
            key=lambda event: event.window_index,
        )
        pred_events = sorted(
            pred_by_segment.get(segment_key, []),
            key=lambda event: event.window_index,
        )
        used_pred_indices: set[int] = set()

        for gt_index, gt_event in enumerate(gt_events):
            next_gt_window = (
                gt_events[gt_index + 1].window_index
                if gt_index + 1 < len(gt_events)
                else None
            )

            match_index: int | None = None
            for pred_index, pred_event in enumerate(pred_events):
                if pred_index in used_pred_indices:
                    continue
                if pred_event.window_index < gt_event.window_index:
                    continue
                if next_gt_window is not None and pred_event.window_index >= next_gt_window:
                    break
                if pred_event.to_label != gt_event.to_label:
                    continue
                match_index = pred_index
                break

            if match_index is None:
                continue

            used_pred_indices.add(match_index)
            matched_pred = pred_events[match_index]
            matches.append(
                MatchedTransition(
                    ground_truth_event=gt_event,
                    predicted_event=matched_pred,
                    latency_windows=matched_pred.window_index - gt_event.window_index,
                    latency_seconds=matched_pred.offset_s - gt_event.offset_s,
                )
            )

        for pred_index, pred_event in enumerate(pred_events):
            if pred_index not in used_pred_indices:
                unmatched_predicted_events.append(pred_event)

    return matches, unmatched_predicted_events


def extract_ground_truth_abnormal_episodes(
    window_records: list[WindowRhythmRecord],
    abnormal_labels: frozenset[str] = frozenset({"AF", "Other"}),
) -> list[GroundTruthEpisode]:
    """
    Collapse contiguous abnormal truth windows into ground-truth episodes.

    v0.1 treats each run of the same non-normal label as one episode.
    Unknown windows are not scored and break the active episode so that
    missing annotation coverage does not silently extend an abnormal span.
    """
    episodes: list[GroundTruthEpisode] = []
    active_start: WindowRhythmRecord | None = None
    active_last: WindowRhythmRecord | None = None
    active_label: str | None = None
    next_id = 0

    def close_active() -> None:
        nonlocal active_start, active_last, active_label, next_id
        if active_start is None or active_last is None or active_label is None:
            return
        episodes.append(
            GroundTruthEpisode(
                patient_id=active_start.patient_id,
                segment_id=active_start.segment_id,
                episode_id=next_id,
                label=active_label,
                start_window=active_start.window_index,
                end_window=active_last.window_index,
                start_offset_s=active_start.offset_s,
                end_offset_s=active_last.offset_s,
            )
        )
        next_id += 1
        active_start = None
        active_last = None
        active_label = None

    for record in window_records:
        key_changed = (
            active_start is not None
            and (
                record.patient_id != active_start.patient_id
                or record.segment_id != active_start.segment_id
            )
        )
        if key_changed:
            close_active()

        if record.truth_label not in abnormal_labels:
            close_active()
            continue

        if active_start is None:
            active_start = record
            active_last = record
            active_label = record.truth_label
            continue

        if record.truth_label != active_label:
            close_active()
            active_start = record
            active_last = record
            active_label = record.truth_label
            continue

        active_last = record

    close_active()
    return episodes


def match_intervention_events(
    ground_truth_episodes: list[GroundTruthEpisode],
    intervention_events: list[InterventionEvent],
    pre_onset_tolerance_windows: int = 0,
) -> tuple[list[MatchedIntervention], list[InterventionEvent]]:
    """
    Match emitted interventions to ground-truth abnormal episodes.

    The first alert inside an episode's scoring window is the episode hit.
    Additional alerts in the same scoring window are duplicates. Alerts
    outside every scoring window are false alerts.
    """
    matches: list[MatchedIntervention] = []
    false_alerts: list[InterventionEvent] = []
    used_event_indices: set[int] = set()

    episodes_by_segment: dict[tuple[str, str], list[GroundTruthEpisode]] = defaultdict(list)
    events_by_segment: dict[tuple[str, str], list[tuple[int, InterventionEvent]]] = defaultdict(list)
    for episode in ground_truth_episodes:
        episodes_by_segment[(episode.patient_id, episode.segment_id)].append(episode)
    for event_index, event in enumerate(intervention_events):
        events_by_segment[(event.patient_id, event.segment_id)].append((event_index, event))

    for segment_key, episodes in episodes_by_segment.items():
        sorted_episodes = sorted(episodes, key=lambda episode: episode.start_window)
        sorted_events = sorted(
            events_by_segment.get(segment_key, []),
            key=lambda item: item[1].window_index,
        )

        for episode in sorted_episodes:
            scoring_start = episode.start_window - max(0, pre_onset_tolerance_windows)
            candidates: list[tuple[int, InterventionEvent]] = [
                (event_index, event)
                for event_index, event in sorted_events
                if event_index not in used_event_indices
                and scoring_start <= event.window_index <= episode.end_window
            ]
            if not candidates:
                continue

            first_index, first_event = candidates[0]
            duplicate_events = tuple(event for _, event in candidates[1:])
            used_event_indices.update(index for index, _ in candidates)
            matches.append(
                MatchedIntervention(
                    ground_truth_episode=episode,
                    intervention_event=first_event,
                    latency_windows=first_event.window_index - episode.start_window,
                    latency_seconds=first_event.offset_s - episode.start_offset_s,
                    duplicate_events=duplicate_events,
                )
            )
            used_event_indices.add(first_index)

    for event_index, event in enumerate(intervention_events):
        if event_index not in used_event_indices:
            false_alerts.append(event)

    return matches, false_alerts


def evaluate_transition_detection(
    windows: Iterable[ECGWindow],
    pipeline: ProactivePipeline,
    debounce_windows: int = 3,
) -> TransitionEvalArtifacts:
    """Run the proactive pipeline over windows and score transition detection."""
    unknown_raw_label_counts: Counter[str] = Counter()
    window_records: list[WindowRhythmRecord] = []

    for window in windows:
        pipeline.process_window(window)
        predicted_label = _normalise_predicted_label(
            (pipeline.tracker.get_current_state().get("short") or {}).get("rhythm_class")
        )
        truth_label = derive_window_truth_label(window, unknown_raw_label_counts)
        raw_truth_labels = tuple(
            str(span.rhythm_type)
            for span in (window.annotations.rhythm_spans if window.annotations else [])
        )
        window_records.append(
            WindowRhythmRecord(
                patient_id=window.patient_id,
                segment_id=window.segment_id,
                window_index=window.window_index,
                offset_s=window.stream_offset_s,
                truth_label=truth_label,
                predicted_label=predicted_label,
                raw_truth_labels=raw_truth_labels,
            )
        )

    ground_truth_events, skipped_ground_truth_transition_count = (
        extract_ground_truth_transitions(window_records)
    )
    if debounce_windows <= 1:
        predicted_events = extract_predicted_transitions(window_records)
    else:
        predicted_events = extract_predicted_transitions_debounced(
            window_records, min_consecutive=debounce_windows,
        )
    matches, unmatched_predicted_events = match_transition_events(
        ground_truth_events,
        predicted_events,
    )
    report = _build_report(
        window_records=window_records,
        ground_truth_events=ground_truth_events,
        predicted_events=predicted_events,
        matches=matches,
        unmatched_predicted_events=unmatched_predicted_events,
        skipped_ground_truth_transition_count=skipped_ground_truth_transition_count,
        unknown_raw_label_counts=unknown_raw_label_counts,
    )
    return TransitionEvalArtifacts(
        report=report,
        window_records=window_records,
        ground_truth_events=ground_truth_events,
        predicted_events=predicted_events,
        matches=matches,
    )


def evaluate_intervention_replay(
    windows: Iterable[ECGWindow],
    pipeline: ProactivePipeline,
    abnormal_labels: frozenset[str] = frozenset({"AF", "Other"}),
    pre_onset_tolerance_windows: int = 0,
) -> InterventionReplayArtifacts:
    """
    Replay annotated windows through the proactive pipeline and score alerts.

    This is the v0.1 offline-intervention loop: continuous annotated data
    goes through ``ProactivePipeline.process_window()``, emitted intervention
    events are collected, and those events are aligned to annotation-derived
    abnormal episodes.
    """
    unknown_raw_label_counts: Counter[str] = Counter()
    window_records: list[WindowRhythmRecord] = []
    intervention_events: list[InterventionEvent] = []

    for window in windows:
        result = pipeline.process_window(window)
        predicted_label = _normalise_pipeline_label(pipeline, result)
        truth_label = derive_window_truth_label(window, unknown_raw_label_counts)
        raw_truth_labels = tuple(
            str(span.rhythm_type)
            for span in (window.annotations.rhythm_spans if window.annotations else [])
        )
        window_records.append(
            WindowRhythmRecord(
                patient_id=window.patient_id,
                segment_id=window.segment_id,
                window_index=window.window_index,
                offset_s=window.stream_offset_s,
                truth_label=truth_label,
                predicted_label=predicted_label,
                raw_truth_labels=raw_truth_labels,
            )
        )

        if bool(result.get("intervene", False)):
            intervention_events.append(_intervention_event_from_result(result, window))

    ground_truth_episodes = extract_ground_truth_abnormal_episodes(
        window_records,
        abnormal_labels=abnormal_labels,
    )
    matches, false_alerts = match_intervention_events(
        ground_truth_episodes,
        intervention_events,
        pre_onset_tolerance_windows=pre_onset_tolerance_windows,
    )
    matched_episode_ids = {
        match.ground_truth_episode.episode_id
        for match in matches
    }
    missed_episodes = [
        episode
        for episode in ground_truth_episodes
        if episode.episode_id not in matched_episode_ids
    ]
    report = _build_intervention_report(
        window_records=window_records,
        ground_truth_episodes=ground_truth_episodes,
        intervention_events=intervention_events,
        matches=matches,
        false_alerts=false_alerts,
        unknown_raw_label_counts=unknown_raw_label_counts,
    )
    patient_context = _build_patient_context_snapshot(
        patient_id=(window_records[0].patient_id if window_records else ""),
        intervention_events=intervention_events,
        matches=matches,
        false_alerts=false_alerts,
    )
    return InterventionReplayArtifacts(
        report=report,
        window_records=window_records,
        ground_truth_episodes=ground_truth_episodes,
        intervention_events=intervention_events,
        matches=matches,
        false_alerts=false_alerts,
        missed_episodes=missed_episodes,
        patient_context=patient_context,
    )


def _normalise_pipeline_label(
    pipeline: ProactivePipeline,
    result: dict[str, Any],
) -> str:
    tracker = getattr(pipeline, "tracker", None)
    if tracker is not None and hasattr(tracker, "get_current_state"):
        state = tracker.get_current_state()
        label = (state.get("short") or {}).get("rhythm_class")
        return _normalise_predicted_label(label)
    return _normalise_predicted_label(result.get("rhythm_class"))


def _intervention_event_from_result(
    result: dict[str, Any],
    window: ECGWindow,
) -> InterventionEvent:
    return InterventionEvent(
        patient_id=str(result.get("patient_id", window.patient_id)),
        segment_id=str(result.get("segment_id", window.segment_id)),
        window_index=int(result.get("window_index", window.window_index)),
        offset_s=float(result.get("stream_offset_s", window.stream_offset_s)),
        urgency=str(result.get("urgency", "none")),
        triggered_rules=tuple(str(rule) for rule in (result.get("triggered_rules") or [])),
        reason=str(result.get("reason", "")),
        condition_active=bool(result.get("condition_active", False)),
        rule_layer=dict(result.get("rule_layer") or {}),
        judge_layer=dict(result.get("judge_layer") or {}),
    )


def _event_key(event: InterventionEvent) -> tuple[str, str, int, float]:
    return (
        event.patient_id,
        event.segment_id,
        event.window_index,
        event.offset_s,
    )


def _build_patient_context_snapshot(
    *,
    patient_id: str,
    intervention_events: list[InterventionEvent],
    matches: list[MatchedIntervention],
    false_alerts: list[InterventionEvent],
) -> dict[str, Any]:
    from agent.proactive.patient_context import PatientContext

    context = PatientContext(patient_id=patient_id)
    match_by_event = {
        _event_key(match.intervention_event): match
        for match in matches
    }
    false_alert_keys = {_event_key(event) for event in false_alerts}

    for event in intervention_events:
        match = match_by_event.get(_event_key(event))
        episode = match.ground_truth_episode if match is not None else None
        context.record_alert(
            patient_id=event.patient_id,
            segment_id=event.segment_id,
            window_index=event.window_index,
            offset_s=event.offset_s,
            urgency=event.urgency,
            triggered_rules=event.triggered_rules,
            reason=event.reason,
            condition_active=event.condition_active,
            rule_layer=event.rule_layer,
            judge_layer=event.judge_layer,
            matched_episode_id=(episode.episode_id if episode else None),
            matched_episode_label=(episode.label if episode else None),
            matched_episode_start_offset_s=(
                episode.start_offset_s if episode else None
            ),
            matched_episode_duration_s=(
                episode.duration_seconds if episode else None
            ),
            latency_seconds=(
                match.latency_seconds if match is not None else None
            ),
            is_false_alert=_event_key(event) in false_alert_keys,
        )

    return context.to_dict()


def _normalise_predicted_label(label: str | None) -> str:
    token = str(label).strip() if label is not None else ""
    return token if token in KNOWN_RHYTHM_LABELS else UNKNOWN_RHYTHM_LABEL


def _build_report(
    *,
    window_records: list[WindowRhythmRecord],
    ground_truth_events: list[TransitionEvent],
    predicted_events: list[TransitionEvent],
    matches: list[MatchedTransition],
    unmatched_predicted_events: list[TransitionEvent],
    skipped_ground_truth_transition_count: int,
    unknown_raw_label_counts: Counter[str],
) -> TransitionEvalReport:
    ground_truth_total = len(ground_truth_events)
    predicted_total = len(predicted_events)
    matched_total = len(matches)
    false_negative_total = max(0, ground_truth_total - matched_total)
    false_positive_total = len(unmatched_predicted_events)

    precision = matched_total / predicted_total if predicted_total else 0.0
    recall = matched_total / ground_truth_total if ground_truth_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision > 0 and recall > 0
        else 0.0
    )

    latency_windows = np.asarray([match.latency_windows for match in matches], dtype=float)
    latency_seconds = np.asarray([match.latency_seconds for match in matches], dtype=float)

    truth_unknown_window_count = sum(
        1 for record in window_records if record.truth_label == UNKNOWN_RHYTHM_LABEL
    )
    predicted_unknown_window_count = sum(
        1 for record in window_records if record.predicted_label == UNKNOWN_RHYTHM_LABEL
    )

    return TransitionEvalReport(
        windows_processed=len(window_records),
        ground_truth_transition_total=ground_truth_total,
        predicted_transition_total=predicted_total,
        matched_transition_total=matched_total,
        false_positive_total=false_positive_total,
        false_negative_total=false_negative_total,
        precision=precision,
        recall=recall,
        f1=f1,
        latency_windows_mean=_stat_mean(latency_windows),
        latency_windows_median=_stat_median(latency_windows),
        latency_windows_p90=_stat_percentile(latency_windows, 90),
        latency_windows_max=_stat_max_int(latency_windows),
        latency_seconds_mean=_stat_mean(latency_seconds),
        latency_seconds_median=_stat_median(latency_seconds),
        latency_seconds_p90=_stat_percentile(latency_seconds, 90),
        latency_seconds_max=_stat_max_float(latency_seconds),
        per_transition_pair=_build_per_pair_breakdown(
            ground_truth_events=ground_truth_events,
            predicted_events=predicted_events,
            matches=matches,
        ),
        truth_unknown_window_count=truth_unknown_window_count,
        predicted_unknown_window_count=predicted_unknown_window_count,
        skipped_ground_truth_transition_count=skipped_ground_truth_transition_count,
        unknown_raw_label_counts=dict(sorted(unknown_raw_label_counts.items())),
    )


def _build_intervention_report(
    *,
    window_records: list[WindowRhythmRecord],
    ground_truth_episodes: list[GroundTruthEpisode],
    intervention_events: list[InterventionEvent],
    matches: list[MatchedIntervention],
    false_alerts: list[InterventionEvent],
    unknown_raw_label_counts: Counter[str],
) -> InterventionReplayReport:
    episode_total = len(ground_truth_episodes)
    event_total = len(intervention_events)
    matched_total = len(matches)
    missed_total = max(0, episode_total - matched_total)
    duplicate_total = sum(len(match.duplicate_events) for match in matches)
    false_total = len(false_alerts)

    precision = matched_total / event_total if event_total else 0.0
    recall = matched_total / episode_total if episode_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision > 0 and recall > 0
        else 0.0
    )

    latency_windows = np.asarray([match.latency_windows for match in matches], dtype=float)
    latency_seconds = np.asarray([match.latency_seconds for match in matches], dtype=float)

    truth_unknown_window_count = sum(
        1 for record in window_records if record.truth_label == UNKNOWN_RHYTHM_LABEL
    )

    return InterventionReplayReport(
        windows_processed=len(window_records),
        ground_truth_episode_total=episode_total,
        intervention_event_total=event_total,
        matched_episode_total=matched_total,
        missed_episode_total=missed_total,
        false_alert_total=false_total,
        duplicate_alert_total=duplicate_total,
        precision=precision,
        recall=recall,
        f1=f1,
        latency_windows_mean=_stat_mean(latency_windows),
        latency_windows_median=_stat_median(latency_windows),
        latency_windows_p90=_stat_percentile(latency_windows, 90),
        latency_windows_max=_stat_max_int(latency_windows),
        latency_seconds_mean=_stat_mean(latency_seconds),
        latency_seconds_median=_stat_median(latency_seconds),
        latency_seconds_p90=_stat_percentile(latency_seconds, 90),
        latency_seconds_max=_stat_max_float(latency_seconds),
        per_label=_build_per_label_intervention_breakdown(
            ground_truth_episodes=ground_truth_episodes,
            matches=matches,
        ),
        per_rule=_build_per_rule_intervention_breakdown(
            intervention_events=intervention_events,
            matches=matches,
            false_alerts=false_alerts,
        ),
        truth_unknown_window_count=truth_unknown_window_count,
        unknown_raw_label_counts=dict(sorted(unknown_raw_label_counts.items())),
    )


def _build_per_label_intervention_breakdown(
    *,
    ground_truth_episodes: list[GroundTruthEpisode],
    matches: list[MatchedIntervention],
) -> dict[str, dict[str, Any]]:
    gt_by_label = Counter(episode.label for episode in ground_truth_episodes)
    matched_by_label = Counter(match.ground_truth_episode.label for match in matches)
    duplicate_by_label = Counter(
        match.ground_truth_episode.label
        for match in matches
        for _event in match.duplicate_events
    )

    labels = sorted(set(gt_by_label) | set(matched_by_label))
    breakdown: dict[str, dict[str, Any]] = {}
    for label in labels:
        gt_total = gt_by_label.get(label, 0)
        matched_total = matched_by_label.get(label, 0)
        recall = matched_total / gt_total if gt_total else 0.0
        breakdown[label] = {
            "ground_truth_total": gt_total,
            "matched_total": matched_total,
            "missed_total": max(0, gt_total - matched_total),
            "duplicate_alert_total": duplicate_by_label.get(label, 0),
            "recall": recall,
        }
    return breakdown


def _rule_keys(event: InterventionEvent) -> tuple[str, ...]:
    return event.triggered_rules or ("(none)",)


def _build_per_rule_intervention_breakdown(
    *,
    intervention_events: list[InterventionEvent],
    matches: list[MatchedIntervention],
    false_alerts: list[InterventionEvent],
) -> dict[str, dict[str, Any]]:
    event_by_rule: Counter[str] = Counter()
    matched_by_rule: Counter[str] = Counter()
    duplicate_by_rule: Counter[str] = Counter()
    false_by_rule: Counter[str] = Counter()

    for event in intervention_events:
        for rule in _rule_keys(event):
            event_by_rule[rule] += 1

    for match in matches:
        for rule in _rule_keys(match.intervention_event):
            matched_by_rule[rule] += 1
        for duplicate in match.duplicate_events:
            for rule in _rule_keys(duplicate):
                duplicate_by_rule[rule] += 1

    for event in false_alerts:
        for rule in _rule_keys(event):
            false_by_rule[rule] += 1

    rules = sorted(set(event_by_rule) | set(matched_by_rule) | set(false_by_rule))
    breakdown: dict[str, dict[str, Any]] = {}
    for rule in rules:
        alert_total = event_by_rule.get(rule, 0)
        matched_total = matched_by_rule.get(rule, 0)
        false_total = false_by_rule.get(rule, 0)
        duplicate_total = duplicate_by_rule.get(rule, 0)
        precision = matched_total / alert_total if alert_total else 0.0
        breakdown[rule] = {
            "alert_total": alert_total,
            "matched_episode_total": matched_total,
            "false_alert_total": false_total,
            "duplicate_alert_total": duplicate_total,
            "precision": precision,
        }
    return breakdown


def _build_per_pair_breakdown(
    *,
    ground_truth_events: list[TransitionEvent],
    predicted_events: list[TransitionEvent],
    matches: list[MatchedTransition],
) -> dict[str, dict[str, Any]]:
    gt_by_pair = Counter(event.pair for event in ground_truth_events)
    pred_by_pair = Counter(event.pair for event in predicted_events)
    matched_by_pair = Counter(
        match.ground_truth_event.pair
        for match in matches
        if match.ground_truth_event.pair == match.predicted_event.pair
    )

    pairs = sorted(set(gt_by_pair) | set(pred_by_pair))
    breakdown: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        gt_total = gt_by_pair.get(pair, 0)
        pred_total = pred_by_pair.get(pair, 0)
        matched_total = matched_by_pair.get(pair, 0)
        precision = matched_total / pred_total if pred_total else 0.0
        recall = matched_total / gt_total if gt_total else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision > 0 and recall > 0
            else 0.0
        )
        breakdown[pair] = {
            "ground_truth_total": gt_total,
            "predicted_total": pred_total,
            "matched_total": matched_total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return breakdown


def _stat_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _stat_median(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.median(values))


def _stat_percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _stat_max_int(values: np.ndarray) -> int | None:
    if values.size == 0:
        return None
    return int(np.max(values))


def _stat_max_float(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.max(values))

