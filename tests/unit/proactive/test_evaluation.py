from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from agent.data.streaming import ECGWindow, RhythmSpan, WindowAnnotations
from agent.evaluation.proactive.multi_patient import (
    MultiPatientEntry,
    build_multi_patient_entries,
)
from agent.evaluation.proactive.main import (
    _build_parser,
    _discover_icentia11k_patient_ids,
)
from agent.evaluation.proactive.config import (
    PROJECT_ROOT,
    default_icentia11k_dataset_root,
    default_afppgecg_dataset_root,
)
from agent.evaluation.proactive.metrics import (
    GroundTruthEpisode,
    InterventionEvent,
    KNOWN_RHYTHM_LABELS,
    UNKNOWN_RHYTHM_LABEL,
    TransitionEvent,
    derive_window_truth_label,
    evaluate_intervention_replay,
    evaluate_transition_detection,
    extract_ground_truth_abnormal_episodes,
    extract_ground_truth_transitions,
    extract_predicted_transitions,
    match_intervention_events,
    match_transition_events,
    normalize_annotation_rhythm,
    WindowRhythmRecord,
)


def _make_window(
    *,
    window_index: int,
    segment_id: str = "03",
    offset_s: float | None = None,
    spans: list[RhythmSpan] | None = None,
) -> ECGWindow:
    signal_len = 2500
    return ECGWindow(
        signal=[0.0] * signal_len,  # type: ignore[arg-type]
        fs=250,
        patient_id="00012",
        segment_id=segment_id,
        window_index=window_index,
        start_sample=window_index * signal_len,
        end_sample=(window_index + 1) * signal_len,
        n_samples=signal_len,
        stream_offset_s=float(window_index * 10) if offset_s is None else offset_s,
        annotations=WindowAnnotations(rhythm_spans=spans or []) if spans is not None else None,
    )


def _record(
    *,
    window_index: int,
    truth_label: str,
    predicted_label: str,
    segment_id: str = "03",
    offset_s: float | None = None,
) -> WindowRhythmRecord:
    return WindowRhythmRecord(
        patient_id="00012",
        segment_id=segment_id,
        window_index=window_index,
        offset_s=float(window_index * 10) if offset_s is None else offset_s,
        truth_label=truth_label,
        predicted_label=predicted_label,
    )


class _FakePipeline:
    def __init__(self, predicted_labels: list[str]) -> None:
        self._predicted_labels = iter(predicted_labels)
        self._state = {
            "short": {"rhythm_class": UNKNOWN_RHYTHM_LABEL},
            "medium": {},
            "long": {},
            "embedding": {"latest_zscore": 0.0},
        }
        self.tracker = SimpleNamespace(get_current_state=self._get_state)

    def process_window(self, window: ECGWindow) -> dict[str, object]:
        del window
        self._state = {
            "short": {"rhythm_class": next(self._predicted_labels)},
            "medium": {},
            "long": {},
            "embedding": {"latest_zscore": 0.0},
        }
        return {}

    def _get_state(self) -> dict[str, object]:
        return self._state


class _FakeInterventionPipeline:
    def __init__(
        self,
        predicted_labels: list[str],
        intervention_windows: set[int],
    ) -> None:
        self._predicted_labels = iter(predicted_labels)
        self._intervention_windows = intervention_windows
        self._state = {
            "short": {"rhythm_class": UNKNOWN_RHYTHM_LABEL},
            "medium": {},
            "long": {},
            "embedding": {"latest_zscore": 0.0},
        }
        self.tracker = SimpleNamespace(get_current_state=self._get_state)

    def process_window(self, window: ECGWindow) -> dict[str, object]:
        label = next(self._predicted_labels)
        self._state = {
            "short": {"rhythm_class": label},
            "medium": {},
            "long": {},
            "embedding": {"latest_zscore": 0.0},
        }
        intervene = window.window_index in self._intervention_windows
        return {
            "intervene": intervene,
            "condition_active": intervene,
            "urgency": "medium" if intervene else "none",
            "reason": "test alert" if intervene else "",
            "triggered_rules": ["test_rule"] if intervene else [],
            "window_index": window.window_index,
            "patient_id": window.patient_id,
            "segment_id": window.segment_id,
            "stream_offset_s": window.stream_offset_s,
            "rule_layer": {"intervene": intervene},
            "judge_layer": {"intervene": False},
        }

    def _get_state(self) -> dict[str, object]:
        return self._state


class _FakeMultiPatientLoader:
    @staticmethod
    def normalize_patient_id(patient_id: str | int) -> str:
        return f"{int(patient_id):05d}"

    @staticmethod
    def available_patient_ids() -> list[str]:
        return ["001", "002"]

    @staticmethod
    def get_patient_segments(patient_id: str):
        del patient_id
        return [SimpleNamespace(segment_id="ecg")]


class _FakeAFPPGECGLoaderWithBadPatient(_FakeMultiPatientLoader):
    @staticmethod
    def available_patient_ids() -> list[str]:
        return ["001", "028", "002"]

    @staticmethod
    def get_patient_segments(patient_id: str):
        if patient_id == "00028":
            raise OSError("file signature not found")
        return [SimpleNamespace(segment_id="ecg")]


def test_normalize_annotation_rhythm_maps_known_labels() -> None:
    assert normalize_annotation_rhythm("AFIB") == "AF"
    assert normalize_annotation_rhythm("AFL") == "AF"    # AFL merged into AF
    assert normalize_annotation_rhythm("AFLUT") == "AF"  # AFLUT merged into AF
    assert normalize_annotation_rhythm("N") == "N"
    assert normalize_annotation_rhythm("NSR") == "N"


def test_default_icentia11k_dataset_root_reads_env_relative_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.config import reset_config

    monkeypatch.setenv("DATASET_ICENTIA11K_ROOT", "dataset/custom_icentia")
    reset_config()

    expected = (PROJECT_ROOT / "dataset/custom_icentia").resolve()
    assert default_icentia11k_dataset_root() == expected
    reset_config()


def test_default_afppgecg_dataset_root_reads_env_absolute_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from agent.config import reset_config

    monkeypatch.setenv("DATASET_AFPPGECG_ROOT", str(tmp_path))
    reset_config()

    assert default_afppgecg_dataset_root() == tmp_path
    reset_config()


def test_parser_does_not_default_to_single_patient_or_window_limit() -> None:
    args = _build_parser().parse_args([])

    assert args.patient_id is None
    assert args.max_windows is None


def test_parser_uses_patient_id_as_the_only_patient_selector() -> None:
    args = _build_parser().parse_args(
        [
            "--patient-id",
            "1182",
            "--output-dir",
            "results/proactive/rule_only_icentia25",
            "--llm-mode",
            "disabled",
        ]
    )

    assert args.patient_id == "1182"
    assert str(args.output_dir).endswith("rule_only_icentia25")
    assert args.llm_mode == "disabled"
    assert not hasattr(args, "batch")
    assert not hasattr(args, "patients")


def test_build_multi_patient_entries_preserves_curated_icentia_notes() -> None:
    patients = build_multi_patient_entries(
        dataset="icentia11k",
        loader=_FakeMultiPatientLoader(),
        dataset_root=PROJECT_ROOT,
        requested_patient_ids=["1182", "500"],
        explicit_segment_id=None,
    )

    assert patients == [
        MultiPatientEntry("01182", "01", "AFIB+N holdout"),
        MultiPatientEntry("00500", None, "ad-hoc"),
    ]


def test_build_multi_patient_entries_applies_curated_icentia_notes_to_discovery(tmp_path) -> None:
    (tmp_path / "p01" / "p01182").mkdir(parents=True)
    (tmp_path / "p05" / "p00500").mkdir(parents=True)

    patients = build_multi_patient_entries(
        dataset="icentia11k",
        loader=_FakeMultiPatientLoader(),
        dataset_root=tmp_path,
        requested_patient_ids=None,
        explicit_segment_id=None,
    )

    assert patients == [
        MultiPatientEntry("00500", None, "ad-hoc"),
        MultiPatientEntry("01182", "01", "AFIB+N holdout"),
    ]


def test_build_multi_patient_entries_uses_available_afppgecg_patients() -> None:
    patients = build_multi_patient_entries(
        dataset="afppgecg",
        loader=_FakeMultiPatientLoader(),
        dataset_root=PROJECT_ROOT,
        requested_patient_ids=None,
        explicit_segment_id=None,
    )

    assert patients == [
        MultiPatientEntry("00001", None, "afppgecg"),
        MultiPatientEntry("00002", None, "afppgecg"),
    ]


def test_build_multi_patient_entries_skips_unreadable_afppgecg_patients() -> None:
    patients = build_multi_patient_entries(
        dataset="afppgecg",
        loader=_FakeAFPPGECGLoaderWithBadPatient(),
        dataset_root=PROJECT_ROOT,
        requested_patient_ids=None,
        explicit_segment_id=None,
    )

    assert patients == [
        MultiPatientEntry("00001", None, "afppgecg"),
        MultiPatientEntry("00002", None, "afppgecg"),
    ]


def test_discover_icentia11k_patient_ids_from_local_layout(tmp_path) -> None:
    (tmp_path / "p00" / "p00012").mkdir(parents=True)
    (tmp_path / "p01" / "p01000").mkdir(parents=True)
    (tmp_path / "notes").mkdir()

    assert _discover_icentia11k_patient_ids(tmp_path) == ["00012", "01000"]


def test_discover_icentia11k_patient_ids_requires_local_patients(tmp_path) -> None:
    with pytest.raises(ValueError, match="No Icentia11k patient directories"):
        _discover_icentia11k_patient_ids(tmp_path)


def test_normalize_annotation_rhythm_counts_unknown_labels_as_other() -> None:
    counter: Counter[str] = Counter()
    result = normalize_annotation_rhythm("vt", counter)

    assert result == "Other"
    assert counter == {"VT": 1}


def test_derive_window_truth_label_uses_single_span() -> None:
    window = _make_window(
        window_index=0,
        spans=[RhythmSpan(start_sample=0, end_sample=2500, rhythm_type="AFIB")],
    )

    assert derive_window_truth_label(window) == "AF"


def test_derive_window_truth_label_chooses_longest_duration() -> None:
    window = _make_window(
        window_index=0,
        spans=[
            RhythmSpan(start_sample=0, end_sample=1000, rhythm_type="AFIB"),
            RhythmSpan(start_sample=1000, end_sample=2500, rhythm_type="N"),
        ],
    )

    assert derive_window_truth_label(window) == "N"


def test_derive_window_truth_label_returns_unknown_for_missing_annotations() -> None:
    window = _make_window(window_index=0, spans=None)

    assert derive_window_truth_label(window) == UNKNOWN_RHYTHM_LABEL


def test_extract_ground_truth_transitions_creates_events_and_skips_unknown() -> None:
    records = [
        _record(window_index=0, truth_label="N", predicted_label="N"),
        _record(window_index=1, truth_label="AF", predicted_label="N"),
        _record(window_index=2, truth_label=UNKNOWN_RHYTHM_LABEL, predicted_label="N"),
        _record(window_index=3, truth_label="N", predicted_label="N"),
    ]

    events, skipped = extract_ground_truth_transitions(records)

    assert events == [
        TransitionEvent(
            patient_id="00012",
            segment_id="03",
            window_index=1,
            offset_s=10.0,
            from_label="N",
            to_label="AF",
        )
    ]
    assert skipped == 2


def test_extract_predicted_transitions_ignores_unknown() -> None:
    records = [
        _record(window_index=0, truth_label="N", predicted_label=UNKNOWN_RHYTHM_LABEL),
        _record(window_index=1, truth_label="N", predicted_label="N"),
        _record(window_index=2, truth_label="N", predicted_label="AF"),
        _record(window_index=3, truth_label="N", predicted_label="AF"),
        _record(window_index=4, truth_label="N", predicted_label="Other"),
    ]

    events = extract_predicted_transitions(records)

    assert [event.pair for event in events] == ["N->AF", "AF->Other"]


def test_match_transition_events_matches_exact_window() -> None:
    gt = [
        TransitionEvent("00012", "03", 5, 50.0, "N", "AF"),
    ]
    pred = [
        TransitionEvent("00012", "03", 5, 50.0, "Other", "AF"),
    ]

    matches, unmatched = match_transition_events(gt, pred)

    assert len(matches) == 1
    assert matches[0].latency_windows == 0
    assert matches[0].latency_seconds == pytest.approx(0.0)
    assert unmatched == []


def test_match_transition_events_matches_delayed_prediction() -> None:
    gt = [
        TransitionEvent("00012", "03", 5, 50.0, "N", "AF"),
    ]
    pred = [
        TransitionEvent("00012", "03", 7, 70.0, "N", "AF"),
    ]

    matches, unmatched = match_transition_events(gt, pred)

    assert len(matches) == 1
    assert matches[0].latency_windows == 2
    assert matches[0].latency_seconds == pytest.approx(20.0)
    assert unmatched == []


def test_match_transition_events_misses_prediction_after_next_gt_transition() -> None:
    gt = [
        TransitionEvent("00012", "03", 5, 50.0, "N", "AF"),
        TransitionEvent("00012", "03", 8, 80.0, "AF", "N"),
    ]
    pred = [
        TransitionEvent("00012", "03", 8, 80.0, "N", "AF"),
    ]

    matches, unmatched = match_transition_events(gt, pred)

    assert matches == []
    assert unmatched == pred


def test_match_transition_events_counts_unmatched_extra_prediction() -> None:
    gt = [
        TransitionEvent("00012", "03", 5, 50.0, "N", "AF"),
    ]
    pred = [
        TransitionEvent("00012", "03", 5, 50.0, "N", "AF"),
        TransitionEvent("00012", "03", 7, 70.0, "AF", "Other"),
    ]

    matches, unmatched = match_transition_events(gt, pred)

    assert len(matches) == 1
    assert unmatched == [pred[1]]


def test_extract_ground_truth_abnormal_episodes_closes_on_normal_and_unknown() -> None:
    records = [
        _record(window_index=0, truth_label="N", predicted_label="N"),
        _record(window_index=1, truth_label="AF", predicted_label="AF"),
        _record(window_index=2, truth_label="AF", predicted_label="AF"),
        _record(window_index=3, truth_label=UNKNOWN_RHYTHM_LABEL, predicted_label="AF"),
        _record(window_index=4, truth_label="Other", predicted_label="Other"),
        _record(window_index=5, truth_label="N", predicted_label="N"),
    ]

    episodes = extract_ground_truth_abnormal_episodes(records)

    assert [
        (episode.label, episode.start_window, episode.end_window)
        for episode in episodes
    ] == [("AF", 1, 2), ("Other", 4, 4)]


def test_match_intervention_events_counts_hit_duplicate_false_and_miss() -> None:
    episodes = [
        GroundTruthEpisode("00012", "03", 0, "AF", 2, 4, 20.0, 40.0),
        GroundTruthEpisode("00012", "03", 1, "AF", 8, 9, 80.0, 90.0),
    ]
    events = [
        InterventionEvent("00012", "03", 3, 30.0, "medium", ("rule",)),
        InterventionEvent("00012", "03", 4, 40.0, "medium", ("rule",)),
        InterventionEvent("00012", "03", 6, 60.0, "medium", ("rule",)),
    ]

    matches, false_alerts = match_intervention_events(episodes, events)

    assert len(matches) == 1
    assert matches[0].ground_truth_episode.episode_id == 0
    assert matches[0].latency_windows == 1
    assert len(matches[0].duplicate_events) == 1
    assert false_alerts == [events[2]]


def test_evaluate_transition_detection_aggregates_report_and_latency() -> None:
    windows = [
        _make_window(
            window_index=0,
            spans=[RhythmSpan(0, 2500, "N")],
        ),
        _make_window(
            window_index=1,
            spans=[RhythmSpan(0, 2500, "AFIB")],
        ),
        _make_window(
            window_index=2,
            spans=[RhythmSpan(0, 2500, "AFIB")],
        ),
        _make_window(
            window_index=3,
            spans=[RhythmSpan(0, 2500, "N")],
        ),
    ]
    pipeline = _FakePipeline(["N", "N", "AF", "N"])

    artifacts = evaluate_transition_detection(windows, pipeline, debounce_windows=0)  # type: ignore[arg-type]
    report = artifacts.report

    assert report.windows_processed == 4
    assert report.ground_truth_transition_total == 2
    assert report.predicted_transition_total == 2
    assert report.matched_transition_total == 2
    assert report.false_positive_total == 0
    assert report.false_negative_total == 0
    assert report.precision == pytest.approx(1.0)
    assert report.recall == pytest.approx(1.0)
    assert report.f1 == pytest.approx(1.0)
    assert report.latency_windows_mean == pytest.approx(0.5)
    assert report.latency_windows_median == pytest.approx(0.5)
    assert report.latency_windows_p90 == pytest.approx(0.9)
    assert report.latency_windows_max == 1
    assert report.latency_seconds_mean == pytest.approx(5.0)
    assert report.per_transition_pair["N->AF"]["ground_truth_total"] == 1
    assert report.per_transition_pair["AF->N"]["ground_truth_total"] == 1
    assert report.unknown_raw_label_counts == {}
    assert all(
        record.predicted_label in KNOWN_RHYTHM_LABELS
        for record in artifacts.window_records
    )


def test_evaluate_intervention_replay_scores_alerts_against_abnormal_episodes() -> None:
    windows = [
        _make_window(window_index=0, spans=[RhythmSpan(0, 2500, "N")]),
        _make_window(window_index=1, spans=[RhythmSpan(0, 2500, "AFIB")]),
        _make_window(window_index=2, spans=[RhythmSpan(0, 2500, "AFIB")]),
        _make_window(window_index=3, spans=[RhythmSpan(0, 2500, "N")]),
        _make_window(window_index=4, spans=[RhythmSpan(0, 2500, "N")]),
    ]
    pipeline = _FakeInterventionPipeline(
        predicted_labels=["N", "AF", "AF", "N", "N"],
        intervention_windows={2, 4},
    )

    artifacts = evaluate_intervention_replay(windows, pipeline)  # type: ignore[arg-type]
    report = artifacts.report

    assert report.windows_processed == 5
    assert report.ground_truth_episode_total == 1
    assert report.intervention_event_total == 2
    assert report.matched_episode_total == 1
    assert report.missed_episode_total == 0
    assert report.false_alert_total == 1
    assert report.duplicate_alert_total == 0
    assert report.precision == pytest.approx(0.5)
    assert report.recall == pytest.approx(1.0)
    assert report.latency_windows_mean == pytest.approx(1.0)
    assert report.latency_seconds_mean == pytest.approx(10.0)
    assert report.per_label["AF"]["matched_total"] == 1
    assert report.per_rule["test_rule"]["alert_total"] == 2
    assert report.per_rule["test_rule"]["matched_episode_total"] == 1
    assert report.per_rule["test_rule"]["false_alert_total"] == 1
    assert artifacts.false_alerts[0].window_index == 4
    assert artifacts.missed_episodes == []
    assert artifacts.patient_context is not None
    assert artifacts.patient_context["alert_summary"]["total_alerts"] == 2
    assert artifacts.patient_context["alert_summary"]["matched_alert_count"] == 1
    assert artifacts.patient_context["alert_summary"]["false_alert_count"] == 1
    assert artifacts.patient_context["recent_alerts"][0]["matched_episode_label"] == "AF"
    assert artifacts.patient_context["recent_alerts"][0]["latency_seconds"] == 10.0
    assert artifacts.patient_context["recent_alerts"][1]["is_false_alert"] is True


def test_evaluate_intervention_replay_exports_missed_episodes() -> None:
    windows = [
        _make_window(window_index=0, spans=[RhythmSpan(0, 2500, "N")]),
        _make_window(window_index=1, spans=[RhythmSpan(0, 2500, "AFIB")]),
        _make_window(window_index=2, spans=[RhythmSpan(0, 2500, "AFIB")]),
        _make_window(window_index=3, spans=[RhythmSpan(0, 2500, "N")]),
    ]
    pipeline = _FakeInterventionPipeline(
        predicted_labels=["N", "AF", "AF", "N"],
        intervention_windows=set(),
    )

    artifacts = evaluate_intervention_replay(windows, pipeline)  # type: ignore[arg-type]
    debug_types = [record["record_type"] for record in artifacts.iter_debug_records()]

    assert artifacts.report.missed_episode_total == 1
    assert len(artifacts.missed_episodes) == 1
    assert artifacts.missed_episodes[0].start_window == 1
    assert "missed_episode" in debug_types
