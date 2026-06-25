import pytest
import numpy as np

from agent.data.icentia11k_loader import BeatAnnotation, RhythmAnnotation, RawECGData
from agent.mhealth.state_builders.icentia11k_state_builder import (
    IcentiaWindowConfig,
    RhythmInterval,
    build_monitoring_state_from_icentia_window,
    build_monitoring_states_from_icentia_segment,
    build_monitoring_states_from_icentia_raw,
    compute_af_burden_ratio,
    compute_hr_bpm,
    compute_max_hr_bpm,
    compute_rhythm_transition_count_per_hour,
    compute_segment_aggregates,
    dominant_rhythm,
    map_icentia_rhythm,
    parse_rhythm_intervals_from_aux_notes,
)


def make_beats_for_hr(
    *,
    start_s: int,
    end_s: int,
    bpm: int,
    sampling_rate_hz: int,
) -> list[int]:
    """Create evenly spaced fake beat sample indices for a target HR."""

    interval_samples = int(sampling_rate_hz * 60 / bpm)

    if interval_samples <= 0:
        raise ValueError("interval_samples must be positive.")

    return list(
        range(
            start_s * sampling_rate_hz,
            end_s * sampling_rate_hz,
            interval_samples,
        )
    )


@pytest.fixture
def config() -> IcentiaWindowConfig:
    return IcentiaWindowConfig(
        sampling_rate_hz=10,
        current_window_sec=300,
        previous_window_sec=300,
        window_stride_sec=300,
        max_hr_subwindow_sec=30,
        rhythm_min_overlap_sec=2.0,
    )


@pytest.fixture
def synthetic_segment(config: IcentiaWindowConfig) -> dict:
    """900-second fake segment.

    Windows:
    - 0-300s: N, 60 bpm
    - 300-600s: AFIB, 120 bpm
    - 600-900s: N, 60 bpm

    Segment-level expected values:
    - AF burden = 1 / 3
    - max HR = 120 bpm
    - rhythm transitions = N -> AF -> N = 2 transitions
    - duration = 900s = 0.25h
    - transition rate = 8 transitions/hour
    """

    sr = config.sampling_rate_hz

    beat_samples = (
        make_beats_for_hr(
            start_s=0,
            end_s=300,
            bpm=60,
            sampling_rate_hz=sr,
        )
        + make_beats_for_hr(
            start_s=300,
            end_s=600,
            bpm=120,
            sampling_rate_hz=sr,
        )
        + make_beats_for_hr(
            start_s=600,
            end_s=900,
            bpm=60,
            sampling_rate_hz=sr,
        )
    )

    rhythm_intervals = [
        RhythmInterval(0 * sr, 300 * sr, "N"),
        RhythmInterval(300 * sr, 600 * sr, "AFIB"),
        RhythmInterval(600 * sr, 900 * sr, "N"),
    ]

    return {
        "patient_id": "patient_001",
        "segment_id": "segment_001",
        "beat_samples": beat_samples,
        "rhythm_intervals": rhythm_intervals,
        "segment_start_sample": 0,
        "segment_end_sample": 900 * sr,
    }


def test_map_icentia_rhythm_known_labels() -> None:
    assert map_icentia_rhythm("N") == "N"
    assert map_icentia_rhythm("(N") == "N"
    assert map_icentia_rhythm("AFIB") == "AF"
    assert map_icentia_rhythm("(AFIB") == "AF"
    assert map_icentia_rhythm("AF") == "AF"
    assert map_icentia_rhythm("AFL") == "AF"
    assert map_icentia_rhythm("(AFL") == "AF"


def test_map_icentia_rhythm_unknown_and_empty_labels_return_none() -> None:
    assert map_icentia_rhythm(None) is None
    assert map_icentia_rhythm("") is None
    assert map_icentia_rhythm("   ") is None
    assert map_icentia_rhythm("(") is None
    assert map_icentia_rhythm(")") is None

    # Important: templates.py does not include "Other" in the rhythm answer set.
    # Unknown Icentia labels should therefore be skipped instead of emitted.
    assert map_icentia_rhythm("XYZ") is None


def test_parse_rhythm_intervals_from_aux_notes_basic() -> None:
    intervals = parse_rhythm_intervals_from_aux_notes(
        annotation_samples=[0, 100, 100, 200, 200],
        aux_notes=["(N", ")", "(AFIB", ")", "(AFL"],
        segment_end_sample=300,
    )

    assert intervals == [
        RhythmInterval(start_sample=0, end_sample=100, label="N"),
        RhythmInterval(start_sample=100, end_sample=200, label="AFIB"),
        RhythmInterval(start_sample=200, end_sample=300, label="AFL"),
    ]


def test_parse_rhythm_intervals_uses_labelled_explicit_end_marker() -> None:
    intervals = parse_rhythm_intervals_from_aux_notes(
        annotation_samples=[12000, 18500, 60000],
        aux_notes=["(AFIB", "AFIB)", "(AFIB"],
        segment_end_sample=100000,
    )

    assert intervals == [
        RhythmInterval(start_sample=12000, end_sample=18500, label="AFIB"),
        RhythmInterval(start_sample=60000, end_sample=100000, label="AFIB"),
    ]


def test_parse_rhythm_intervals_closes_previous_when_new_starts() -> None:
    intervals = parse_rhythm_intervals_from_aux_notes(
        annotation_samples=[0, 100],
        aux_notes=["(N", "(AFIB"],
        segment_end_sample=200,
    )

    assert intervals == [
        RhythmInterval(start_sample=0, end_sample=100, label="N"),
        RhythmInterval(start_sample=100, end_sample=200, label="AFIB"),
    ]


def test_parse_rhythm_intervals_skips_empty_start_marker() -> None:
    intervals = parse_rhythm_intervals_from_aux_notes(
        annotation_samples=[0, 100],
        aux_notes=["(", "(N"],
        segment_end_sample=200,
    )

    assert intervals == [
        RhythmInterval(start_sample=100, end_sample=200, label="N"),
    ]


def test_parse_rhythm_intervals_requires_same_length_inputs() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        parse_rhythm_intervals_from_aux_notes(
            annotation_samples=[0, 100],
            aux_notes=["(N"],
            segment_end_sample=200,
        )


def test_compute_hr_bpm_returns_expected_average() -> None:
    sr = 10
    beats = make_beats_for_hr(
        start_s=0,
        end_s=300,
        bpm=60,
        sampling_rate_hz=sr,
    )

    assert compute_hr_bpm(
        beat_samples=beats,
        start_sample=0,
        end_sample=300 * sr,
        sampling_rate_hz=sr,
    ) == pytest.approx(60.0)


def test_compute_hr_bpm_zero_beats_returns_none() -> None:
    assert (
        compute_hr_bpm(
            beat_samples=[],
            start_sample=0,
            end_sample=300 * 10,
            sampling_rate_hz=10,
        )
        is None
    )


def test_compute_hr_bpm_invalid_window_returns_none() -> None:
    assert (
        compute_hr_bpm(
            beat_samples=[1, 2, 3],
            start_sample=100,
            end_sample=100,
            sampling_rate_hz=10,
        )
        is None
    )


def test_dominant_rhythm_uses_largest_overlap() -> None:
    intervals = [
        RhythmInterval(0, 100, "N"),
        RhythmInterval(100, 300, "AFIB"),
    ]

    assert (
        dominant_rhythm(
            intervals,
            0,
            300,
            sampling_rate_hz=10,
            min_overlap_sec=1.0,
        )
        == "AF"
    )


def test_dominant_rhythm_tie_breaks_towards_af() -> None:
    intervals = [
        RhythmInterval(0, 100, "N"),
        RhythmInterval(100, 200, "AFIB"),
    ]

    assert (
        dominant_rhythm(
            intervals,
            0,
            200,
            sampling_rate_hz=10,
            min_overlap_sec=1.0,
        )
        == "AF"
    )


def test_dominant_rhythm_collapses_afl_to_af() -> None:
    intervals = [
        RhythmInterval(0, 100, "N"),
        RhythmInterval(100, 300, "AFL"),
    ]

    assert (
        dominant_rhythm(
            intervals,
            0,
            300,
            sampling_rate_hz=10,
            min_overlap_sec=1.0,
        )
        == "AF"
    )


def test_dominant_rhythm_ignores_too_short_overlap() -> None:
    sr = 10
    intervals = [
        RhythmInterval(0, 10, "AFIB"),      # 1 second, should be ignored
        RhythmInterval(10, 300 * sr, "N"),
    ]

    assert (
        dominant_rhythm(
            intervals,
            0,
            300 * sr,
            sampling_rate_hz=sr,
            min_overlap_sec=2.0,
        )
        == "N"
    )


def test_dominant_rhythm_returns_none_when_only_unknown_labels() -> None:
    intervals = [
        RhythmInterval(0, 100, "XYZ"),
    ]

    assert (
        dominant_rhythm(
            intervals,
            0,
            100,
            sampling_rate_hz=10,
            min_overlap_sec=1.0,
        )
        is None
    )


def test_compute_af_burden_ratio() -> None:
    intervals = [
        RhythmInterval(0, 100, "N"),
        RhythmInterval(100, 300, "AFIB"),
        RhythmInterval(300, 400, "AFL"),
    ]

    assert compute_af_burden_ratio(
        intervals,
        0,
        400,
        sampling_rate_hz=10,
        min_overlap_sec=1.0,
    ) == pytest.approx(0.75)


def test_compute_rhythm_transition_count_collapses_afib_and_afl() -> None:
    intervals = [
        RhythmInterval(0, 100, "N"),
        RhythmInterval(100, 200, "AFIB"),
        RhythmInterval(200, 300, "AFL"),
        RhythmInterval(300, 400, "N"),
    ]

    assert compute_rhythm_transition_count_per_hour(
        intervals,
        0,
        400,
        sampling_rate_hz=10,
        min_overlap_sec=1.0,
    ) == pytest.approx(180.0)


def test_compute_af_burden_ratio_returns_none_without_valid_rhythm() -> None:
    intervals = [
        RhythmInterval(0, 100, "XYZ"),
    ]

    assert (
        compute_af_burden_ratio(
            intervals,
            0,
            100,
            sampling_rate_hz=10,
            min_overlap_sec=1.0,
        )
        is None
    )


def test_compute_max_hr_bpm_uses_subwindows() -> None:
    sr = 10

    beats = (
        make_beats_for_hr(
            start_s=0,
            end_s=30,
            bpm=60,
            sampling_rate_hz=sr,
        )
        + make_beats_for_hr(
            start_s=30,
            end_s=60,
            bpm=120,
            sampling_rate_hz=sr,
        )
    )

    assert compute_max_hr_bpm(
        beats,
        0,
        60 * sr,
        sampling_rate_hz=sr,
        subwindow_sec=30,
    ) == pytest.approx(120.0)


def test_compute_max_hr_bpm_skips_empty_subwindows() -> None:
    sr = 10

    beats = make_beats_for_hr(
        start_s=30,
        end_s=60,
        bpm=60,
        sampling_rate_hz=sr,
    )

    assert compute_max_hr_bpm(
        beats,
        0,
        60 * sr,
        sampling_rate_hz=sr,
        subwindow_sec=30,
    ) == pytest.approx(60.0)


def test_compute_max_hr_bpm_returns_none_when_no_valid_subwindow() -> None:
    assert (
        compute_max_hr_bpm(
            [],
            0,
            60 * 10,
            sampling_rate_hz=10,
            subwindow_sec=30,
        )
        is None
    )


def test_compute_rhythm_transition_count_per_hour_collapses_repeated_labels() -> None:
    sr = 1
    intervals = [
        RhythmInterval(0, 100, "N"),
        RhythmInterval(100, 200, "N"),
        RhythmInterval(200, 300, "AFIB"),
        RhythmInterval(300, 400, "AFIB"),
        RhythmInterval(400, 3600, "N"),
    ]

    assert compute_rhythm_transition_count_per_hour(
        intervals,
        0,
        3600,
        sampling_rate_hz=sr,
        min_overlap_sec=1.0,
    ) == pytest.approx(2.0)


def test_compute_segment_aggregates(
    config: IcentiaWindowConfig,
    synthetic_segment: dict,
) -> None:
    aggregates = compute_segment_aggregates(
        beat_samples=synthetic_segment["beat_samples"],
        rhythm_intervals=synthetic_segment["rhythm_intervals"],
        segment_start_sample=synthetic_segment["segment_start_sample"],
        segment_end_sample=synthetic_segment["segment_end_sample"],
        config=config,
    )

    assert aggregates.af_burden_ratio == pytest.approx(1 / 3)
    assert aggregates.max_hr_bpm == pytest.approx(120.0)
    assert aggregates.rhythm_transition_count_per_hour == pytest.approx(8.0)


def test_build_first_window_state_has_no_previous_context(
    config: IcentiaWindowConfig,
    synthetic_segment: dict,
) -> None:
    state = build_monitoring_state_from_icentia_window(
        patient_id=synthetic_segment["patient_id"],
        segment_id=synthetic_segment["segment_id"],
        window_index=0,
        beat_samples=synthetic_segment["beat_samples"],
        rhythm_intervals=synthetic_segment["rhythm_intervals"],
        current_start_sample=0,
        segment_start_sample=synthetic_segment["segment_start_sample"],
        segment_end_sample=synthetic_segment["segment_end_sample"],
        config=config,
    )

    assert state.state_id == "icentia11k_patient_patient_001_segment_segment_001_window_0"
    assert state.patient_id == "patient_001"
    assert state.dataset == "icentia11k"
    assert state.modality == "ecg"

    assert state.window_index == 0
    assert state.window_start_s == pytest.approx(0.0)
    assert state.window_duration_s == pytest.approx(300.0)

    assert state.hr_bpm == pytest.approx(60.0)
    assert state.rhythm_class == "N"

    assert state.previous_hr_bpm is None
    assert state.previous_rhythm_class is None

    assert state.af_burden_ratio == pytest.approx(1 / 3)
    assert state.max_hr_bpm == pytest.approx(120.0)
    assert state.rhythm_transition_count_per_hour == pytest.approx(8.0)

    assert state.metadata["segment_id"] == "segment_001"
    assert state.metadata["sampling_rate_hz"] == 10
    assert state.metadata["previous_start_sample"] is None
    assert state.metadata["rhythm_class_source"] == "wfdb_aux_note_dominant_overlap"
    assert (
        state.metadata["rhythm_label_mapping_version"]
        == "icentia11k_v1_afl_collapsed_to_af"
    )
    assert state.metadata["aggregate_scope"] == "monitoring_window_level"
    assert state.metadata["aggregate_scope_samples"] == 9000
    assert state.metadata["aggregate_scope_seconds"] == pytest.approx(900.0)


def test_build_second_window_state_has_previous_context(
    config: IcentiaWindowConfig,
    synthetic_segment: dict,
) -> None:
    sr = config.sampling_rate_hz

    state = build_monitoring_state_from_icentia_window(
        patient_id=synthetic_segment["patient_id"],
        segment_id=synthetic_segment["segment_id"],
        window_index=1,
        beat_samples=synthetic_segment["beat_samples"],
        rhythm_intervals=synthetic_segment["rhythm_intervals"],
        current_start_sample=300 * sr,
        segment_start_sample=synthetic_segment["segment_start_sample"],
        segment_end_sample=synthetic_segment["segment_end_sample"],
        config=config,
    )

    assert state.window_index == 1
    assert state.window_start_s == pytest.approx(300.0)
    assert state.window_duration_s == pytest.approx(300.0)

    assert state.hr_bpm == pytest.approx(120.0)
    assert state.rhythm_class == "AF"

    assert state.previous_hr_bpm == pytest.approx(60.0)
    assert state.previous_rhythm_class == "N"

    assert state.metadata["previous_start_sample"] == 0
    assert state.metadata["current_start_sample"] == 300 * sr
    assert state.metadata["current_end_sample"] == 600 * sr


def test_build_monitoring_states_from_icentia_segment(
    config: IcentiaWindowConfig,
    synthetic_segment: dict,
) -> None:
    states = build_monitoring_states_from_icentia_segment(
        patient_id=synthetic_segment["patient_id"],
        segment_id=synthetic_segment["segment_id"],
        beat_samples=synthetic_segment["beat_samples"],
        rhythm_intervals=synthetic_segment["rhythm_intervals"],
        segment_start_sample=synthetic_segment["segment_start_sample"],
        segment_end_sample=synthetic_segment["segment_end_sample"],
        config=config,
    )

    assert len(states) == 3

    assert [state.window_index for state in states] == [0, 1, 2]
    assert [state.window_start_s for state in states] == [0.0, 300.0, 600.0]
    assert [state.rhythm_class for state in states] == ["N", "AF", "N"]
    assert [state.previous_rhythm_class for state in states] == [None, "N", "AF"]

    assert [state.hr_bpm for state in states] == pytest.approx([60.0, 120.0, 60.0])
    assert states[0].previous_hr_bpm is None
    assert states[1].previous_hr_bpm == pytest.approx(60.0)
    assert states[2].previous_hr_bpm == pytest.approx(120.0)


def test_segment_aggregates_are_identical_across_states(
    config: IcentiaWindowConfig,
    synthetic_segment: dict,
) -> None:
    states = build_monitoring_states_from_icentia_segment(
        patient_id=synthetic_segment["patient_id"],
        segment_id=synthetic_segment["segment_id"],
        beat_samples=synthetic_segment["beat_samples"],
        rhythm_intervals=synthetic_segment["rhythm_intervals"],
        segment_start_sample=synthetic_segment["segment_start_sample"],
        segment_end_sample=synthetic_segment["segment_end_sample"],
        config=config,
    )

    for state in states:
        assert state.af_burden_ratio == pytest.approx(1 / 3)
        assert state.max_hr_bpm == pytest.approx(120.0)
        assert state.rhythm_transition_count_per_hour == pytest.approx(8.0)


def test_build_window_rejects_current_start_before_segment(
    config: IcentiaWindowConfig,
    synthetic_segment: dict,
) -> None:
    with pytest.raises(ValueError, match="before segment_start_sample"):
        build_monitoring_state_from_icentia_window(
            patient_id=synthetic_segment["patient_id"],
            segment_id=synthetic_segment["segment_id"],
            window_index=0,
            beat_samples=synthetic_segment["beat_samples"],
            rhythm_intervals=synthetic_segment["rhythm_intervals"],
            current_start_sample=-1,
            segment_start_sample=synthetic_segment["segment_start_sample"],
            segment_end_sample=synthetic_segment["segment_end_sample"],
            config=config,
        )


def test_build_window_rejects_truncated_current_window(
    config: IcentiaWindowConfig,
    synthetic_segment: dict,
) -> None:
    sr = config.sampling_rate_hz

    with pytest.raises(ValueError, match="does not fit inside the segment"):
        build_monitoring_state_from_icentia_window(
            patient_id=synthetic_segment["patient_id"],
            segment_id=synthetic_segment["segment_id"],
            window_index=99,
            beat_samples=synthetic_segment["beat_samples"],
            rhythm_intervals=synthetic_segment["rhythm_intervals"],
            current_start_sample=800 * sr,
            segment_start_sample=synthetic_segment["segment_start_sample"],
            segment_end_sample=synthetic_segment["segment_end_sample"],
            config=config,
        )


def test_build_segment_rejects_invalid_stride(
    config: IcentiaWindowConfig,
    synthetic_segment: dict,
) -> None:
    bad_config = IcentiaWindowConfig(
        sampling_rate_hz=config.sampling_rate_hz,
        current_window_sec=config.current_window_sec,
        previous_window_sec=config.previous_window_sec,
        window_stride_sec=0,
        max_hr_subwindow_sec=config.max_hr_subwindow_sec,
        rhythm_min_overlap_sec=config.rhythm_min_overlap_sec,
    )

    with pytest.raises(ValueError, match="window_stride_sec must be positive"):
        build_monitoring_states_from_icentia_segment(
            patient_id=synthetic_segment["patient_id"],
            segment_id=synthetic_segment["segment_id"],
            beat_samples=synthetic_segment["beat_samples"],
            rhythm_intervals=synthetic_segment["rhythm_intervals"],
            segment_start_sample=synthetic_segment["segment_start_sample"],
            segment_end_sample=synthetic_segment["segment_end_sample"],
            config=bad_config,
        )

def test_build_monitoring_states_from_icentia_raw() -> None:
    config = IcentiaWindowConfig(
        sampling_rate_hz=10,
        current_window_sec=300,
        previous_window_sec=300,
        window_stride_sec=300,
        max_hr_subwindow_sec=30,
        rhythm_min_overlap_sec=2.0,
    )

    sr = config.sampling_rate_hz

    beats = (
        list(range(0, 300 * sr, sr))        # 60 bpm
        + list(range(300 * sr, 600 * sr, sr // 2))  # 120 bpm
        + list(range(600 * sr, 900 * sr, sr))       # 60 bpm
    )

    raw = RawECGData(
        signal=np.zeros(900 * sr),
        fs=sr,
        patient_id="00012",
        segment_id="03",
        n_samples=900 * sr,
        sample_start=0,
        sample_end=900 * sr,
        segment_total_samples=900 * sr,
        beat_annotations=BeatAnnotation(
            samples=np.asarray(beats, dtype=int),
            symbols=["N"] * len(beats),
        ),
        rhythm_annotations=RhythmAnnotation(
            start_samples=[0, 300 * sr, 600 * sr],
            end_samples=[300 * sr, 600 * sr, 900 * sr],
            rhythm_types=["N", "AFIB", "N"],
        ),
        metadata={"source": "unit_test"},
    )

    states = build_monitoring_states_from_icentia_raw(raw, config=config)

    assert len(states) == 3
    assert [state.rhythm_class for state in states] == ["N", "AF", "N"]
    assert states[0].previous_rhythm_class is None
    assert states[1].previous_rhythm_class == "N"
    assert states[2].previous_rhythm_class == "AF"

    assert states[0].hr_bpm == pytest.approx(60.0)
    assert states[1].hr_bpm == pytest.approx(120.0)
    assert states[2].hr_bpm == pytest.approx(60.0)

    for state in states:
        assert state.state_id.startswith(
            "icentia11k_patient_00012_segment_03_samples_0_9000_window_"
        )
        assert state.dataset == "icentia11k"
        assert state.modality == "ecg"
        assert state.af_burden_ratio == pytest.approx(1 / 3)
        assert state.max_hr_bpm == pytest.approx(120.0)
        assert state.rhythm_transition_count_per_hour == pytest.approx(8.0)
        assert state.metadata["aggregate_scope"] == "monitoring_window_level"
        assert state.metadata["aggregate_scope_samples"] == 900 * sr
        assert state.metadata["aggregate_scope_seconds"] == pytest.approx(900.0)
        assert state.metadata["source_sample_start"] == 0
        assert state.metadata["source_sample_end"] == 900 * sr
        assert state.metadata["source_segment_total_samples"] == 900 * sr


def test_build_monitoring_states_from_icentia_raw_aligns_config_sampling_rate() -> None:
    config = IcentiaWindowConfig(
        sampling_rate_hz=250,
        current_window_sec=10,
        previous_window_sec=10,
        window_stride_sec=10,
        max_hr_subwindow_sec=5,
        rhythm_min_overlap_sec=1.0,
    )

    raw = RawECGData(
        signal=np.zeros(200),
        fs=20,
        patient_id="00012",
        segment_id="03",
        n_samples=200,
        sample_start=1000,
        sample_end=1200,
        segment_total_samples=5000,
        beat_annotations=BeatAnnotation(
            samples=np.arange(0, 200, 20, dtype=int),
            symbols=["N"] * 10,
        ),
        rhythm_annotations=RhythmAnnotation(
            start_samples=[0],
            end_samples=[200],
            rhythm_types=["N"],
        ),
        metadata={"source": "unit_test"},
    )

    states = build_monitoring_states_from_icentia_raw(raw, config=config)

    assert len(states) == 1
    assert states[0].metadata["sampling_rate_hz"] == 20
    assert states[0].window_duration_s == pytest.approx(10.0)
    assert states[0].metadata["aggregate_scope_samples"] == 200
    assert states[0].metadata["aggregate_scope_seconds"] == pytest.approx(10.0)
    assert (
        states[0].state_id
        == "icentia11k_patient_00012_segment_03_samples_1000_1200_window_0"
    )

