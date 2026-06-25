"""
MonitoringState fixtures for VitalBench generator tests.

These states are intentionally small and hand-crafted. They are used to
smoke-test the template + schema + state-to-QA generator pipeline before
connecting real Icentia11k or AF-PPG-ECG preprocessing.

The goal is coverage, not realistic cohort statistics.
"""

from __future__ import annotations

from agent.mhealth.schemas import MonitoringState


DEMO_STATES: tuple[MonitoringState, ...] = (
    # ------------------------------------------------------------------
    # 1. Normal current-window ECG state
    # ------------------------------------------------------------------
    MonitoringState(
        state_id="demo_ecg_normal_0001",
        patient_id="demo_ecg_001",
        dataset="afppgecg",
        modality="ecg",
        window_index=1,
        window_start_s=0.0,
        window_duration_s=10.0,
        hr_bpm=78.0,
        previous_hr_bpm=76.0,
        rhythm_class="N",
        previous_rhythm_class="N",
        af_burden_ratio=0.0,
        max_hr_bpm=92.0,
        rhythm_transition_count_per_hour=0.0,
        recording_duration_s=3600.0,
        metadata={
            "description": "Normal ECG state with stable rhythm and normal HR.",
        },
    ),

    # ------------------------------------------------------------------
    # 2. Current AF ECG state
    # ------------------------------------------------------------------
    MonitoringState(
        state_id="demo_ecg_af_0002",
        patient_id="demo_ecg_001",
        dataset="afppgecg",
        modality="ecg",
        window_index=2,
        window_start_s=10.0,
        window_duration_s=10.0,
        hr_bpm=112.0,
        previous_hr_bpm=78.0,
        rhythm_class="AF",
        previous_rhythm_class="N",
        af_burden_ratio=0.18,
        max_hr_bpm=128.0,
        rhythm_transition_count_per_hour=2.0,
        recording_duration_s=3600.0,
        metadata={
            "description": "ECG state with current AF and elevated HR.",
        },
    ),

    # ------------------------------------------------------------------
    # 3. PPG normal/no-AF state
    # ------------------------------------------------------------------
    MonitoringState(
        state_id="demo_ppg_no_af_0003",
        patient_id="demo_ppg_001",
        dataset="afppgecg",
        modality="ppg",
        window_index=1,
        window_start_s=0.0,
        window_duration_s=10.0,
        hr_bpm=84.0,
        previous_hr_bpm=82.0,
        rhythm_class="N",
        previous_rhythm_class="N",
        af_burden_ratio=0.0,
        max_hr_bpm=98.0,
        rhythm_transition_count_per_hour=0.0,
        signal_quality_score=0.88,
        recording_duration_s=3600.0,
        metadata={
            "description": "Normal PPG state with no AF-like rhythm.",
        },
    ),

    # ------------------------------------------------------------------
    # 4. PPG AF-like state with high HR and recording-level AF burden
    # ------------------------------------------------------------------
    MonitoringState(
        state_id="demo_ppg_af_0004",
        patient_id="demo_ppg_001",
        dataset="afppgecg",
        modality="ppg",
        window_index=2,
        window_start_s=10.0,
        window_duration_s=10.0,
        hr_bpm=118.0,
        previous_hr_bpm=84.0,
        rhythm_class="AF",
        previous_rhythm_class="N",
        af_burden_ratio=0.32,
        max_hr_bpm=136.0,
        rhythm_transition_count_per_hour=4.5,
        signal_quality_score=0.81,
        recording_duration_s=3600.0,
        metadata={
            "description": (
                "PPG state with AF-like rhythm. AF on PPG is an indirect "
                "rhythm-irregularity inference."
            ),
        },
    ),

    # ------------------------------------------------------------------
    # 5. HR increase without rhythm change
    # ------------------------------------------------------------------
    MonitoringState(
        state_id="demo_ppg_hr_increase_0005",
        patient_id="demo_ppg_002",
        dataset="afppgecg",
        modality="ppg",
        window_index=5,
        window_start_s=40.0,
        window_duration_s=10.0,
        hr_bpm=104.0,
        previous_hr_bpm=94.0,
        rhythm_class="N",
        previous_rhythm_class="N",
        af_burden_ratio=0.0,
        max_hr_bpm=112.0,
        rhythm_transition_count_per_hour=0.0,
        signal_quality_score=0.76,
        recording_duration_s=1800.0,
        metadata={
            "description": "PPG state where HR increased by more than min_delta_bpm.",
        },
    ),

    # ------------------------------------------------------------------
    # 6. Low HR state
    # ------------------------------------------------------------------
    MonitoringState(
        state_id="demo_ecg_low_hr_0006",
        patient_id="demo_ecg_002",
        dataset="afppgecg",
        modality="ecg",
        window_index=6,
        window_start_s=50.0,
        window_duration_s=10.0,
        hr_bpm=52.0,
        previous_hr_bpm=54.0,
        rhythm_class="N",
        previous_rhythm_class="N",
        af_burden_ratio=0.0,
        max_hr_bpm=88.0,
        rhythm_transition_count_per_hour=0.0,
        recording_duration_s=3600.0,
        metadata={
            "description": "ECG state with low current HR.",
        },
    ),

    # ------------------------------------------------------------------
    # 7. Monitoring-window frequent rhythm changes
    # ------------------------------------------------------------------
    MonitoringState(
        state_id="demo_ecg_frequent_transitions_0007",
        patient_id="demo_ecg_003",
        dataset="afppgecg",
        modality="ecg",
        window_index=20,
        window_start_s=190.0,
        window_duration_s=10.0,
        hr_bpm=96.0,
        previous_hr_bpm=98.0,
        rhythm_class="AFL",
        previous_rhythm_class="AF",
        af_burden_ratio=0.08,
        max_hr_bpm=122.0,
        rhythm_transition_count=9,
        rhythm_transition_count_per_hour=6.0,
        recording_duration_s=5400.0,
        metadata={
            "description": "ECG state with frequent rhythm transitions.",
        },
    ),

    # ------------------------------------------------------------------
    # 8. Unknown rhythm input should be normalized to None and skipped
    # ------------------------------------------------------------------
    MonitoringState.from_dict(
        {
            "state_id": "demo_ppg_unknown_rhythm_0008",
            "patient_id": "demo_ppg_003",
            "dataset": "afppgecg",
            "modality": "ppg",
            "window_index": 8,
            "window_start_s": 70.0,
            "window_duration_s": 10.0,
            "hr_bpm": 89.0,
            "previous_hr_bpm": 90.0,
            "rhythm_class": "Unknown",
            "previous_rhythm_class": "Unknown",
            "af_burden_ratio": 0.0,
            "max_hr_bpm": 100.0,
            "rhythm_transition_count_per_hour": 0.0,
            "signal_quality_score": 0.42,
            "recording_duration_s": 1200.0,
            "metadata": {
                "description": (
                    "Unknown rhythm state. from_dict should convert Unknown "
                    "to None, causing rhythm-dependent templates to skip."
                ),
            },
        }
    ),
)


def get_demo_states() -> tuple[MonitoringState, ...]:
    """Return all demo states."""
    return DEMO_STATES

