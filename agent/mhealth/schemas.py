"""
Core schemas for mHealth monitoring state timelines.

This module intentionally contains no benchmark or evaluation types; it is used
by runtime state stores, tools, and dataset builders.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DatasetName = Literal[
    "icentia11k",
    "afppgecg",
    "wesad",
    "ppg_dalia",
]

AFPPGECG_DATASET = "afppgecg"

_DATASET_ALIASES = {
    "af_ppg_ecg": AFPPGECG_DATASET,
    "zenodo": AFPPGECG_DATASET,
    "zenodo_af_monitoring": AFPPGECG_DATASET,
}


def canonicalize_dataset_name(value: str) -> str:
    """Return the canonical identifier while accepting legacy persisted names."""
    normalized = str(value).strip().lower()
    return _DATASET_ALIASES.get(normalized, normalized)


def is_afppgecg_dataset(value: str | None) -> bool:
    """Return whether a dataset identifier refers to AFPPGECG."""
    return value is not None and canonicalize_dataset_name(value) == AFPPGECG_DATASET

ModalityName = Literal[
    "ecg",
    "ppg",
    "wearable",
]

logger = logging.getLogger(__name__)

RhythmClass = Literal[
    "N",
    "AF",
    "AFL",
]

StressClass = Literal[
    "baseline",
    "stress",
    "amusement",
    "meditation",
]

@dataclass
class MonitoringState:
    """
    Structured state used to generate VitalBench samples.

    This state can come from:
    - Proactive pipeline output.
    - ECG/PPG window processing.
    - Aggregated monitoring-window statistics.
    - Demo / synthetic states for early testing.

    Important convention:
    - None means the field is missing or not usable for GT generation.
    - "Unknown" should not be treated as a valid rhythm class.
      If an upstream pipeline outputs "Unknown", from_dict() converts it to None.
      This ensures templates requiring rhythm_class are skipped rather than
      generating ambiguous ground truth.
    """

    # ------------------------------------------------------------------
    # Identity / metadata
    # ------------------------------------------------------------------

    state_id: str
    patient_id: str
    dataset: DatasetName
    modality: ModalityName
    subject_id: str | None = None
    recording_id: str | None = None

    # Window-level position
    window_index: int | None = None
    window_start_s: float | None = None
    window_end_s: float | None = None
    window_duration_s: float | None = None

    # ------------------------------------------------------------------
    # Fields actively used by current benchmark templates
    # ------------------------------------------------------------------

    # Current-window fields
    hr_bpm: float | None = None
    rhythm_class: RhythmClass | None = None

    # Stress / affect protocol fields
    stress_label: StressClass | None = None
    protocol_label_id: int | None = None

    # Activity-context fields. For PPG-DaLiA these are protocol activity
    # labels, not stress, sleep, or clinical diagnosis annotations.
    activity_label: str | None = None

    # Previous-window comparison fields
    previous_hr_bpm: float | None = None
    previous_rhythm_class: RhythmClass | None = None
    previous_stress_label: StressClass | None = None
    previous_protocol_label_id: int | None = None
    previous_activity_label: str | None = None

    dataset_specific: dict[str, Any] = field(default_factory=dict)

    # Monitoring-window aggregate fields
    af_burden_ratio: float | None = None
    # Fraction of the aggregate monitoring window labelled as AF, e.g. 0.12 = 12%.
    # This value is expected to be computed upstream. For episode-level
    # aggregation, upstream preprocessing should exclude AF episodes shorter
    # than the configured minimum duration, e.g. 30 seconds.

    stress_burden_ratio: float | None = None
    # Fraction of the aggregate monitoring window labelled as WESAD stress.

    stress_duration_s: float | None = None
    dominant_stress_label: StressClass | None = None

    max_hr_bpm: float | None = None

    mean_hr_bpm: float | None = None
    ecg_reference_hr_bpm: float | None = None

    rhythm_transition_count_per_hour: float | None = None
    stress_transition_count_per_hour: float | None = None

    # ------------------------------------------------------------------
    # Reserved fields for future templates / proactive integration
    # ------------------------------------------------------------------
    # These fields are not required by the current main benchmark templates,
    # but are kept here to support future VitalBench tasks such as sustained
    # tachycardia, AF episode explanation, HRV/PRV questions, signal-quality
    # checks, baseline comparison, and proactive alert explanation.

    # Short-term / rolling-window fields
    mean_hr_5min: float | None = None
    tachycardia_ratio_5min: float | None = None

    # Rhythm / AF episode fields
    af_episode_duration_s: float | None = None

    # HRV / PRV fields
    sdnn_ms: float | None = None
    rmssd_ms: float | None = None

    # Signal quality fields
    signal_quality_score: float | None = None
    motion_level: str | None = None
    # Expected examples:
    # - "low"
    # - "medium"
    # - "high"

    # Baseline fields
    baseline_resting_hr: float | None = None
    hr_deviation_from_baseline: float | None = None

    # Additional aggregate fields
    rhythm_transition_count: int | None = None
    stress_transition_count: int | None = None
    recording_duration_s: float | None = None

    # Alert-related fields
    alert_triggered: bool = False
    alert_rule: str | None = None
    alert_reason: str | None = None
    urgency: str | None = None
    # Expected examples:
    # - "none"
    # - "low"
    # - "medium"
    # - "high"
    # - "critical"

    # ------------------------------------------------------------------
    # Extra metadata
    # ------------------------------------------------------------------

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset = canonicalize_dataset_name(self.dataset)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MonitoringState":
        """
        Construct MonitoringState from a dict.

        Unknown fields are ignored so this can safely consume richer
        proactive pipeline outputs.

        Rhythm convention:
        - Upstream "Unknown" is normalized to None.
        - None means missing / unusable for automatic GT generation.
        """
        valid_fields = set(cls.__dataclass_fields__.keys())
        unknown_fields = set(payload) - valid_fields
        if unknown_fields:
            logger.warning(
                "Dropping unknown MonitoringState fields during from_dict: %s",
                sorted(unknown_fields),
            )
        filtered = {k: v for k, v in payload.items() if k in valid_fields}
        if "dataset" in filtered:
            filtered["dataset"] = canonicalize_dataset_name(filtered["dataset"])

        for key in ("rhythm_class", "previous_rhythm_class"):
            value = filtered.get(key)
            if isinstance(value, str) and value.strip().lower() == "unknown":
                filtered[key] = None

        return cls(**filtered)

    def get(self, field_name: str, default: Any = None) -> Any:
        """Dictionary-like field access."""
        return getattr(self, field_name, default)

    def has_required_fields(self, required_fields: tuple[str, ...]) -> bool:
        """
        Return True if all required fields are present and non-null.

        Note:
        - False and 0 are valid values.
        - Only None means missing.
        - Upstream "Unknown" rhythm values should already be converted to None.
        """
        for field_name in required_fields:
            if not hasattr(self, field_name):
                return False
            if getattr(self, field_name) is None:
                return False
        return True

