from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent.encoder.ecg_signal_processor import ECGSignalProcessor, VitalParams
from agent.tools import proactive_evaluate_tools as tool_mod
from agent.tools.registry import list_tools


def _patch_signal(
    monkeypatch,
    *,
    duration_s: float,
    fs_hz: float = 250.0,
) -> None:
    signal = np.zeros(int(duration_s * fs_hz), dtype=float)
    monkeypatch.setattr(
        tool_mod,
        "_load_ecg_signal_window",
        lambda **kwargs: (
            signal,
            fs_hz,
            {
                "segment_id": kwargs.get("segment_id") or kwargs.get("recording_id"),
                "recording_id": kwargs.get("recording_id") or kwargs.get("segment_id"),
                "source_path": "synthetic",
            },
        ),
    )


def _patch_afppgecg_loader(
    monkeypatch,
    *,
    duration_s: float,
    fs_hz: float = 100.0,
    qrs_index: np.ndarray | None = None,
) -> list[dict[str, object]]:
    from agent.data.ppg_loader import PPGLoader

    calls: list[dict[str, object]] = []
    total_samples = int(max(duration_s, 120.0) * fs_hz)
    qrs = (
        np.asarray(qrs_index, dtype=int)
        if qrs_index is not None
        else np.arange(0, total_samples, int(fs_hz), dtype=int)
    )

    def fake_read_ecg_record(
        self,
        patient_id,
        *,
        include_signal=False,
        sampfrom=0,
        sampto=None,
    ):
        del self
        calls.append(
            {
                "patient_id": patient_id,
                "include_signal": include_signal,
                "sampfrom": sampfrom,
                "sampto": sampto,
            }
        )
        stop = total_samples if sampto is None else min(int(sampto), total_samples)
        signal = None
        if include_signal:
            signal = np.zeros(max(0, stop - int(sampfrom)), dtype=float)
        return SimpleNamespace(
            patient_id=str(patient_id).zfill(3),
            fs=int(fs_hz),
            n_samples=total_samples,
            qrs_index=qrs,
            rr_seconds=np.diff(qrs) / float(fs_hz) if qrs.size >= 2 else np.asarray([]),
            signal=signal,
            metadata={"record_path": "synthetic/001_ECG.mat"},
        )

    monkeypatch.setattr(PPGLoader, "read_ecg_record", fake_read_ecg_record)
    return calls


def test_evaluate_proactive_rules_registered() -> None:
    names = {item["name"] for item in list_tools()}

    assert "evaluate_proactive_rules" in names


def test_evaluate_proactive_rules_fires_extreme_tachycardia(monkeypatch) -> None:
    _patch_signal(monkeypatch, duration_s=30.0)
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(
            hr_bpm=210.0,
            rr_intervals_ms=[285.0] * 30,
            signal_quality_score=0.95,
        ),
    )

    result = tool_mod.evaluate_proactive_rules(
        dataset="icentia11k",
        patient_id="00012",
        recording_id="00",
        window_start_s=0.0,
        window_end_s=30.0,
    )

    assert result["success"] is True
    outcome = result["data"]["rule_outcome"]
    assert outcome["intervene"] is True
    assert outcome["urgency"] == "critical"
    assert outcome["triggered_rules"] == ["extreme_tachycardia"]
    assert outcome["first_trigger_offset_s"] == {"extreme_tachycardia": 0.0}
    assert result["data"]["trace"]["n_sub_windows"] == 3
    assert result["data"]["trace"]["hr_bpm_series"] == [210.0, 210.0, 210.0]
    assert result["data"]["trace"]["per_window_triggered_rules"][0] == [
        "extreme_tachycardia"
    ]
    assert result["data"]["rule_descriptions"][0]["rule_id"] == "extreme_tachycardia"
    summary = result["data"]["rhythm_summary"]
    assert summary["dominant_class"] in {"N", "AF", "Other"}
    assert set(summary["label_counts"]) == {"N", "AF", "Other", "Unknown"}
    assert set(summary["label_fractions"]) == {"N", "AF", "Other", "Unknown"}
    assert set(summary["max_consecutive_runs"]) == {"N", "AF", "Other", "Unknown"}
    assert set(summary["max_consecutive_seconds"]) == {"N", "AF", "Other", "Unknown"}
    assert summary["stability_label"] in {
        "stable",
        "occasional_changes",
        "frequent_changes",
        "unknown",
    }
    assert isinstance(summary["sustained_episode_classes"], list)
    assert isinstance(summary["sustained_af_episode_detected"], bool)


def test_rhythm_summary_classifies_isolated_af_flicker_as_unsustained(monkeypatch) -> None:
    _patch_signal(monkeypatch, duration_s=300.0)
    labels = [
        "AF",
        *["N"] * 6,
        "Other",
        "AF",
        *["N"] * 5,
        "Other",
        "AF",
        *["N"] * 5,
        "Other",
        "AF",
        *["N"] * 5,
        "Other",
        "Other",
    ]
    assert len(labels) == 30
    iterator = iter(labels)
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(
            hr_bpm=80.0,
            rr_intervals_ms=[750.0] * 10,
            signal_quality_score=0.95,
        ),
    )
    monkeypatch.setattr(tool_mod, "classify_rhythm", lambda rr: next(iterator))

    result = tool_mod.evaluate_proactive_rules(
        dataset="icentia11k",
        patient_id="00012",
        recording_id="00",
        window_start_s=0.0,
        window_end_s=300.0,
    )

    summary = result["data"]["rhythm_summary"]
    assert summary["label_counts"] == {"N": 21, "AF": 4, "Other": 5, "Unknown": 0}
    assert summary["sustained_af_episode_detected"] is False
    assert summary["sustained_n_episode_detected"] is True
    assert summary["dominant_class"] == "N"
    assert summary["dominant_fraction_of_valid"] == 0.7
    assert summary["sustained_episode_classes"] == ["N"]
    assert summary["stability_label"] == "stable"


def test_rhythm_summary_classifies_continuous_af_episode_as_sustained(monkeypatch) -> None:
    _patch_signal(monkeypatch, duration_s=300.0)
    labels = [
        "AF",
        "AF",
        "AF",
        "Other",
        *["AF", "Other"] * 13,
    ]
    assert len(labels) == 30
    iterator = iter(labels)
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(
            hr_bpm=90.0,
            rr_intervals_ms=[667.0] * 10,
            signal_quality_score=0.95,
        ),
    )
    monkeypatch.setattr(tool_mod, "classify_rhythm", lambda rr: next(iterator))

    result = tool_mod.evaluate_proactive_rules(
        dataset="icentia11k",
        patient_id="00012",
        recording_id="00",
        window_start_s=0.0,
        window_end_s=300.0,
    )

    summary = result["data"]["rhythm_summary"]
    assert summary["sustained_af_episode_detected"] is True
    assert summary["sustained_n_episode_detected"] is False
    assert summary["dominant_class"] == "AF"
    assert summary["max_consecutive_runs"]["AF"] >= 3
    assert summary["sustained_episode_classes"] == ["AF"]
    assert summary["stability_label"] == "stable"


def test_rhythm_summary_treats_af_other_flicker_as_stable_irregular_family(monkeypatch) -> None:
    _patch_signal(monkeypatch, duration_s=300.0)
    labels = [
        "AF",
        "AF",
        "AF",
        "Other",
        "Other",
        "Other",
        *["AF", "Other"] * 12,
    ]
    assert len(labels) == 30
    iterator = iter(labels)
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(
            hr_bpm=90.0,
            rr_intervals_ms=[667.0] * 10,
            signal_quality_score=0.95,
        ),
    )
    monkeypatch.setattr(tool_mod, "classify_rhythm", lambda rr: next(iterator))

    result = tool_mod.evaluate_proactive_rules(
        dataset="icentia11k",
        patient_id="00012",
        recording_id="00",
        window_start_s=0.0,
        window_end_s=300.0,
    )

    summary = result["data"]["rhythm_summary"]
    assert summary["sustained_af_episode_detected"] is True
    assert summary["sustained_other_episode_detected"] is True
    assert summary["sustained_n_episode_detected"] is False
    assert summary["sustained_episode_classes"] == ["AF", "Other"]
    assert summary["stability_label"] == "stable"


def test_evaluate_proactive_rules_rejects_range_below_10s() -> None:
    result = tool_mod.evaluate_proactive_rules(
        dataset="icentia11k",
        patient_id="00012",
        recording_id="00",
        window_start_s=0.0,
        window_end_s=9.0,
    )

    assert result["success"] is False
    assert "at least 10 seconds" in result["error"]


def test_evaluate_proactive_rules_clamps_range_above_600s(monkeypatch) -> None:
    _patch_signal(monkeypatch, duration_s=600.0)
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(
            hr_bpm=75.0,
            rr_intervals_ms=[800.0] * 10,
            signal_quality_score=0.95,
        ),
    )

    result = tool_mod.evaluate_proactive_rules(
        dataset="icentia11k",
        patient_id="00012",
        recording_id="00",
        window_start_s=0.0,
        window_end_s=900.0,
        include_phase2_rules=True,
    )

    assert result["success"] is True
    assert result["metadata"]["range_clamped"] is True
    assert result["metadata"]["window_end_s"] == 600.0
    assert result["data"]["trace"]["n_sub_windows"] == 60
    assert "phase2_caveat" in result["metadata"]


def test_evaluate_proactive_rules_30s_series_correct(monkeypatch) -> None:
    _patch_signal(monkeypatch, duration_s=60.0)
    hrs = iter([80.0, 90.0, 100.0, 70.0, 80.0, 110.0])
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(
            hr_bpm=next(hrs),
            rr_intervals_ms=[750.0] * 10,
            signal_quality_score=0.95,
        ),
    )

    result = tool_mod.evaluate_proactive_rules(
        dataset="icentia11k",
        patient_id="00012",
        recording_id="00",
        window_start_s=0.0,
        window_end_s=60.0,
    )

    trace = result["data"]["trace"]
    assert trace["hr_bpm_30s_series"] == pytest.approx([90.0, 86.6666667])
    assert trace["max_hr_30s_bpm"] == pytest.approx(90.0)
    assert trace["min_hr_30s_bpm"] == pytest.approx(86.6666667)
    assert trace["sub_window_30s_seconds"] == 30.0


def test_evaluate_proactive_rules_quality_warning_when_no_hr(monkeypatch) -> None:
    _patch_signal(monkeypatch, duration_s=20.0)
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(signal_quality_score=0.2),
    )

    result = tool_mod.evaluate_proactive_rules(
        dataset="wesad",
        patient_id="S2",
        window_start_s=0.0,
        window_end_s=20.0,
    )

    assert result["success"] is True
    assert result["data"]["rule_outcome"]["intervene"] is False
    assert result["data"]["rule_outcome"]["urgency"] == "none"
    assert result["data"]["trace"]["hr_bpm_series"] == [None, None]
    assert result["data"]["trace"]["rhythm_series"] == ["Unknown", "Unknown"]
    assert "No sub-window produced usable heart-rate vitals" in result["metadata"]["quality_warning"]


def test_rhythm_summary_handles_all_unknown_window(monkeypatch) -> None:
    _patch_signal(monkeypatch, duration_s=30.0)
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(signal_quality_score=0.2),
    )

    result = tool_mod.evaluate_proactive_rules(
        dataset="wesad",
        patient_id="S2",
        window_start_s=0.0,
        window_end_s=30.0,
    )

    summary = result["data"]["rhythm_summary"]
    assert summary["dominant_class"] is None
    assert summary["label_counts"]["Unknown"] == 3
    assert summary["sustained_af_episode_detected"] is False
    assert summary["sustained_n_episode_detected"] is False
    assert summary["sustained_other_episode_detected"] is False
    assert summary["valid_label_count"] == 0
    assert summary["dominant_fraction_of_valid"] == 0.0
    assert summary["sustained_episode_classes"] == []
    assert summary["stability_label"] == "unknown"


def test_afppgecg_loader_invocation(monkeypatch) -> None:
    calls = _patch_afppgecg_loader(monkeypatch, duration_s=120.0, fs_hz=100.0)
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(
            hr_bpm=78.0,
            rr_intervals_ms=[769.0] * 10,
            signal_quality_score=0.95,
        ),
    )

    result = tool_mod.evaluate_proactive_rules(
        dataset="afppgecg",
        patient_id="1",
        window_start_s=10.0,
        window_end_s=40.0,
    )

    assert result["success"] is True
    assert result["data"]["trace"]["n_sub_windows"] == 3
    assert result["metadata"]["ecg_channel"] == "aligned_ecg"
    assert result["metadata"]["sample_start"] == 1000
    assert result["metadata"]["sample_end"] == 4000
    assert calls == [
        {
            "patient_id": "1",
            "include_signal": False,
            "sampfrom": 0,
            "sampto": None,
        },
        {
            "patient_id": "1",
            "include_signal": True,
            "sampfrom": 1000,
            "sampto": 4000,
        },
    ]


def test_afppgecg_rhythm_summary_populated(monkeypatch) -> None:
    _patch_afppgecg_loader(monkeypatch, duration_s=120.0, fs_hz=100.0)
    labels = ["N", "AF", "AF", "AF", "Other"]
    iterator = iter(labels)
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(
            hr_bpm=92.0,
            rr_intervals_ms=[652.0] * 10,
            signal_quality_score=0.95,
        ),
    )
    monkeypatch.setattr(tool_mod, "classify_rhythm", lambda rr: next(iterator))

    result = tool_mod.evaluate_proactive_rules(
        dataset="afppgecg",
        patient_id="001",
        window_start_s=0.0,
        window_end_s=50.0,
    )

    summary = result["data"]["rhythm_summary"]
    assert result["success"] is True
    assert result["data"]["modality"] == "ecg"
    assert summary["label_counts"] == {"N": 1, "AF": 3, "Other": 1, "Unknown": 0}
    assert summary["dominant_class"] == "AF"
    assert summary["sustained_af_episode_detected"] is True
    assert summary["max_consecutive_runs"]["AF"] == 3
    assert summary["stability_label"] == "stable"


def test_afppgecg_30s_uses_qrs_index(monkeypatch) -> None:
    first_bin = np.linspace(0, 2999, 45, dtype=int)
    second_bin = np.linspace(3000, 5999, 30, dtype=int)
    qrs_index = np.concatenate([first_bin, second_bin])
    _patch_afppgecg_loader(
        monkeypatch,
        duration_s=60.0,
        fs_hz=100.0,
        qrs_index=qrs_index,
    )
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(
            hr_bpm=10.0,
            rr_intervals_ms=[1000.0] * 10,
            signal_quality_score=0.95,
        ),
    )

    result = tool_mod.evaluate_proactive_rules(
        dataset="afppgecg",
        patient_id="001",
        window_start_s=0.0,
        window_end_s=60.0,
    )

    trace = result["data"]["trace"]
    assert trace["hr_bpm_series"] == [90.0, 90.0, 90.0, 60.0, 60.0, 60.0]
    assert trace["hr_bpm_30s_series"] == [90.0, 60.0]
    assert trace["max_hr_30s_bpm"] == 90.0
    assert trace["min_hr_30s_bpm"] == 60.0


def test_rhythm_context_locator_window_scope(monkeypatch) -> None:
    _patch_afppgecg_loader(monkeypatch, duration_s=600.0, fs_hz=100.0)

    result = tool_mod.analyze_afppgecg_rhythm_context(
        patient_id="001",
        anchor_window_start_s=0.0,
        anchor_window_end_s=60.0,
        scope="locator_window",
    )

    assert result["success"] is True
    assert result["data"]["anchor_window_summary"]["scope"] == "locator_window"
    assert result["data"]["extended_context_summary"] is None
    assert result["metadata"]["uses_af_annotation"] is False


def test_rhythm_context_sampled_recording_scope(monkeypatch) -> None:
    qrs = np.cumsum(np.tile([100, 70, 130, 95, 105], 120)).astype(int)
    _patch_afppgecg_loader(monkeypatch, duration_s=600.0, fs_hz=100.0, qrs_index=qrs)

    result = tool_mod.analyze_afppgecg_rhythm_context(
        patient_id="001",
        anchor_window_start_s=0.0,
        anchor_window_end_s=60.0,
        scope="sampled_recording",
        n_sampled_chunks=20,
    )

    extended = result["data"]["extended_context_summary"]
    assert result["success"] is True
    assert extended["total_recording_seconds"] == 600.0
    assert len(extended["chunk_labels"]) == 20
    assert isinstance(extended["af_chunk_count"], int)
    assert 0.0 <= extended["af_chunk_fraction"] <= 1.0
    assert extended["signal_derived_variability_label_sampled"] in {
        "stable",
        "occasional_changes",
        "frequent_changes",
    }
    assert np.isfinite(extended["rr_sample_entropy"])


def test_rhythm_context_variability_label_uses_rate_not_raw_count(monkeypatch) -> None:
    qrs = np.arange(0, int(12 * 3600 * 100), 100, dtype=int)
    _patch_afppgecg_loader(monkeypatch, duration_s=12 * 3600.0, fs_hz=100.0, qrs_index=qrs)
    labels = iter(
        [
            "N",
            "AF",
            "N",
            "AF",
            "N",
            "AF",
            "N",
            "AF",
            "N",
            "AF",
            "N",
        ]
    )
    monkeypatch.setattr(tool_mod, "classify_rhythm", lambda rr: next(labels))

    result = tool_mod.analyze_afppgecg_rhythm_context(
        patient_id="001",
        anchor_window_start_s=0.0,
        anchor_window_end_s=60.0,
        scope="sampled_recording",
        n_sampled_chunks=10,
    )

    extended = result["data"]["extended_context_summary"]
    assert result["success"] is True
    assert extended["signal_derived_transition_count"] == 9
    assert extended["signal_derived_transition_per_hour_sampled"] == pytest.approx(0.75)
    assert extended["signal_derived_variability_label_sampled"] == "occasional_changes"


def test_rhythm_context_does_not_read_af_annotation(monkeypatch) -> None:
    from agent.data.ppg_loader import PPGLoader

    class FakeRecord:
        patient_id = "001"
        fs = 100
        n_samples = 60000
        qrs_index = np.arange(0, 60000, 100, dtype=int)
        rr_seconds = np.ones(599, dtype=float)
        signal = None
        metadata = {"record_path": "synthetic/001_ECG.mat"}

        @property
        def af_annotation(self):
            raise AssertionError("af_annotation must not be read")

    monkeypatch.setattr(
        PPGLoader,
        "read_ecg_record",
        lambda self, patient_id, **kwargs: FakeRecord(),
    )

    result = tool_mod.analyze_afppgecg_rhythm_context(
        patient_id="001",
        anchor_window_start_s=0.0,
        anchor_window_end_s=60.0,
        scope="sampled_recording",
        n_sampled_chunks=10,
    )

    assert result["success"] is True
    assert result["metadata"]["uses_af_annotation"] is False


def test_quality_gating_filters_low_quality_af(monkeypatch) -> None:
    _patch_signal(monkeypatch, duration_s=50.0)
    qualities = iter([0.4, 0.4, 0.4, 0.9, 0.9])
    labels = iter(["AF", "AF", "AF", "N", "N"])
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(
            hr_bpm=82.0,
            rr_intervals_ms=[731.0] * 10,
            signal_quality_score=next(qualities),
        ),
    )
    monkeypatch.setattr(tool_mod, "classify_rhythm", lambda rr: next(labels))

    result = tool_mod.evaluate_proactive_rules(
        dataset="icentia11k",
        patient_id="00012",
        recording_id="00",
        window_start_s=0.0,
        window_end_s=50.0,
    )

    summary = result["data"]["rhythm_summary"]
    assert result["data"]["trace"]["signal_quality_series"] == [0.4, 0.4, 0.4, 0.9, 0.9]
    assert summary["sustained_af_episode_detected"] is True
    assert summary["high_confidence_sustained_af_episode_detected"] is False
    assert summary["high_confidence_af_sub_window_count"] == 0
    assert summary["high_confidence_max_consecutive_af"] == 0
    assert summary["quality_gated_window_count"] == 2


def test_quality_gating_high_quality_passes(monkeypatch) -> None:
    _patch_signal(monkeypatch, duration_s=50.0)
    qualities = iter([0.85, 0.85, 0.85, 0.9, 0.9])
    labels = iter(["AF", "AF", "AF", "N", "N"])
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(
            hr_bpm=82.0,
            rr_intervals_ms=[731.0] * 10,
            signal_quality_score=next(qualities),
        ),
    )
    monkeypatch.setattr(tool_mod, "classify_rhythm", lambda rr: next(labels))

    result = tool_mod.evaluate_proactive_rules(
        dataset="icentia11k",
        patient_id="00012",
        recording_id="00",
        window_start_s=0.0,
        window_end_s=50.0,
    )

    summary = result["data"]["rhythm_summary"]
    assert result["data"]["trace"]["signal_quality_series"] == [0.85, 0.85, 0.85, 0.9, 0.9]
    assert summary["sustained_af_episode_detected"] is True
    assert summary["high_confidence_sustained_af_episode_detected"] is True
    assert summary["high_confidence_af_sub_window_count"] == 3
    assert summary["high_confidence_max_consecutive_af"] == 3
    assert summary["quality_gated_window_count"] == 5


def test_evaluate_proactive_rules_icentia_fixture_smoke() -> None:
    fixture = Path("dataset/icentia11k_subset/1.0/p00/p00012/p00012_s00.hea")
    if not fixture.exists():
        pytest.skip("Icentia11k local fixture is not available.")

    result = tool_mod.evaluate_proactive_rules(
        dataset="icentia11k",
        patient_id="00012",
        recording_id="00",
        window_start_s=0.0,
        window_end_s=20.0,
    )

    assert result["success"] is True
    assert result["data"]["trace"]["n_sub_windows"] == 2
    assert len(result["data"]["trace"]["hr_bpm_series"]) == 2


def test_icentia_max_30s_bpm_from_annotations() -> None:
    from agent.tools.datasets import icentia11k as icentia_tools

    fixture = Path("dataset/icentia11k_subset/1.0/p00/p00012/p00012_s00.hea")
    if not fixture.exists():
        pytest.skip("Icentia11k local fixture is not available.")

    result = icentia_tools.analyze_icentia11k_ecg_window_signal(
        dataset="icentia11k",
        patient_id="00012",
        recording_id="00",
        window_start_s=0.0,
        window_end_s=300.0,
    )

    assert result["success"] is True
    heart_rate = result["data"]["heart_rate"]
    assert np.isfinite(heart_rate["max_30s_bpm"])
    assert 30 <= heart_rate["max_30s_bpm"] <= 250
    assert heart_rate["max_30s_bpm_source"] == "icentia11k_beat_annotations"
