from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from agent.data.icentia11k_loader import Icentia11kLoader, PHYSIONET_DB, SegmentInfo


@pytest.mark.parametrize(
    ("raw_patient_id", "expected"),
    [
        ("12", "00012"),
        ("p-12", "00012"),
        (12, "00012"),
    ],
)
def test_normalize_patient_id_accepts_digits(raw_patient_id: str | int, expected: str) -> None:
    assert Icentia11kLoader.normalize_patient_id(raw_patient_id) == expected


@pytest.mark.parametrize("raw_patient_id", ["", "abc", "--"])
def test_normalize_patient_id_rejects_invalid_values(raw_patient_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid patient_id"):
        Icentia11kLoader.normalize_patient_id(raw_patient_id)


@pytest.mark.parametrize(
    ("raw_segment_id", "expected"),
    [
        ("3", "03"),
        ("s03", "03"),
        (3, "03"),
    ],
)
def test_normalize_segment_id_accepts_digits(raw_segment_id: str | int, expected: str) -> None:
    assert Icentia11kLoader.normalize_segment_id(raw_segment_id) == expected


@pytest.mark.parametrize("raw_segment_id", ["", "abc", "--"])
def test_normalize_segment_id_rejects_invalid_values(raw_segment_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid segment_id"):
        Icentia11kLoader.normalize_segment_id(raw_segment_id)


def test_build_record_name_remote() -> None:
    loader = Icentia11kLoader()

    assert loader.build_record_name("12", "3") == "p00/p00012/p00012_s03"


def test_build_record_name_local() -> None:
    loader = Icentia11kLoader(local_root="/tmp/icentia11k")

    assert loader.build_record_name("12", "3") == str(
        Path("/tmp/icentia11k") / "p00/p00012/p00012_s03"
    )


def test_read_segment_header_reads_wfdb_header() -> None:
    loader = Icentia11kLoader()
    start_time = datetime(2024, 5, 1, 8, 30, 0)

    with patch("agent.data.icentia11k_loader.wfdb.rdheader") as mock_rdheader:
        mock_rdheader.return_value = SimpleNamespace(sig_len=5000, fs=250, base_datetime=start_time)

        info = loader.read_segment_header("12", "3")

    assert info == SegmentInfo(
        patient_id="00012",
        segment_id="03",
        n_samples=5000,
        duration_seconds=20.0,
        fs=250,
        start_time=start_time,
    )
    mock_rdheader.assert_called_once_with(
        "p00012_s03",
        pn_dir=f"{PHYSIONET_DB}/p00/p00012",
    )


def test_read_segment_header_uses_base_date_and_time_when_needed() -> None:
    loader = Icentia11kLoader()
    expected_start = datetime(2024, 5, 1, 8, 30, 0)

    with patch("agent.data.icentia11k_loader.wfdb.rdheader") as mock_rdheader:
        mock_rdheader.return_value = SimpleNamespace(
            sig_len=5000,
            fs=250,
            base_date=date(2024, 5, 1),
            base_time=time(8, 30, 0),
        )

        info = loader.read_segment_header("12", "3")

    assert info.start_time == expected_start


def test_read_segment_parses_signal_and_annotations() -> None:
    loader = Icentia11kLoader()
    segment_start = datetime(2024, 5, 1, 8, 30, 0)
    annotation = SimpleNamespace(
        sample=np.array([100, 150, 400]),
        symbol=["N", "+", "V"],
        aux_note=["", "(AFIB", ""],
    )

    with patch("agent.data.icentia11k_loader.wfdb.rdheader") as mock_rdheader, patch(
        "agent.data.icentia11k_loader.wfdb.rdsamp"
    ) as mock_rdsamp, patch("agent.data.icentia11k_loader.wfdb.rdann") as mock_rdann:
        mock_rdheader.return_value = SimpleNamespace(
            sig_len=1000,
            fs=250,
            base_datetime=segment_start,
        )
        mock_rdsamp.return_value = (
            np.arange(400, dtype=float).reshape(-1, 1),
            {"sig_name": ["I"], "units": ["mV"]},
        )
        mock_rdann.return_value = annotation

        data = loader.read_segment("12", "3", sampfrom=100, sampto=500)

    assert data.patient_id == "00012"
    assert data.segment_id == "03"
    assert data.sample_start == 100
    assert data.sample_end == 500
    assert data.n_samples == 400
    assert data.fs == 250
    assert data.segment_start_time == segment_start
    assert data.window_start_time == segment_start + timedelta(seconds=0.4)
    assert data.window_duration_seconds == 1.6
    assert data.metadata["lead_names"] == ["I"]
    assert data.metadata["record_name"] == "p00/p00012/p00012_s03"
    assert np.array_equal(data.signal, np.arange(400, dtype=float))
    assert data.beat_annotations is not None
    assert np.array_equal(data.beat_annotations.samples, np.array([0, 300]))
    assert data.beat_annotations.symbols == ["N", "V"]
    assert data.rhythm_annotations is not None
    assert data.rhythm_annotations.start_samples == [50]
    assert data.rhythm_annotations.end_samples == [400]
    assert data.rhythm_annotations.rhythm_types == ["AFIB"]

    expected_kwargs = {"pn_dir": f"{PHYSIONET_DB}/p00/p00012"}
    mock_rdsamp.assert_called_once_with("p00012_s03", sampfrom=100, sampto=500, **expected_kwargs)
    mock_rdann.assert_called_once_with("p00012_s03", "atr", sampfrom=100, sampto=500, **expected_kwargs)


def test_parse_beat_annotations_keeps_beat_with_rhythm_aux_note() -> None:
    annotation = SimpleNamespace(
        sample=np.array([100, 150, 200]),
        symbol=["N", "+", "V"],
        aux_note=["(AFIB", "(AFIB", ""],
    )

    beats = Icentia11kLoader._parse_beat_annotations(annotation, sampfrom=100)

    assert beats is not None
    assert np.array_equal(beats.samples, np.array([0, 100]))
    assert beats.symbols == ["N", "V"]


def test_read_segment_truncates_sampto_to_segment_length() -> None:
    loader = Icentia11kLoader()

    with patch("agent.data.icentia11k_loader.wfdb.rdheader") as mock_rdheader, patch(
        "agent.data.icentia11k_loader.wfdb.rdsamp"
    ) as mock_rdsamp, patch(
        "agent.data.icentia11k_loader.wfdb.rdann",
        side_effect=FileNotFoundError("missing atr file"),
    ):
        mock_rdheader.return_value = SimpleNamespace(sig_len=300, fs=250, base_datetime=None)
        mock_rdsamp.return_value = (
            np.arange(50, dtype=float).reshape(-1, 1),
            {"sig_name": ["I"], "units": ["mV"]},
        )

        data = loader.read_segment("12", "3", sampfrom=250, sampto=500)

    assert data.sample_end == 300
    assert data.n_samples == 50
    mock_rdsamp.assert_called_once_with(
        "p00012_s03",
        sampfrom=250,
        sampto=300,
        pn_dir=f"{PHYSIONET_DB}/p00/p00012",
    )


@pytest.mark.parametrize(
    ("sampfrom", "sampto"),
    [
        (-1, 10),
        (50, 50),
        (60, 50),
    ],
)
def test_read_segment_rejects_invalid_sample_ranges(sampfrom: int, sampto: int) -> None:
    loader = Icentia11kLoader()

    with patch("agent.data.icentia11k_loader.wfdb.rdheader") as mock_rdheader:
        mock_rdheader.return_value = SimpleNamespace(sig_len=1000, fs=250, base_datetime=None)
        with pytest.raises(ValueError):
            loader.read_segment("12", "3", sampfrom=sampfrom, sampto=sampto)


def test_read_segment_rejects_multi_lead_signal() -> None:
    loader = Icentia11kLoader()

    with patch("agent.data.icentia11k_loader.wfdb.rdheader") as mock_rdheader, patch(
        "agent.data.icentia11k_loader.wfdb.rdsamp"
    ) as mock_rdsamp:
        mock_rdheader.return_value = SimpleNamespace(sig_len=1000, fs=250, base_datetime=None)
        mock_rdsamp.return_value = (
            np.zeros((10, 2), dtype=float),
            {"sig_name": ["I", "II"], "units": ["mV", "mV"]},
        )

        with pytest.raises(ValueError, match="single-lead signal"):
            loader.read_segment("12", "3", sampfrom=0, sampto=10)


def test_read_segment_allows_missing_annotations() -> None:
    loader = Icentia11kLoader()

    with patch("agent.data.icentia11k_loader.wfdb.rdheader") as mock_rdheader, patch(
        "agent.data.icentia11k_loader.wfdb.rdsamp"
    ) as mock_rdsamp, patch(
        "agent.data.icentia11k_loader.wfdb.rdann",
        side_effect=FileNotFoundError("missing atr file"),
    ):
        mock_rdheader.return_value = SimpleNamespace(sig_len=1000, fs=250, base_datetime=None)
        mock_rdsamp.return_value = (
            np.arange(50, dtype=float).reshape(-1, 1),
            {"sig_name": ["I"], "units": ["mV"]},
        )

        data = loader.read_segment("12", "3", sampfrom=10, sampto=60)

    assert data.beat_annotations is None
    assert data.rhythm_annotations is None


def test_read_segment_header_falls_back_to_remote_when_local_record_missing() -> None:
    loader = Icentia11kLoader(local_root="/tmp/icentia11k")
    start_time = datetime(2024, 5, 1, 8, 30, 0)

    with patch("agent.data.icentia11k_loader.wfdb.rdheader") as mock_rdheader:
        mock_rdheader.side_effect = [
            FileNotFoundError("missing local record"),
            SimpleNamespace(sig_len=5000, fs=250, base_datetime=start_time),
        ]

        info = loader.read_segment_header("12", "3")

    assert info == SegmentInfo(
        patient_id="00012",
        segment_id="03",
        n_samples=5000,
        duration_seconds=20.0,
        fs=250,
        start_time=start_time,
    )
    assert mock_rdheader.call_args_list[0].args[0] == str(
        Path("/tmp/icentia11k") / "p00/p00012/p00012_s03"
    )
    assert mock_rdheader.call_args_list[1].args[0] == "p00012_s03"
    assert mock_rdheader.call_args_list[1].kwargs == {
        "pn_dir": f"{PHYSIONET_DB}/p00/p00012",
    }


def test_read_segment_falls_back_to_remote_when_local_record_missing() -> None:
    loader = Icentia11kLoader(local_root="/tmp/icentia11k")
    segment_start = datetime(2024, 5, 1, 8, 30, 0)
    annotation = SimpleNamespace(
        sample=np.array([100, 150, 400]),
        symbol=["N", "+", "V"],
        aux_note=["", "(AFIB", ""],
    )

    with patch("agent.data.icentia11k_loader.wfdb.rdheader") as mock_rdheader, patch(
        "agent.data.icentia11k_loader.wfdb.rdsamp"
    ) as mock_rdsamp, patch("agent.data.icentia11k_loader.wfdb.rdann") as mock_rdann:
        mock_rdheader.side_effect = [
            FileNotFoundError("missing local header"),
            SimpleNamespace(sig_len=1000, fs=250, base_datetime=segment_start),
        ]
        mock_rdsamp.return_value = (
            np.arange(400, dtype=float).reshape(-1, 1),
            {"sig_name": ["I"], "units": ["mV"]},
        )
        mock_rdann.return_value = annotation

        data = loader.read_segment("12", "3", sampfrom=100, sampto=500)

    assert data.metadata["source"] == "physionet"
    assert data.metadata["record_name"] == "p00/p00012/p00012_s03"
    mock_rdsamp.assert_called_once_with(
        "p00012_s03",
        sampfrom=100,
        sampto=500,
        pn_dir=f"{PHYSIONET_DB}/p00/p00012",
    )
    mock_rdann.assert_called_once_with(
        "p00012_s03",
        "atr",
        sampfrom=100,
        sampto=500,
        pn_dir=f"{PHYSIONET_DB}/p00/p00012",
    )


def test_read_segment_falls_back_to_remote_annotations_when_local_atr_missing() -> None:
    loader = Icentia11kLoader(local_root="/tmp/icentia11k")
    segment_start = datetime(2024, 5, 1, 8, 30, 0)
    annotation = SimpleNamespace(
        sample=np.array([100, 150, 400]),
        symbol=["N", "+", "V"],
        aux_note=["", "(AFIB", ""],
    )

    with patch("agent.data.icentia11k_loader.wfdb.rdheader") as mock_rdheader, patch(
        "agent.data.icentia11k_loader.wfdb.rdsamp"
    ) as mock_rdsamp, patch("agent.data.icentia11k_loader.wfdb.rdann") as mock_rdann:
        mock_rdheader.return_value = SimpleNamespace(
            sig_len=1000,
            fs=250,
            base_datetime=segment_start,
        )
        mock_rdsamp.return_value = (
            np.arange(400, dtype=float).reshape(-1, 1),
            {"sig_name": ["I"], "units": ["mV"]},
        )
        mock_rdann.side_effect = [
            FileNotFoundError("missing local atr"),
            annotation,
        ]

        data = loader.read_segment("12", "3", sampfrom=100, sampto=500)

    assert data.rhythm_annotations is not None
    assert data.rhythm_annotations.rhythm_types == ["AFIB"]
    assert mock_rdann.call_args_list[0].args[0] == str(
        Path("/tmp/icentia11k") / "p00/p00012/p00012_s03"
    )
    assert mock_rdann.call_args_list[1].args[0] == "p00012_s03"


def test_get_patient_segments_stops_at_missing_segment() -> None:
    loader = Icentia11kLoader()
    seg0 = SegmentInfo("00012", "00", 5000, 20.0, 250)
    seg1 = SegmentInfo("00012", "01", 6000, 24.0, 250)

    with patch.object(
        loader,
        "read_segment_header",
        side_effect=[seg0, seg1, FileNotFoundError("missing segment")],
    ) as mock_read_header:
        segments = loader.get_patient_segments("12")

    assert segments == [seg0, seg1]
    assert mock_read_header.call_count == 3

def test_parse_rhythm_annotations_uses_explicit_end_marker() -> None:
    annotation = SimpleNamespace(
        sample=np.array([0, 75000, 100000]),
        symbol=["+", "+", "+"],
        aux_note=["(AFIB", ")", "(N"],
    )

    parsed = Icentia11kLoader._parse_rhythm_annotations(
        annotation,
        sampfrom=0,
        window_n_samples=150000,
    )

    assert parsed is not None
    assert parsed.start_samples == [0, 100000]
    assert parsed.end_samples == [75000, 150000]
    assert parsed.rhythm_types == ["AFIB", "N"]


def test_parse_rhythm_annotations_uses_labelled_explicit_end_marker() -> None:
    annotation = SimpleNamespace(
        sample=np.array([12000, 18500, 60000]),
        symbol=["+", "+", "+"],
        aux_note=["(AFIB", "AFIB)", "(AFIB"],
    )

    parsed = Icentia11kLoader._parse_rhythm_annotations(
        annotation,
        sampfrom=0,
        window_n_samples=100000,
    )

    assert parsed is not None
    assert parsed.start_samples == [12000, 60000]
    assert parsed.end_samples == [18500, 100000]
    assert parsed.rhythm_types == ["AFIB", "AFIB"]


def test_parse_rhythm_annotations_closes_previous_when_new_rhythm_starts() -> None:
    annotation = SimpleNamespace(
        sample=np.array([0, 75000]),
        symbol=["+", "+"],
        aux_note=["(AFIB", "(N"],
    )

    parsed = Icentia11kLoader._parse_rhythm_annotations(
        annotation,
        sampfrom=0,
        window_n_samples=150000,
    )

    assert parsed is not None
    assert parsed.start_samples == [0, 75000]
    assert parsed.end_samples == [75000, 150000]
    assert parsed.rhythm_types == ["AFIB", "N"]
