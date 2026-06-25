from __future__ import annotations

import json

import pytest

from agent.data.apple_watch_loader import AppleWatchECGLoader, AppleWatchUploadError


def test_create_record_from_json_payload_persists_and_reloads(tmp_path) -> None:
    loader = AppleWatchECGLoader(storage_root=tmp_path)

    record = loader.create_record_from_payload(
        {
            "samples": [0.0, 0.1, 0.2, 0.3],
            "sampling_rate": 512,
            "patient_id": "watch-user",
            "recorded_at": "2026-04-05T10:00:00Z",
        }
    )
    reloaded = loader.get_record(record.record_id)
    listed = loader.list_records(limit=5)

    assert reloaded is not None
    assert reloaded.record_id == record.record_id
    assert reloaded.lead_names == ["I"]
    assert reloaded.metadata["source"] == "apple_watch"
    assert listed[0]["record_id"] == f"apple_watch:{record.record_id}"
    assert listed[0]["sampling_rate"] == 512
    assert listed[0]["num_leads"] == 1


def test_create_record_from_csv_bytes_with_explicit_sampling_rate(tmp_path) -> None:
    loader = AppleWatchECGLoader(storage_root=tmp_path)

    record = loader.create_record_from_csv_bytes(
        b"sample\n0.0\n0.1\n0.2\n0.3\n",
        sampling_rate=250,
    )

    assert record.sampling_rate == 250
    assert record.lead_names == ["I"]


def test_create_record_from_csv_bytes_can_derive_sampling_rate_from_time_ms(tmp_path) -> None:
    loader = AppleWatchECGLoader(storage_root=tmp_path)

    record = loader.create_record_from_csv_bytes(
        b"sample,time_ms\n0.0,0\n0.1,4\n0.2,8\n0.3,12\n",
    )

    assert record.sampling_rate == 250


def test_create_record_from_csv_bytes_can_fallback_to_apple_export_text(tmp_path) -> None:
    loader = AppleWatchECGLoader(storage_root=tmp_path)

    record = loader.create_record_from_csv_bytes(
        (
            "姓名\t\n"
            "记录日期\t2026-03-04 22:21:22 +1000\n"
            "设备\tWatch7,9\n"
            "采样率\t512赫兹\n"
            "导联\t导联I\n"
            "单位\tµV\n"
            "\n"
            "8.569\t\n"
            "14.872\t\n"
            "20.928\t\n"
        ).encode("utf-8")
    )

    assert record.sampling_rate == 512
    assert record.lead_names == ["I"]
    assert record.metadata["recorded_at"] == "2026-03-04 22:21:22 +1000"
    assert record.metadata["device"] == "Watch7,9"
    assert record.metadata["signal_unit"] == "mV"
    assert record.metadata["signal_unit_original"] == "µV"
    assert record.metadata["signal_scale_applied"] == 0.001
    assert record.get_lead("I")[0] == pytest.approx(0.008569, rel=1e-6)


def test_create_record_from_upload_accepts_txt_apple_export(tmp_path) -> None:
    loader = AppleWatchECGLoader(storage_root=tmp_path)

    record = loader.create_record_from_upload(
        filename="watch.txt",
        content=(
            "记录日期\t2026-03-04 22:21:22 +1000\n"
            "设备\tWatch7,9\n"
            "采样率\t512赫兹\n"
            "导联\t导联I\n"
            "单位\tµV\n"
            "\n"
            "8.569\n"
            "14.872\n"
        ).encode("utf-8"),
    )

    assert record.sampling_rate == 512
    assert record.lead_names == ["I"]
    assert record.metadata["file_name"] == "watch.txt"
    assert record.metadata["file_format"] == "apple_export_text"


def test_create_record_from_csv_bytes_rejects_missing_sampling_information(tmp_path) -> None:
    loader = AppleWatchECGLoader(storage_root=tmp_path)

    with pytest.raises(AppleWatchUploadError, match="sampling_rate"):
        loader.create_record_from_csv_bytes(b"sample\n0.0\n0.1\n")


def test_record_id_rejects_path_traversal(tmp_path) -> None:
    loader = AppleWatchECGLoader(storage_root=tmp_path)

    with pytest.raises(AppleWatchUploadError, match="Invalid record_id"):
        loader.create_record_from_payload(
            {
                "samples": [0.0, 0.1],
                "sampling_rate": 250,
            },
            record_id="../../etc/passwd",
        )


def test_create_record_from_upload_json_applies_sampling_rate_and_lead_overrides(tmp_path) -> None:
    loader = AppleWatchECGLoader(storage_root=tmp_path)

    record = loader.create_record_from_upload(
        filename="watch.json",
        content=json.dumps({"samples": [0.0, 0.1, 0.2], "sampling_rate": 128, "lead_name": "II"}).encode("utf-8"),
        sampling_rate=512,
        lead_name="I",
    )

    assert record.sampling_rate == 512
    assert record.lead_names == ["I"]


def test_list_records_skips_corrupted_payloads(tmp_path) -> None:
    loader = AppleWatchECGLoader(storage_root=tmp_path)
    valid = loader.create_record_from_payload({"samples": [0.0, 0.1], "sampling_rate": 250})
    (tmp_path / "broken.json").write_text(
        json.dumps(
            {
                "record_id": "../broken",
                "sampling_rate": 250,
                "samples": [0.0, 0.1],
                "lead_name": "I",
                "metadata": {"source": "apple_watch"},
            }
        ),
        encoding="utf-8",
    )

    listed = loader.list_records(limit=None)

    assert [item["record_id"] for item in listed] == [f"apple_watch:{valid.record_id}"]
