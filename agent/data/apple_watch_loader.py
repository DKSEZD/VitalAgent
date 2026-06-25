"""Persistent loader for uploaded Apple Watch single-lead ECG records."""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from agent.config import get_config
from agent.data.ecg_loader import ECGRecord
from agent.data.single_lead import (
    SingleLeadNormalizationError,
    build_single_lead_ecg_record,
    derive_sampling_rate_from_time_ms,
    normalize_single_lead_name,
    validate_sampling_rate,
)

logger = logging.getLogger(__name__)

SUPPORTED_UPLOAD_EXTENSIONS = {".json", ".csv", ".txt", ".tsv"}
_VALID_RECORD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_NUMERIC_VALUE_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


class AppleWatchUploadError(ValueError):
    """Raised when an Apple Watch ECG upload payload is invalid."""


class AppleWatchECGLoader:
    """Persist and load uploaded Apple Watch ECG waveform records."""

    def __init__(self, storage_root: Path | str | None = None):
        cfg = get_config().ecg
        self.storage_root = Path(storage_root) if storage_root else cfg.upload_root
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ECGRecord] = {}

    def create_record_from_payload(
        self,
        payload: dict[str, Any],
        *,
        record_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        sampling_rate_override: int | None = None,
        lead_name_override: str | None = None,
    ) -> ECGRecord:
        samples = payload.get("samples")
        if samples is None:
            raise AppleWatchUploadError("JSON uploads must include a 'samples' array.")

        metadata = self._build_metadata(
            payload=payload,
            extra_metadata=extra_metadata,
        )
        sampling_rate = sampling_rate_override if sampling_rate_override is not None else payload.get("sampling_rate")
        lead_name = lead_name_override if lead_name_override is not None else payload.get("lead_name")
        return self._persist_record(
            record_id=record_id,
            signal=samples,
            sampling_rate=sampling_rate,
            lead_name=lead_name,
            metadata=metadata,
        )

    def create_record_from_json_bytes(
        self,
        raw_bytes: bytes,
        *,
        record_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        sampling_rate_override: int | None = None,
        lead_name_override: str | None = None,
    ) -> ECGRecord:
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AppleWatchUploadError(f"Invalid JSON ECG upload: {exc}") from exc

        if not isinstance(payload, dict):
            raise AppleWatchUploadError("JSON ECG upload must be an object.")

        return self.create_record_from_payload(
            payload,
            record_id=record_id,
            extra_metadata=extra_metadata,
            sampling_rate_override=sampling_rate_override,
            lead_name_override=lead_name_override,
        )

    def create_record_from_csv_bytes(
        self,
        raw_bytes: bytes,
        *,
        sampling_rate: int | None = None,
        lead_name: str | None = None,
        record_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ECGRecord:
        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise AppleWatchUploadError(f"Invalid CSV ECG upload encoding: {exc}") from exc

        reader = csv.DictReader(io.StringIO(text))
        fieldnames = [str(name).strip() for name in (reader.fieldnames or []) if name is not None]
        if not fieldnames:
            return self.create_record_from_apple_export_bytes(
                raw_bytes,
                sampling_rate=sampling_rate,
                lead_name=lead_name,
                record_id=record_id,
                extra_metadata=extra_metadata,
            )

        sample_column = next(
            (
                candidate
                for candidate in fieldnames
                if candidate.lower() in {"sample", "voltage"}
            ),
            None,
        )
        if sample_column is None:
            return self.create_record_from_apple_export_bytes(
                raw_bytes,
                sampling_rate=sampling_rate,
                lead_name=lead_name,
                record_id=record_id,
                extra_metadata=extra_metadata,
            )

        time_column = None
        for candidate in fieldnames:
            if candidate.lower() in {"time_ms", "timestamp_ms"}:
                time_column = candidate
                break

        samples: list[float] = []
        time_ms: list[float] = []
        for row in reader:
            value = row.get(sample_column)
            if value is None or str(value).strip() == "":
                continue
            try:
                samples.append(float(value))
            except ValueError as exc:
                raise AppleWatchUploadError(
                    f"CSV sample values must be numeric. Invalid value: {value!r}."
                ) from exc

            if time_column is not None:
                raw_time = row.get(time_column)
                if raw_time is None or str(raw_time).strip() == "":
                    raise AppleWatchUploadError(
                        f"CSV column '{time_column}' must be populated for every sample when provided."
                    )
                try:
                    time_ms.append(float(raw_time))
                except ValueError as exc:
                    raise AppleWatchUploadError(
                        f"CSV time values must be numeric. Invalid value: {raw_time!r}."
                    ) from exc

        if not samples:
            raise AppleWatchUploadError("CSV upload contains no ECG samples.")

        resolved_sampling_rate = sampling_rate
        if resolved_sampling_rate is None and time_column is not None:
            try:
                resolved_sampling_rate = derive_sampling_rate_from_time_ms(time_ms)
            except SingleLeadNormalizationError as exc:
                raise AppleWatchUploadError(str(exc)) from exc

        if resolved_sampling_rate is None:
            raise AppleWatchUploadError(
                "CSV uploads require an explicit sampling_rate unless a time_ms or timestamp_ms column is provided."
            )

        metadata = self._build_metadata(payload={}, extra_metadata=extra_metadata)
        metadata["file_format"] = "csv"
        if time_column is not None:
            metadata["time_column"] = time_column

        return self._persist_record(
            record_id=record_id,
            signal=samples,
            sampling_rate=resolved_sampling_rate,
            lead_name=lead_name,
            metadata=metadata,
        )

    def create_record_from_apple_export_bytes(
        self,
        raw_bytes: bytes,
        *,
        sampling_rate: int | None = None,
        lead_name: str | None = None,
        record_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ECGRecord:
        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise AppleWatchUploadError(f"Invalid Apple Watch ECG upload encoding: {exc}") from exc

        parsed = self._parse_apple_export_text(text)
        resolved_sampling_rate = sampling_rate
        if resolved_sampling_rate is None:
            resolved_sampling_rate = self._extract_sampling_rate_from_metadata(parsed["raw_metadata"])
        if resolved_sampling_rate is None:
            raise AppleWatchUploadError(
                "Apple Watch text exports require a sampling rate, either in the file metadata or at upload time."
            )

        resolved_lead_name = lead_name
        if resolved_lead_name is None:
            resolved_lead_name = self._extract_lead_name_from_metadata(parsed["raw_metadata"])

        normalized_samples, unit_metadata = self._normalize_signal_units(
            parsed["samples"],
            parsed["raw_metadata"],
        )

        metadata = self._build_metadata(payload={}, extra_metadata=extra_metadata)
        metadata.update(parsed["metadata"])
        metadata.update(unit_metadata)
        metadata["file_format"] = "apple_export_text"

        return self._persist_record(
            record_id=record_id,
            signal=normalized_samples,
            sampling_rate=resolved_sampling_rate,
            lead_name=resolved_lead_name,
            metadata=metadata,
        )

    def create_record_from_upload(
        self,
        *,
        filename: str,
        content: bytes,
        sampling_rate: int | None = None,
        lead_name: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ECGRecord:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise AppleWatchUploadError(
                f"Unsupported upload format {suffix or '<none>'!r}. Supported formats: JSON, CSV, TXT, and TSV."
            )

        metadata = dict(extra_metadata or {})
        metadata.setdefault("file_name", filename)
        metadata.setdefault("file_format", suffix.lstrip("."))

        if suffix == ".json":
            return self.create_record_from_json_bytes(
                content,
                extra_metadata=metadata,
                sampling_rate_override=sampling_rate,
                lead_name_override=lead_name,
            )

        if suffix in {".txt", ".tsv"}:
            return self.create_record_from_apple_export_bytes(
                content,
                sampling_rate=sampling_rate,
                lead_name=lead_name,
                extra_metadata=metadata,
            )

        return self.create_record_from_csv_bytes(
            content,
            sampling_rate=sampling_rate,
            lead_name=lead_name,
            extra_metadata=metadata,
        )

    def get_record(self, record_id: str) -> ECGRecord | None:
        cache_key = self._validate_record_id(record_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._record_path(cache_key)
        if not path.exists():
            return None

        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = dict(payload.get("metadata") or {})
        record = build_single_lead_ecg_record(
            record_id=cache_key,
            signal=payload.get("samples", []),
            sampling_rate=payload.get("sampling_rate"),
            lead_name=payload.get("lead_name"),
            metadata=metadata,
        )
        self._cache[cache_key] = record
        return record

    def list_records(self, limit: int | None = 50) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        paths = sorted(
            self.storage_root.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        selected_paths = paths if limit is None else paths[:limit]
        for path in selected_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payloads.append(self._summarize_payload(payload))
            except Exception as exc:
                logger.warning("Skipping unreadable Apple Watch upload %s: %s", path, exc)
                continue
        return payloads

    def _persist_record(
        self,
        *,
        record_id: str | None,
        signal: Any,
        sampling_rate: Any,
        lead_name: str | None,
        metadata: dict[str, Any] | None,
    ) -> ECGRecord:
        resolved_id = self._validate_record_id(record_id or self._generate_record_id())
        resolved_sampling_rate = validate_sampling_rate(sampling_rate)
        resolved_metadata = dict(metadata or {})
        resolved_metadata.setdefault("source", "apple_watch")
        resolved_metadata.setdefault("device", "Apple Watch")

        try:
            record = build_single_lead_ecg_record(
                record_id=resolved_id,
                signal=signal,
                sampling_rate=resolved_sampling_rate,
                lead_name=lead_name,
                metadata=resolved_metadata,
            )
        except SingleLeadNormalizationError as exc:
            raise AppleWatchUploadError(str(exc)) from exc

        payload = {
            "record_id": resolved_id,
            "sampling_rate": int(record.sampling_rate),
            "lead_name": record.lead_names[0],
            "samples": record.get_lead(record.lead_names[0]).astype(float).tolist(),
            "metadata": record.metadata,
        }
        self._record_path(resolved_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._cache[resolved_id] = record
        return record

    def _record_path(self, record_id: str) -> Path:
        safe_record_id = self._validate_record_id(record_id)
        storage_root = self.storage_root.resolve()
        candidate = (storage_root / f"{safe_record_id}.json").resolve()
        try:
            candidate.relative_to(storage_root)
        except ValueError as exc:
            raise AppleWatchUploadError(f"Unsafe record_id path: {record_id!r}.") from exc
        return candidate

    @staticmethod
    def _validate_record_id(record_id: str) -> str:
        normalized = str(record_id).strip()
        if not normalized or not _VALID_RECORD_ID_RE.fullmatch(normalized):
            raise AppleWatchUploadError(
                f"Invalid record_id {record_id!r}. Only letters, numbers, dot, underscore, and dash are allowed."
            )
        return normalized

    @staticmethod
    def _generate_record_id() -> str:
        return f"aw-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _split_delimited_line(raw_line: str) -> list[str]:
        if "\t" in raw_line:
            return [cell.strip() for cell in raw_line.split("\t")]
        if "," in raw_line:
            return [cell.strip() for cell in next(csv.reader([raw_line]))]
        return [raw_line.strip()]

    @classmethod
    def _is_numeric_token(cls, value: str) -> bool:
        return bool(_NUMERIC_VALUE_RE.fullmatch(value.strip()))

    @classmethod
    def _parse_apple_export_text(cls, text: str) -> dict[str, Any]:
        raw_metadata: dict[str, str] = {}
        samples: list[float] = []

        for raw_line in text.splitlines():
            cells = cls._split_delimited_line(raw_line)
            non_empty = [cell for cell in cells if cell != ""]
            if not non_empty:
                continue

            if len(non_empty) == 1 and cls._is_numeric_token(non_empty[0]):
                samples.append(float(non_empty[0]))
                continue

            if len(non_empty) >= 2:
                key = non_empty[0]
                value = non_empty[1]
                raw_metadata[key] = value
                continue

            if len(cells) >= 1 and cells[0] != "":
                raw_metadata[cells[0]] = ""
                continue

            raise AppleWatchUploadError(f"Unsupported Apple Watch ECG row format: {raw_line!r}.")

        if not samples:
            raise AppleWatchUploadError("Apple Watch text upload contains no ECG samples.")

        metadata: dict[str, Any] = {
            "source": "apple_watch",
            "device": raw_metadata.get("设备") or raw_metadata.get("Device") or "Apple Watch",
            "apple_watch_raw_metadata": raw_metadata,
        }
        if recorded_at := (
            raw_metadata.get("记录日期")
            or raw_metadata.get("记录时间")
            or raw_metadata.get("Recorded Date")
            or raw_metadata.get("recorded_at")
        ):
            metadata["recorded_at"] = recorded_at
        if classification := (
            raw_metadata.get("分类")
            or raw_metadata.get("Classification")
            or raw_metadata.get("Rhythm Classification")
        ):
            metadata["apple_watch_classification"] = classification
        if symptoms := raw_metadata.get("症状") or raw_metadata.get("Symptoms"):
            metadata["apple_watch_symptoms"] = symptoms
        if software_version := raw_metadata.get("软件版本") or raw_metadata.get("Software Version"):
            metadata["software_version"] = software_version
        if name := raw_metadata.get("姓名") or raw_metadata.get("Name"):
            metadata["patient_name"] = name
        if birth_date := raw_metadata.get("出生日期") or raw_metadata.get("Date of Birth"):
            metadata["birth_date"] = birth_date

        return {"samples": samples, "metadata": metadata, "raw_metadata": raw_metadata}

    @staticmethod
    def _extract_sampling_rate_from_metadata(raw_metadata: dict[str, str]) -> int | None:
        for key in ("采样率", "Sampling Rate", "sampling_rate"):
            value = raw_metadata.get(key)
            if not value:
                continue
            match = re.search(r"(\d+(?:\.\d+)?)", value)
            if match is None:
                continue
            return int(round(float(match.group(1))))
        return None

    @staticmethod
    def _extract_lead_name_from_metadata(raw_metadata: dict[str, str]) -> str | None:
        for key in ("导联", "Lead", "lead_name"):
            value = raw_metadata.get(key)
            if value:
                return value
        return None

    @staticmethod
    def _normalize_signal_units(
        samples: list[float],
        raw_metadata: dict[str, str],
    ) -> tuple[list[float], dict[str, Any]]:
        unit_value = raw_metadata.get("单位") or raw_metadata.get("Unit") or raw_metadata.get("Units")
        normalized_samples = np.asarray(samples, dtype=np.float32)
        metadata: dict[str, Any] = {}

        if unit_value is None or str(unit_value).strip() == "":
            metadata["signal_unit"] = "mV"
            metadata["signal_scale_applied"] = 1.0
            return normalized_samples.astype(float).tolist(), metadata

        cleaned_unit = str(unit_value).strip().replace("μ", "µ").lower()
        metadata["signal_unit_original"] = unit_value

        if cleaned_unit in {"µv", "uv", "microvolt", "microvolts"}:
            metadata["signal_unit"] = "mV"
            metadata["signal_scale_applied"] = 0.001
            return (normalized_samples / 1000.0).astype(float).tolist(), metadata

        metadata["signal_unit"] = unit_value
        metadata["signal_scale_applied"] = 1.0
        return normalized_samples.astype(float).tolist(), metadata

    @staticmethod
    def _build_metadata(
        *,
        payload: dict[str, Any],
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = dict(extra_metadata or {})
        for key in ("recorded_at", "device", "patient_id", "source", "age", "sex"):
            value = payload.get(key)
            if value is not None and value != "":
                metadata[key] = value
        metadata.setdefault("source", "apple_watch")
        metadata.setdefault("device", "Apple Watch")
        return metadata

    @classmethod
    def _summarize_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(payload.get("metadata") or {})
        sample_count = len(payload.get("samples") or [])
        sampling_rate = int(payload.get("sampling_rate") or 0)
        duration_seconds = round(sample_count / sampling_rate, 1) if sampling_rate > 0 else 0.0
        lead_name = normalize_single_lead_name(payload.get("lead_name"))
        source = str(metadata.get("source", "apple_watch"))
        raw_record_id = cls._validate_record_id(str(payload.get("record_id")))
        return {
            "record_id": f"{source}:{raw_record_id}",
            "raw_record_id": raw_record_id,
            "source": source,
            "patient_id": metadata.get("patient_id"),
            "age": metadata.get("age"),
            "sex": metadata.get("sex"),
            "recording_date": metadata.get("recorded_at"),
            "diagnostic_class": metadata.get("diagnostic_superclass", []),
            "lead_names": [lead_name],
            "num_leads": 1,
            "sampling_rate": sampling_rate,
            "duration_seconds": duration_seconds,
            "label": (
                f"apple_watch:{raw_record_id} | lead={lead_name} | sr={sampling_rate}Hz | "
                f"duration={duration_seconds:.1f}s"
            ),
        }


__all__ = [
    "AppleWatchECGLoader",
    "AppleWatchUploadError",
    "SUPPORTED_UPLOAD_EXTENSIONS",
]
