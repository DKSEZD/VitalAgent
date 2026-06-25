"""Source-aware ECG record repository for PTB-XL and uploaded single-lead data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.data.apple_watch_loader import AppleWatchECGLoader
from agent.data.ecg_loader import ECGDatasetNotFoundError, ECGRecord, get_loader

PTBXL_SOURCE = "ptbxl"
APPLE_WATCH_SOURCE = "apple_watch"
PPG_SOURCE = "ppg"
SUPPORTED_RECORD_SOURCES = {PTBXL_SOURCE, APPLE_WATCH_SOURCE, PPG_SOURCE}


@dataclass(frozen=True)
class RecordReference:
    source: str
    raw_record_id: str

    @property
    def record_id(self) -> str:
        if self.source == PTBXL_SOURCE:
            return self.raw_record_id
        return f"{self.source}:{self.raw_record_id}"


class RecordRepository:
    """Lookup and list ECG records across multiple backing sources."""

    def __init__(self, apple_watch_loader: AppleWatchECGLoader | None = None):
        self.apple_watch_loader = apple_watch_loader or AppleWatchECGLoader()

    def parse_record_reference(self, record_id: str | int) -> RecordReference:
        text = str(record_id).strip()
        if ":" in text:
            source, raw_record_id = text.split(":", 1)
            source = source.strip().lower()
            raw_record_id = raw_record_id.strip()
            if source in SUPPORTED_RECORD_SOURCES and raw_record_id:
                return RecordReference(source=source, raw_record_id=raw_record_id)
        return RecordReference(source=PTBXL_SOURCE, raw_record_id=text)

    def get_record(self, record_id: str | int, *, sampling_rate: int = 100) -> ECGRecord | None:
        ref = self.parse_record_reference(record_id)
        if ref.source == PTBXL_SOURCE:
            try:
                return get_loader(sampling_rate).get_record(ref.raw_record_id)
            except ECGDatasetNotFoundError:
                return None
        if ref.source == APPLE_WATCH_SOURCE:
            return self.apple_watch_loader.get_record(ref.raw_record_id)
        if ref.source == PPG_SOURCE:
            return None
        return None

    def list_records(
        self,
        *,
        limit: int = 50,
        sampling_rate: int = 100,
        age_min: int | None = None,
        age_max: int | None = None,
        sex: int | None = None,
        diagnostic_class: str | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        records.extend(
            self._list_uploaded_records(
                age_min=age_min,
                age_max=age_max,
                sex=sex,
                diagnostic_class=diagnostic_class,
                limit=None,
            )
        )
        remaining = max(limit - len(records), 0)
        if remaining > 0:
            records.extend(
                self._list_ptbxl_records(
                    limit=remaining,
                    sampling_rate=sampling_rate,
                    age_min=age_min,
                    age_max=age_max,
                    sex=sex,
                    diagnostic_class=diagnostic_class,
                )
            )
        return records[:limit]

    def _list_ptbxl_records(
        self,
        *,
        limit: int,
        sampling_rate: int,
        age_min: int | None,
        age_max: int | None,
        sex: int | None,
        diagnostic_class: str | None,
    ) -> list[dict[str, Any]]:
        try:
            loader = get_loader(sampling_rate)
            record_ids = loader.search_records(
                age_min=age_min,
                age_max=age_max,
                sex=sex,
                diagnostic_class=diagnostic_class,
                limit=limit,
            )
            return loader.summarize_records(record_ids)
        except ECGDatasetNotFoundError:
            return []

    def _list_uploaded_records(
        self,
        *,
        age_min: int | None,
        age_max: int | None,
        sex: int | None,
        diagnostic_class: str | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        uploaded = self.apple_watch_loader.list_records(limit=limit)
        filtered: list[dict[str, Any]] = []
        for record in uploaded:
            age = record.get("age")
            if age is not None:
                try:
                    age = int(age)
                except (TypeError, ValueError):
                    age = None

            record_sex = record.get("sex")
            if age_min is not None and (age is None or age < age_min):
                continue
            if age_max is not None and (age is None or age > age_max):
                continue
            if sex is not None and record_sex is not None:
                sex_text = str(record_sex).strip().lower()
                if sex == 0 and sex_text not in {"0", "male", "m"}:
                    continue
                if sex == 1 and sex_text not in {"1", "female", "f"}:
                    continue
            if diagnostic_class is not None:
                diag = record.get("diagnostic_class") or []
                if diagnostic_class not in diag:
                    continue
            filtered.append(record)
        return filtered if limit is None else filtered[:limit]


_record_repository: RecordRepository | None = None


def get_record_repository() -> RecordRepository:
    global _record_repository
    if _record_repository is None:
        _record_repository = RecordRepository()
    return _record_repository


__all__ = [
    "APPLE_WATCH_SOURCE",
    "PPG_SOURCE",
    "PTBXL_SOURCE",
    "RecordReference",
    "RecordRepository",
    "SUPPORTED_RECORD_SOURCES",
    "get_record_repository",
]
