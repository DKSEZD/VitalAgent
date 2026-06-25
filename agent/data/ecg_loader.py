"""
ECG data loader for PTB-XL dataset.

Loads ECG signals from the PTB-XL dataset and provides text representations
for LLM understanding.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from agent.config import get_config

logger = logging.getLogger(__name__)

LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class ECGDatasetNotFoundError(FileNotFoundError):
    """Raised when the configured PTB-XL dataset root is unavailable."""


class ECGRecord:
    """A single ECG record with signal data and metadata."""

    def __init__(
        self,
        record_id: str,
        signal: np.ndarray,           # shape: (num_samples, num_leads)
        sampling_rate: int,
        lead_names: list[str],
        metadata: dict[str, Any] | None = None,
    ):
        self.record_id = record_id
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.lead_names = lead_names
        self.metadata = metadata or {}

    @property
    def num_leads(self) -> int:
        return self.signal.shape[1]

    @property
    def num_samples(self) -> int:
        return self.signal.shape[0]

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / self.sampling_rate

    def get_lead(self, lead: int | str) -> np.ndarray:
        """Return 1-D signal for a specific lead (by index or name)."""
        if isinstance(lead, str):
            lead = self.lead_names.index(lead)
        return self.signal[:, lead]

    def to_text(self) -> str:
        """Generate a text description of this ECG record for LLM understanding."""
        parts = []
        meta = self.metadata

        # Patient info
        age = meta.get("age")
        sex = meta.get("sex")
        sex_str = "Male" if sex == 0 else "Female" if sex == 1 else "Unknown"
        parts.append(f"ECG Record {self.record_id}:")
        if age is not None:
            parts.append(f"  Patient: age {age}, sex {sex_str}")
        parts.append(
            f"  Recording: {self.num_leads} leads, {self.sampling_rate}Hz, "
            f"{self.duration_seconds:.1f}s"
        )

        # Clinical report
        report = meta.get("report", "")
        if report:
            parts.append(f"  Clinical report: {report}")

        # Diagnostic codes
        scp_codes = meta.get("scp_codes", {})
        if scp_codes:
            codes_str = ", ".join(f"{k} ({v}%)" for k, v in scp_codes.items())
            parts.append(f"  SCP diagnostic codes: {codes_str}")

        diagnostic = meta.get("diagnostic_superclass", [])
        if diagnostic:
            parts.append(f"  Diagnostic class: {', '.join(diagnostic)}")

        return "\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"ECGRecord(id={self.record_id!r}, leads={self.num_leads}, "
            f"samples={self.num_samples}, sr={self.sampling_rate}Hz, "
            f"duration={self.duration_seconds:.1f}s)"
        )


class ECGDataLoader:
    """
    Loads ECG records from the PTB-XL dataset.

    Lazily loads individual records on-demand and caches them.
    Raises an explicit error if the configured PTB-XL dataset is missing.
    """

    def __init__(
        self,
        dataset_root: Path | str | None = None,
        sampling_rate: int = 100,
    ):
        cfg = get_config().ecg
        self.dataset_root = Path(dataset_root) if dataset_root else cfg.dataset_root
        self.sampling_rate = sampling_rate
        self._db: pd.DataFrame | None = None
        self._scp_df: pd.DataFrame | None = None
        self._cache: dict[str, ECGRecord] = {}
        self._loaded = False
        self._dataset_available = False

    def load(self) -> None:
        """Load dataset index. Call once before querying."""
        if self._loaded:
            return

        db_path = self.dataset_root / "ptbxl_database.csv"
        scp_path = self.dataset_root / "scp_statements.csv"

        if db_path.exists():
            self._load_ptbxl(db_path, scp_path)
            self._dataset_available = True
        else:
            message = (
                f"PTB-XL dataset not found at {self.dataset_root}. "
                f"Missing required file: {db_path}."
            )
            if not scp_path.exists():
                message += f" Optional SCP metadata file also missing: {scp_path}."
            logger.error(message)
            self._dataset_available = False
            raise ECGDatasetNotFoundError(message)

        self._loaded = True
        count = len(self._db) if self._db is not None else len(self._cache)
        logger.info("ECGDataLoader ready: %d records indexed.", count)

    # ------------------------------------------------------------------
    # PTB-XL loading
    # ------------------------------------------------------------------

    def _load_ptbxl(self, db_path: Path, scp_path: Path) -> None:
        self._db = pd.read_csv(db_path, index_col="ecg_id")
        self._db.scp_codes = self._db.scp_codes.apply(
            lambda x: ast.literal_eval(x)
        )

        if scp_path.exists():
            self._scp_df = pd.read_csv(scp_path, index_col=0)
            self._scp_df = self._scp_df[self._scp_df.diagnostic == 1]

            def _agg(codes: dict) -> list[str]:
                out = set()
                for code in codes:
                    if code in self._scp_df.index:
                        out.add(self._scp_df.loc[code].diagnostic_class)
                return list(out)

            self._db["diagnostic_superclass"] = self._db.scp_codes.apply(_agg)

        logger.info("Loaded PTB-XL database: %d records", len(self._db))

    # ------------------------------------------------------------------
    # Record access
    # ------------------------------------------------------------------

    def get_record(self, record_id: str | int) -> ECGRecord | None:
        """Fetch a single ECG record by ID (lazy-loaded and cached)."""
        self.load()
        rid = int(record_id) if isinstance(record_id, str) and record_id.isdigit() else record_id
        cache_key = str(rid)

        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._db is None:
            return None

        if rid not in self._db.index:
            return None

        row = self._db.loc[rid]
        filename = row.filename_lr if self.sampling_rate == 100 else row.filename_hr
        filepath = str(self.dataset_root / filename)

        try:
            signal, meta = wfdb.rdsamp(filepath)
        except Exception as e:
            logger.error("Failed to load record %s: %s", record_id, e)
            return None

        metadata = {
            "patient_id": str(int(row.patient_id)) if pd.notna(row.patient_id) else None,
            "age": row.age if pd.notna(row.age) else None,
            "sex": int(row.sex) if pd.notna(row.sex) else None,
            "report": str(row.report) if pd.notna(row.report) else "",
            "scp_codes": row.scp_codes,
            "recording_date": str(row.recording_date) if pd.notna(row.recording_date) else "",
            "diagnostic_superclass": (
                row.diagnostic_superclass
                if "diagnostic_superclass" in row.index
                else []
            ),
        }

        ecg_record = ECGRecord(
            record_id=cache_key,
            signal=signal,
            sampling_rate=int(meta["fs"]),
            lead_names=meta["sig_name"],
            metadata=metadata,
        )
        self._cache[cache_key] = ecg_record
        return ecg_record

    @staticmethod
    def _sex_to_text(sex_value: Any) -> str:
        if sex_value is None or (isinstance(sex_value, float) and pd.isna(sex_value)):
            return "unknown"
        try:
            sex_int = int(sex_value)
        except (TypeError, ValueError):
            return "unknown"
        if sex_int == 0:
            return "male"
        if sex_int == 1:
            return "female"
        return "unknown"

    def list_record_ids(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return available records with basic person metadata.

        Note: PTB-XL is anonymized and does not provide real names. We return
        `patient_id` as a pseudo-identifier plus demographic info.
        """
        self.load()

        if self._db is not None:
            subset = self._db.head(limit)
            records: list[dict[str, Any]] = []
            for ecg_id, row in subset.iterrows():
                diag = row.diagnostic_superclass if "diagnostic_superclass" in row.index else []
                if not isinstance(diag, list):
                    diag = []
                records.append(
                    {
                        "record_id": str(ecg_id),
                        "patient_id": str(int(row.patient_id)) if pd.notna(row.patient_id) else "unknown",
                        "name": None,
                        "age": int(row.age) if pd.notna(row.age) else None,
                        "sex": self._sex_to_text(row.sex),
                        "recording_date": str(row.recording_date) if pd.notna(row.recording_date) else None,
                        "diagnostic_class": diag,
                    }
                )
            return records
        return []

    def summarize_records(self, record_ids: list[str | int]) -> list[dict[str, Any]]:
        """Return structured metadata summaries for the requested PTB-XL records."""
        self.load()
        if self._db is None:
            return []

        records: list[dict[str, Any]] = []
        for rid in record_ids:
            rid_int = int(rid) if isinstance(rid, str) and rid.isdigit() else rid
            if rid_int not in self._db.index:
                continue

            row = self._db.loc[rid_int]
            diag = row.diagnostic_superclass if "diagnostic_superclass" in row.index else []
            if not isinstance(diag, list):
                diag = []

            patient_id = str(int(row.patient_id)) if pd.notna(row.patient_id) else None
            age = int(row.age) if pd.notna(row.age) else None
            sex_text = self._sex_to_text(row.sex)
            recording_date = str(row.recording_date) if pd.notna(row.recording_date) else None

            records.append(
                {
                    "record_id": str(rid_int),
                    "raw_record_id": str(rid_int),
                    "source": "ptbxl",
                    "patient_id": patient_id,
                    "age": age,
                    "sex": sex_text,
                    "recording_date": recording_date,
                    "diagnostic_class": diag,
                    "lead_names": list(LEAD_NAMES),
                    "num_leads": len(LEAD_NAMES),
                    "sampling_rate": self.sampling_rate,
                    "duration_seconds": 10.0,
                    "label": (
                        f"{rid_int} | source=ptbxl | age={age if age is not None else '-'}"
                        f" | sex={sex_text} | patient={patient_id if patient_id is not None else '-'}"
                    ),
                }
            )
        return records
    def search_records(
        self,
        age_min: int | None = None,
        age_max: int | None = None,
        sex: int | None = None,
        diagnostic_class: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Search records by metadata filters."""
        self.load()
        if self._db is None:
            return list(self._cache.keys())[:limit]

        mask = pd.Series(True, index=self._db.index)
        if age_min is not None:
            mask &= self._db.age >= age_min
        if age_max is not None:
            mask &= self._db.age <= age_max
        if sex is not None:
            mask &= self._db.sex == sex
        if diagnostic_class is not None and "diagnostic_superclass" in self._db.columns:
            mask &= self._db.diagnostic_superclass.apply(
                lambda x: diagnostic_class in x if isinstance(x, list) else False
            )

        return [str(x) for x in self._db[mask].index[:limit].tolist()]

    def get_record_text(self, record_id: str | int) -> str | None:
        """Get text description of a record for LLM context."""
        record = self.get_record(record_id)
        return record.to_text() if record else None

# Global loader instance for tools to use
_loaders: dict[int, ECGDataLoader] = {}

def get_loader(sampling_rate: int = 100) -> ECGDataLoader:
    if sampling_rate not in _loaders:
        _loaders[sampling_rate] = ECGDataLoader(sampling_rate=sampling_rate)
    return _loaders[sampling_rate]

def test_loader():
    loader = ECGDataLoader()
    loader.load()
    items = loader.list_record_ids(limit=3)
    print("Sample records:", items)
    for item in items:
        rid = item["record_id"]
        text = loader.get_record_text(rid)
        print(f"\nRecord {rid} description:\n{text[:500]}...")

if __name__ == "__main__":
    test_loader()

