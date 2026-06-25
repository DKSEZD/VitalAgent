"""Shared helpers for normalizing single-lead ECG records."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from agent.data.ecg_loader import ECGRecord

_LEAD_ALIASES = {
    "I": "I",
    "LEAD I": "I",
    "LEADI": "I",
    "1": "I",
}


class SingleLeadNormalizationError(ValueError):
    """Raised when a single-lead waveform payload cannot be normalized."""


def normalize_single_lead_name(lead_name: str | None) -> str:
    """Normalize single-lead names, defaulting to Lead I."""
    if lead_name is None:
        return "I"
    cleaned = " ".join(str(lead_name).strip().upper().split())
    if not cleaned:
        return "I"
    cleaned = cleaned.replace("导联", "").strip()
    if cleaned.startswith("LEAD"):
        cleaned = cleaned[4:].strip()
    return _LEAD_ALIASES.get(cleaned, cleaned)


def coerce_single_lead_signal(signal: Iterable[Any] | np.ndarray) -> np.ndarray:
    """Coerce 1-D or Nx1 waveform inputs to a validated 1-D float32 array."""
    try:
        signal_array = np.asarray(signal, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise SingleLeadNormalizationError(
            "Single-lead ECG samples must be numeric."
        ) from exc

    if signal_array.ndim == 1:
        normalized = signal_array
    elif signal_array.ndim == 2 and 1 in signal_array.shape:
        normalized = signal_array.reshape(-1)
    else:
        raise SingleLeadNormalizationError(
            f"Expected a single-lead signal, got shape {signal_array.shape!r}."
        )

    if normalized.size == 0:
        raise SingleLeadNormalizationError("Single-lead ECG samples cannot be empty.")
    if not np.isfinite(normalized).all():
        raise SingleLeadNormalizationError(
            "Single-lead ECG samples must be finite numeric values."
        )

    return normalized.astype(np.float32, copy=False)


def validate_sampling_rate(sampling_rate: Any) -> int:
    """Validate and normalize sampling rate to a positive integer."""
    try:
        normalized = int(sampling_rate)
    except (TypeError, ValueError) as exc:
        raise SingleLeadNormalizationError(
            f"Invalid sampling rate: {sampling_rate!r}."
        ) from exc
    if normalized <= 0:
        raise SingleLeadNormalizationError(
            f"Sampling rate must be positive, got {normalized!r}."
        )
    return normalized


def derive_sampling_rate_from_time_ms(time_ms: Iterable[Any] | np.ndarray) -> int:
    """Derive sampling rate from time stamps in milliseconds."""
    try:
        time_array = np.asarray(time_ms, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise SingleLeadNormalizationError(
            "Time columns must contain numeric millisecond values."
        ) from exc

    if time_array.size < 2:
        raise SingleLeadNormalizationError(
            "At least two time points are required to derive sampling rate."
        )

    diffs = np.diff(time_array)
    if np.any(diffs <= 0):
        raise SingleLeadNormalizationError(
            "Time values must be strictly increasing to derive sampling rate."
        )

    median_diff = float(np.median(diffs))
    if median_diff <= 0:
        raise SingleLeadNormalizationError("Derived sampling interval must be positive.")

    if not np.allclose(diffs, median_diff, rtol=0.05, atol=1e-3):
        raise SingleLeadNormalizationError(
            "Time values are not evenly spaced enough to derive a stable sampling rate."
        )

    sampling_rate = round(1000.0 / median_diff)
    if sampling_rate <= 0:
        raise SingleLeadNormalizationError("Failed to derive a valid sampling rate.")
    return int(sampling_rate)


def normalize_single_lead_metadata(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a shallow-copied metadata dict with normalized single-lead hints."""
    normalized = dict(metadata or {})
    normalized["lead_names"] = [normalize_single_lead_name(normalized.get("lead_name"))]
    normalized["num_leads"] = 1
    return normalized


def build_single_lead_ecg_record(
    *,
    record_id: str,
    signal: Iterable[Any] | np.ndarray,
    sampling_rate: Any,
    lead_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ECGRecord:
    """Build an ECGRecord-compatible object for a single-lead waveform."""
    normalized_signal = coerce_single_lead_signal(signal)
    normalized_sampling_rate = validate_sampling_rate(sampling_rate)
    normalized_lead_name = normalize_single_lead_name(lead_name)

    normalized_metadata = normalize_single_lead_metadata(metadata)
    normalized_metadata["lead_name"] = normalized_lead_name

    return ECGRecord(
        record_id=str(record_id),
        signal=normalized_signal.reshape(-1, 1),
        sampling_rate=normalized_sampling_rate,
        lead_names=[normalized_lead_name],
        metadata=normalized_metadata,
    )
