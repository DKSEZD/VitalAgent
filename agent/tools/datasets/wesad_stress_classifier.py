"""Learned WESAD stress-state classifier tool.

The tool intentionally trains only from raw-derived physiological features and
keeps subject-level leakage guards close to the model boundary.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from agent.config import get_config
from agent.mhealth.state_builders.wesad_state_builder import (
    WESAD_LABEL_FS_HZ,
    WESAD_PROTOCOL_LABELS,
)
from agent.tools.datasets.wesad import (
    SAMPLE_WESAD_ROOT,
    WESAD_CHEST_FS_HZ,
    WESAD_WRIST_ACC_FS_HZ,
    WESAD_WRIST_BVP_FS_HZ,
    WESAD_WRIST_LOW_RATE_FS_HZ,
    _load_wesad_pickle,
    _normalize_subject_id,
    _resolve_pickle_path,
)
from agent.tools.ppg.window_analysis import as_float_array, get_nested, slice_by_time
from agent.tools.registry import register_tool


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_NAMES = [
    "bvp_mean",
    "bvp_std",
    "bvp_min",
    "bvp_max",
    "bvp_range",
    "bvp_abs_mean",
    "bvp_diff_std",
    "bvp_dom_hr_bpm",
    "bvp_dom_power_ratio",
    "eda_mean",
    "eda_std",
    "eda_min",
    "eda_max",
    "eda_range",
    "eda_slope_per_s",
    "eda_diff_abs_mean",
    "acc_mag_mean",
    "acc_mag_std",
    "acc_mag_range",
    "acc_diff_abs_mean",
    "acc_x_std",
    "acc_y_std",
    "acc_z_std",
    "temp_mean",
    "temp_std",
    "temp_slope_per_s",
    "resp_mean",
    "resp_std",
    "resp_range",
    "resp_diff_std",
]


@dataclass(frozen=True)
class WESADFeatureWindow:
    subject_id: str
    start_s: float
    end_s: float
    label_id: int
    label_name: str
    features: list[float]


def _as_1d(signal: Any) -> np.ndarray | None:
    arr = as_float_array(signal)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] == 0:
            return None
        arr = arr[:, 0]
    return arr.reshape(-1)


def _as_2d(signal: Any) -> np.ndarray | None:
    arr = as_float_array(signal)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)


def _finite(values: np.ndarray | None) -> np.ndarray:
    if values is None or values.size == 0:
        return np.asarray([], dtype=float)
    flat = np.asarray(values, dtype=float).reshape(-1)
    return flat[np.isfinite(flat)]


def _stats(values: np.ndarray | None, *, include_diff_std: bool = False) -> list[float]:
    clean = _finite(values)
    if clean.size == 0:
        base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return base + ([0.0] if include_diff_std else [])
    base = [
        float(np.mean(clean)),
        float(np.std(clean)),
        float(np.min(clean)),
        float(np.max(clean)),
        float(np.max(clean) - np.min(clean)),
        float(np.mean(np.abs(clean))),
    ]
    if include_diff_std:
        diffs = np.diff(clean)
        base.append(float(np.std(diffs)) if diffs.size else 0.0)
    return base


def _dominant_hr_features(values: np.ndarray | None, fs_hz: int) -> tuple[float, float]:
    clean = _finite(values)
    if clean.size < max(8, fs_hz * 4):
        return 0.0, 0.0
    centered = clean - float(np.mean(clean))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(clean.size, d=1.0 / float(fs_hz))
    band = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(band):
        return 0.0, 0.0
    band_power = spectrum[band]
    if band_power.size == 0 or float(np.sum(band_power)) <= 0:
        return 0.0, 0.0
    best = int(np.argmax(band_power))
    dom_freq = float(freqs[band][best])
    ratio = float(band_power[best] / np.sum(band_power))
    return dom_freq * 60.0, ratio


def _trend_features(values: np.ndarray | None, *, duration_s: float) -> list[float]:
    clean = _finite(values)
    if clean.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    slope = 0.0
    if clean.size >= 2 and duration_s > 0:
        slope = float((clean[-1] - clean[0]) / duration_s)
    diffs = np.diff(clean)
    return [
        float(np.mean(clean)),
        float(np.std(clean)),
        float(np.min(clean)),
        float(np.max(clean)),
        float(np.max(clean) - np.min(clean)),
        slope,
        float(np.mean(np.abs(diffs))) if diffs.size else 0.0,
    ]


def _temp_features(values: np.ndarray | None, *, duration_s: float) -> list[float]:
    clean = _finite(values)
    if clean.size == 0:
        return [0.0, 0.0, 0.0]
    slope = 0.0
    if clean.size >= 2 and duration_s > 0:
        slope = float((clean[-1] - clean[0]) / duration_s)
    return [float(np.mean(clean)), float(np.std(clean)), slope]


def _acc_features(values: np.ndarray | None) -> list[float]:
    if values is None or values.size == 0:
        return [0.0] * 7
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mag = np.linalg.norm(arr, axis=1)
    diffs = np.diff(mag)
    axis_stds = [float(np.std(arr[:, index])) if index < arr.shape[1] else 0.0 for index in range(3)]
    return [
        float(np.mean(mag)) if mag.size else 0.0,
        float(np.std(mag)) if mag.size else 0.0,
        float(np.max(mag) - np.min(mag)) if mag.size else 0.0,
        float(np.mean(np.abs(diffs))) if diffs.size else 0.0,
        *axis_stds,
    ]


def extract_wesad_raw_feature_vector(
    payload: dict[str, Any],
    *,
    start_s: float,
    end_s: float,
) -> list[float]:
    """Extract fixed-order raw-derived features for one WESAD window."""

    duration_s = float(end_s) - float(start_s)
    if duration_s <= 0:
        raise ValueError("end_s must be greater than start_s.")

    wrist_bvp = _as_1d(get_nested(payload, ("signal", "wrist", "BVP")))
    wrist_acc = _as_2d(get_nested(payload, ("signal", "wrist", "ACC")))
    wrist_eda = _as_1d(get_nested(payload, ("signal", "wrist", "EDA")))
    wrist_temp = _as_1d(get_nested(payload, ("signal", "wrist", "TEMP")))
    chest_resp = _as_1d(get_nested(payload, ("signal", "chest", "Resp")))

    bvp_window = slice_by_time(wrist_bvp, fs_hz=WESAD_WRIST_BVP_FS_HZ, start_s=start_s, end_s=end_s)
    acc_window = slice_by_time(wrist_acc, fs_hz=WESAD_WRIST_ACC_FS_HZ, start_s=start_s, end_s=end_s)
    eda_window = slice_by_time(wrist_eda, fs_hz=WESAD_WRIST_LOW_RATE_FS_HZ, start_s=start_s, end_s=end_s)
    temp_window = slice_by_time(wrist_temp, fs_hz=WESAD_WRIST_LOW_RATE_FS_HZ, start_s=start_s, end_s=end_s)
    resp_window = slice_by_time(chest_resp, fs_hz=WESAD_CHEST_FS_HZ, start_s=start_s, end_s=end_s)

    dom_hr, dom_ratio = _dominant_hr_features(bvp_window, WESAD_WRIST_BVP_FS_HZ)
    bvp = _stats(bvp_window, include_diff_std=True)
    bvp.extend([dom_hr, dom_ratio])
    eda = _trend_features(eda_window, duration_s=duration_s)
    acc = _acc_features(acc_window)
    temp = _temp_features(temp_window, duration_s=duration_s)
    resp_stats = _stats(resp_window, include_diff_std=True)
    resp = [resp_stats[0], resp_stats[1], resp_stats[4], resp_stats[6]]
    features = bvp + eda + acc + temp + resp
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError(f"feature length mismatch: {len(features)} != {len(FEATURE_NAMES)}")
    return [float(np.nan_to_num(item, nan=0.0, posinf=0.0, neginf=0.0)) for item in features]


def _dominant_label(
    labels: np.ndarray,
    *,
    min_label_fraction: float,
    allowed_label_ids: set[int],
) -> tuple[int, float] | None:
    if labels.size == 0:
        return None
    values, counts = np.unique(labels.astype(int).reshape(-1), return_counts=True)
    best = int(np.argmax(counts))
    label_id = int(values[best])
    fraction = float(counts[best]) / float(labels.size)
    if label_id not in allowed_label_ids or fraction < min_label_fraction:
        return None
    return label_id, fraction


def _iter_clean_feature_windows(
    payload: dict[str, Any],
    *,
    subject_id: str,
    window_sec: int,
    stride_sec: int,
    max_windows_per_label: int,
    min_label_fraction: float,
) -> Iterable[WESADFeatureWindow]:
    labels = np.asarray(payload.get("label"), dtype=int).reshape(-1)
    if labels.size == 0:
        return
    allowed_label_ids = set(WESAD_PROTOCOL_LABELS)
    label_fs = WESAD_LABEL_FS_HZ
    window_samples = int(round(window_sec * label_fs))
    stride_samples = int(round(stride_sec * label_fs))
    counts: Counter[int] = Counter()
    cursor = 0
    while cursor + window_samples <= labels.size:
        classified = _dominant_label(
            labels[cursor : cursor + window_samples],
            min_label_fraction=min_label_fraction,
            allowed_label_ids=allowed_label_ids,
        )
        if classified is not None:
            label_id, _fraction = classified
            if counts[label_id] < max_windows_per_label:
                start_s = cursor / float(label_fs)
                end_s = (cursor + window_samples) / float(label_fs)
                yield WESADFeatureWindow(
                    subject_id=subject_id,
                    start_s=start_s,
                    end_s=end_s,
                    label_id=label_id,
                    label_name=str(WESAD_PROTOCOL_LABELS[label_id]),
                    features=extract_wesad_raw_feature_vector(
                        payload,
                        start_s=start_s,
                        end_s=end_s,
                    ),
                )
                counts[label_id] += 1
                if len(counts) == len(allowed_label_ids) and all(
                    counts[label_id] >= max_windows_per_label for label_id in allowed_label_ids
                ):
                    break
        cursor += stride_samples


def _candidate_roots(dataset_root: str | None) -> list[Path]:
    return (
        [Path(dataset_root)]
        if dataset_root
        else [get_config().datasets.wesad_root, SAMPLE_WESAD_ROOT]
    )


def _subject_pickle_paths(dataset_root: str | None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for root in _candidate_roots(dataset_root):
        if not root.exists():
            continue
        for subject_dir in sorted(root.glob("S*")):
            if not subject_dir.is_dir():
                continue
            candidate = subject_dir / f"{subject_dir.name}.pkl"
            if candidate.exists():
                paths[subject_dir.name] = candidate
    return paths


def _parse_subjects(value: Sequence[str] | str | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = [str(part).strip() for part in value]
    return {_normalize_subject_id(part) for part in parts if part}


@lru_cache(maxsize=8)
def _release_wesad_subjects_cached(path_text: str, mtime_ns: int) -> tuple[str, ...]:
    del mtime_ns
    path = Path(path_text)
    subjects: set[str] = set()
    if not path.exists():
        return tuple()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("dataset") == "wesad" and record.get("patient_id"):
                subjects.add(_normalize_subject_id(record.get("patient_id")))
    return tuple(sorted(subjects))


def release_wesad_subjects(path: str | Path | None = None) -> set[str]:
    # Released benchmark QA path follows the configured VitalBench root
    # (DATASET_VITALBENCH_ROOT), so the leakage guard tracks the dataset in use.
    qa_path = Path(path) if path else get_config().datasets.vitalbench_root / "qa.jsonl"
    if not qa_path.exists():
        return set()
    return set(_release_wesad_subjects_cached(str(qa_path.resolve()), qa_path.stat().st_mtime_ns))


def _fit_model(X: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # RandomForest stays small, handles non-linear feature interactions, and is
    # robust for the modest tabular feature set used here.
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
    ).fit(X, y)


def _predict_proba(model: Any, X: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(X)[0]
    classes = list(getattr(model, "classes_", getattr(model[-1], "classes_", [])))
    return {
        str(WESAD_PROTOCOL_LABELS.get(int(label_id), label_id)): round(float(prob), 4)
        for label_id, prob in zip(classes, probabilities, strict=False)
    }


def _collect_training_windows(
    *,
    dataset_root: str | None,
    train_subject_ids: set[str],
    excluded_subject_ids: set[str],
    window_sec: int,
    stride_sec: int,
    max_windows_per_label_per_subject: int,
    min_label_fraction: float,
) -> tuple[list[WESADFeatureWindow], dict[str, Any]]:
    subject_paths = _subject_pickle_paths(dataset_root)
    if train_subject_ids:
        subject_paths = {
            subject_id: path
            for subject_id, path in subject_paths.items()
            if subject_id in train_subject_ids
        }
    selected = {
        subject_id: path
        for subject_id, path in subject_paths.items()
        if subject_id not in excluded_subject_ids
    }
    windows: list[WESADFeatureWindow] = []
    errors: dict[str, str] = {}
    by_subject: dict[str, Counter[str]] = {}
    for subject_id, path in sorted(selected.items()):
        try:
            payload = _load_wesad_pickle(path)
            current = list(
                _iter_clean_feature_windows(
                    payload,
                    subject_id=subject_id,
                    window_sec=window_sec,
                    stride_sec=stride_sec,
                    max_windows_per_label=max_windows_per_label_per_subject,
                    min_label_fraction=min_label_fraction,
                )
            )
        except Exception as exc:
            errors[subject_id] = str(exc)
            continue
        windows.extend(current)
        by_subject[subject_id] = Counter(item.label_name for item in current)
    return windows, {
        "candidate_subject_count": len(subject_paths),
        "selected_train_subjects": sorted(selected),
        "excluded_subjects": sorted(excluded_subject_ids),
        "window_count_by_subject": {
            subject_id: dict(counter) for subject_id, counter in sorted(by_subject.items())
        },
        "subject_errors": errors,
    }


def _subject_wise_cv(windows: list[WESADFeatureWindow]) -> dict[str, Any]:
    subjects = sorted({window.subject_id for window in windows})
    if len(subjects) < 2:
        return {"available": False, "reason": "At least two training subjects are required."}
    rows = []
    correct = 0
    total = 0
    for held_out in subjects:
        train = [window for window in windows if window.subject_id != held_out]
        test = [window for window in windows if window.subject_id == held_out]
        if not train or not test:
            continue
        X_train = np.asarray([window.features for window in train], dtype=float)
        y_train = np.asarray([window.label_id for window in train], dtype=int)
        X_test = np.asarray([window.features for window in test], dtype=float)
        y_test = np.asarray([window.label_id for window in test], dtype=int)
        model = _fit_model(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test), dtype=int)
        subject_correct = int(np.sum(y_pred == y_test))
        subject_total = int(y_test.size)
        correct += subject_correct
        total += subject_total
        rows.append(
            {
                "held_out_subject_id": held_out,
                "train_subjects": sorted({window.subject_id for window in train}),
                "test_window_count": subject_total,
                "accuracy": round(subject_correct / subject_total, 4) if subject_total else None,
                "label_counts": dict(Counter(str(WESAD_PROTOCOL_LABELS[int(label)]) for label in y_test)),
            }
        )
    return {
        "available": bool(rows),
        "method": "leave_one_subject_out",
        "fold_count": len(rows),
        "accuracy": round(correct / total, 4) if total else None,
        "total_test_windows": total,
        "folds": rows,
    }


def _save_model_artifact(
    path: Path,
    *,
    model: Any,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "metadata": metadata,
                "feature_names": FEATURE_NAMES,
                "label_map": WESAD_PROTOCOL_LABELS,
            },
            handle,
        )


def _load_model_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Invalid WESAD classifier artifact: {path}")
    return artifact


@register_tool(
    name="classify_wesad_stress_state",
    description=(
        "Predict a WESAD window's baseline / stress / amusement / meditation "
        "state using a small supervised classifier trained only from raw-derived "
        "physiological features (BVP HR/HRV proxies, EDA, ACC/motion, TEMP, Resp). "
        "Training excludes the active subject and, by default, all WESAD subjects "
        "referenced by the release eval set to prevent subject/window leakage."
    ),
    parameters={
        "subject_id": {"type": "string", "description": "WESAD subject ID, e.g. 'S2'.", "required": False},
        "patient_id": {"type": "string", "description": "Alias for subject_id.", "required": False},
        "pickle_path": {"type": "string", "description": "Optional direct path to the active WESAD pickle.", "required": False},
        "dataset_root": {"type": "string", "description": "Optional WESAD root containing S*/S*.pkl folders.", "required": False},
        "dataset": {"type": "string", "description": "Expected dataset name: 'wesad'.", "required": False},
        "window_start_s": {"type": "number", "description": "Window start time in seconds.", "required": True},
        "window_end_s": {"type": "number", "description": "Window end time in seconds.", "required": True},
        "model_path": {"type": "string", "description": "Optional saved classifier artifact path.", "required": False},
        "train_subject_ids": {"type": "array", "description": "Optional explicit training subject IDs.", "required": False},
        "exclude_subject_ids": {"type": "array", "description": "Additional subject IDs to exclude from training.", "required": False},
        "exclude_release_eval_subjects": {"type": "boolean", "description": "Exclude release WESAD eval subjects from training.", "default": True},
        "evaluate_subject_wise": {"type": "boolean", "description": "Run leave-one-subject-out evaluation on training subjects.", "default": False},
    },
)
def classify_wesad_stress_state(
    subject_id: str | None = None,
    patient_id: str | None = None,
    pickle_path: str | None = None,
    dataset_root: str | None = None,
    dataset: str | None = None,
    window_start_s: float | None = None,
    window_end_s: float | None = None,
    model_path: str | None = None,
    train_subject_ids: Sequence[str] | str | None = None,
    exclude_subject_ids: Sequence[str] | str | None = None,
    exclude_release_eval_subjects: bool = True,
    evaluate_subject_wise: bool = False,
) -> dict[str, Any]:
    try:
        if dataset is not None and dataset != "wesad":
            raise ValueError(f"Unsupported dataset for WESAD stress classifier: {dataset!r}")
        if window_start_s is None or window_end_s is None:
            raise ValueError("window_start_s and window_end_s are required.")
        start_s = float(window_start_s)
        end_s = float(window_end_s)
        if end_s <= start_s:
            raise ValueError("window_end_s must be greater than window_start_s.")

        active_path = _resolve_pickle_path(
            pickle_path=pickle_path,
            subject_id=subject_id,
            patient_id=patient_id,
            dataset_root=dataset_root,
        )
        active_payload = _load_wesad_pickle(active_path)
        active_subject = _normalize_subject_id(active_payload.get("subject") or active_path.stem)
        excluded_subjects = _parse_subjects(exclude_subject_ids)
        excluded_subjects.add(active_subject)
        release_subjects = release_wesad_subjects() if exclude_release_eval_subjects else set()
        excluded_subjects.update(release_subjects)

        artifact_path = (
            Path(model_path)
            if model_path
            else get_config().datasets.wesad_stress_model_path
        )
        artifact: dict[str, Any] | None = None
        if artifact_path.exists():
            artifact = _load_model_artifact(artifact_path)
            trained_subjects = set(artifact.get("metadata", {}).get("train_subjects") or [])
            leaked_subjects = sorted(trained_subjects & excluded_subjects)
            if leaked_subjects:
                return {
                    "success": False,
                    "dataset": "wesad",
                    "error": (
                        "Saved WESAD classifier artifact violates subject-wise leakage guard: "
                        + ", ".join(leaked_subjects)
                    ),
                    "leakage_guard": {
                        "active_subject_id": active_subject,
                        "excluded_subjects": sorted(excluded_subjects),
                        "release_eval_subjects": sorted(release_subjects),
                    },
                }

        training_report: dict[str, Any] | None = None
        subject_wise_eval: dict[str, Any] | None = None
        if artifact is None:
            windows, training_report = _collect_training_windows(
                dataset_root=dataset_root,
                train_subject_ids=_parse_subjects(train_subject_ids),
                excluded_subject_ids=excluded_subjects,
                window_sec=60,
                stride_sec=60,
                max_windows_per_label_per_subject=20,
                min_label_fraction=0.95,
            )
            train_subjects = sorted({window.subject_id for window in windows})
            class_counts = Counter(window.label_name for window in windows)
            if len(train_subjects) < 2 or len(class_counts) < 2:
                return {
                    "success": False,
                    "dataset": "wesad",
                    "error": (
                        "Insufficient non-leaking WESAD training data after subject-wise "
                        "exclusions. Need at least two training subjects and two labels."
                    ),
                    "leakage_guard": {
                        "active_subject_id": active_subject,
                        "excluded_subjects": sorted(excluded_subjects),
                        "release_eval_subjects": sorted(release_subjects),
                        "strict_release_subject_exclusion": bool(exclude_release_eval_subjects),
                    },
                    "training_data": {
                        **(training_report or {}),
                        "train_subjects": train_subjects,
                        "class_counts": dict(class_counts),
                    },
                    "hint": (
                        "The local release WESAD split may cover every available WESAD "
                        "subject; add external/non-final subjects or provide a compliant "
                        "artifact whose metadata excludes eval subjects."
                    ),
                }
            X = np.asarray([window.features for window in windows], dtype=float)
            y = np.asarray([window.label_id for window in windows], dtype=int)
            model = _fit_model(X, y)
            metadata = {
                "train_subjects": train_subjects,
                "excluded_subjects": sorted(excluded_subjects),
                "release_eval_subjects": sorted(release_subjects),
                "class_counts": dict(class_counts),
                "feature_names": FEATURE_NAMES,
                "raw_feature_source": "WESAD wrist BVP/ACC/EDA/TEMP and chest Resp",
                "subject_wise_isolation": True,
            }
            if evaluate_subject_wise:
                subject_wise_eval = _subject_wise_cv(windows)
                metadata["subject_wise_eval"] = subject_wise_eval
            artifact = {"model": model, "metadata": metadata, "feature_names": FEATURE_NAMES}
            _save_model_artifact(artifact_path, model=model, metadata=metadata)

        features = extract_wesad_raw_feature_vector(active_payload, start_s=start_s, end_s=end_s)
        X_active = np.asarray([features], dtype=float)
        model = artifact["model"]
        label_id = int(model.predict(X_active)[0])
        label_name = str(WESAD_PROTOCOL_LABELS.get(label_id, label_id))
        probabilities = _predict_proba(model, X_active)
        metadata = dict(artifact.get("metadata") or {})
        return {
            "success": True,
            "dataset": "wesad",
            "subject_id": active_subject,
            "source_path": str(active_path),
            "data": {
                "window": {"start_s": start_s, "end_s": end_s, "duration_s": end_s - start_s},
                "predicted_stress_label": label_name,
                "predicted_protocol_label_id": label_id,
                "class_probabilities": probabilities,
                "top_probability": probabilities.get(label_name),
                "feature_summary": {
                    name: round(value, 6)
                    for name, value in zip(FEATURE_NAMES, features, strict=True)
                },
                "caveats": [
                    "Prediction is from a supervised model over raw-derived physiological features.",
                    "WESAD labels are lab protocol conditions, not clinical stress diagnoses.",
                ],
            },
            "metadata": {
                "model_path": str(artifact_path),
                "model_type": "StandardScaler + RandomForestClassifier",
                "feature_names": list(FEATURE_NAMES),
                "training": metadata,
                "training_report": training_report,
                "subject_wise_eval": subject_wise_eval or metadata.get("subject_wise_eval"),
                "leakage_guard": {
                    "active_subject_id": active_subject,
                    "active_subject_excluded_from_training": active_subject not in set(metadata.get("train_subjects") or []),
                    "excluded_subjects": sorted(excluded_subjects),
                    "release_eval_subjects": sorted(release_subjects),
                    "strict_release_subject_exclusion": bool(exclude_release_eval_subjects),
                },
            },
        }
    except Exception as exc:
        return {"success": False, "dataset": "wesad", "error": str(exc)}


__all__ = [
    "FEATURE_NAMES",
    "classify_wesad_stress_state",
    "extract_wesad_raw_feature_vector",
    "release_wesad_subjects",
]
