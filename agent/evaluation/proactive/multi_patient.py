"""Multi-patient proactive evaluation runner."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from agent.evaluation.proactive.config import (
    default_afppgecg_dataset_root,
    default_icentia11k_dataset_root,
)
from agent.evaluation.proactive.metrics import (
    evaluate_intervention_replay,
    evaluate_transition_detection,
)


@dataclass(frozen=True)
class MultiPatientEntry:
    patient_id: str
    segment_id: str | None
    note: str


ICENTIA_CURATED_PATIENTS: tuple[MultiPatientEntry, ...] = (
    MultiPatientEntry("01182", "00", "AFIB+N calibration"),
    MultiPatientEntry("01182", "01", "AFIB+N holdout"),
    MultiPatientEntry("00060", None, "AFIB"),
    MultiPatientEntry("00065", None, "AFIB"),
    MultiPatientEntry("00075", None, "AFIB"),
    MultiPatientEntry("00230", None, "AFIB"),
    MultiPatientEntry("00287", None, "AFIB"),
    MultiPatientEntry("00299", None, "AFIB"),
    MultiPatientEntry("00323", None, "AFIB"),
    MultiPatientEntry("00365", None, "AFIB"),
    MultiPatientEntry("00404", None, "AFIB"),
    MultiPatientEntry("00434", None, "AFIB"),
    MultiPatientEntry("00119", None, "AFIB low-TPR"),
    MultiPatientEntry("00446", None, "AFIB poor signal"),
    MultiPatientEntry("00135", None, "AFL"),
)

MULTI_PATIENT_CSV_FIELDS = [
    "patient_id",
    "segment_id",
    "note",
    "eval_target",
    "windows_processed",
    "truth_af_windows",
    "truth_n_windows",
    "af_sensitivity",
    "n_specificity",
    "gt_transitions",
    "pred_transitions",
    "gt_episodes",
    "intervention_events",
    "matched_episodes",
    "missed_episodes",
    "false_alerts",
    "duplicate_alerts",
    "precision",
    "recall",
    "f1",
    "latency_windows_mean",
    "latency_windows_median",
    "latency_seconds_mean",
    "truth_unknown_windows",
    "predicted_unknown_windows",
]


def discover_icentia11k_patient_ids(dataset_root: Path) -> list[str]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Icentia11k dataset root does not exist: {root}")

    patient_ids = {
        patient_dir.name.removeprefix("p")
        for shard_dir in root.glob("p*")
        if shard_dir.is_dir()
        for patient_dir in shard_dir.glob("p*")
        if patient_dir.is_dir() and patient_dir.name.removeprefix("p").isdigit()
    }
    if not patient_ids:
        raise ValueError(
            "No Icentia11k patient directories were found. Provide --patient-id "
            "or set PROACTIVE_ICENTIA11K_ROOT to a local dataset mirror."
        )
    return sorted(patient_ids)


def build_multi_patient_entries(
    *,
    dataset: str,
    loader: Any,
    dataset_root: Path,
    requested_patient_ids: Iterable[str] | None,
    explicit_segment_id: str | None,
) -> list[MultiPatientEntry]:
    requested = list(requested_patient_ids or [])
    if requested:
        return [
            _normalise_multi_patient_entry(
                dataset=dataset,
                loader=loader,
                patient_id=patient_id,
                explicit_segment_id=explicit_segment_id,
            )
            for patient_id in requested
        ]

    if dataset == "afppgecg":
        return _available_afppgecg_entries(loader, explicit_segment_id)

    return [
        _normalise_multi_patient_entry(
            dataset=dataset,
            loader=loader,
            patient_id=patient_id,
            explicit_segment_id=explicit_segment_id,
        )
        for patient_id in discover_icentia11k_patient_ids(dataset_root)
    ]


def run_multi_patient_evaluation(
    args: argparse.Namespace,
    *,
    pipeline_factory: Callable[[], Any] | None = None,
) -> int:
    from agent.data.streaming import ECGStreamer

    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root = (
            default_afppgecg_dataset_root()
            if args.dataset == "afppgecg"
            else default_icentia11k_dataset_root()
        )
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if args.dataset == "afppgecg":
        from agent.data.afppgecg_adapter import AFPPGECGSegmentLoader

        loader = AFPPGECGSegmentLoader(dataset_root=dataset_root)
    else:
        from agent.data.icentia11k_loader import Icentia11kLoader

        loader = Icentia11kLoader(local_root=dataset_root)

    patients = build_multi_patient_entries(
        dataset=args.dataset,
        loader=loader,
        dataset_root=dataset_root,
        requested_patient_ids=([args.patient_id] if args.patient_id is not None else None),
        explicit_segment_id=args.segment_id,
    )
    streamer = ECGStreamer(
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )
    output_dir = args.output_dir or Path("results/proactive")
    output_dir.mkdir(parents=True, exist_ok=True)
    make_pipeline = pipeline_factory or (lambda: _make_pipeline(args.llm_mode))

    csv_rows: list[dict[str, Any]] = []
    for patient in patients:
        row = _run_one_patient(
            patient=patient,
            dataset=args.dataset,
            loader=loader,
            streamer=streamer,
            args=args,
            pipeline_factory=make_pipeline,
            output_dir=output_dir,
        )
        csv_rows.append(row)

    csv_path = output_dir / "multi_patient_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MULTI_PATIENT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSummary CSV saved to {csv_path}")
    _print_multi_patient_table(csv_rows, args.eval_target)
    return 0


def _normalise_multi_patient_entry(
    *,
    dataset: str,
    loader: Any,
    patient_id: str,
    explicit_segment_id: str | None,
) -> MultiPatientEntry:
    patient = loader.normalize_patient_id(patient_id)
    segment = explicit_segment_id
    note = "afppgecg" if dataset == "afppgecg" else "ad-hoc"
    if explicit_segment_id is None and dataset == "icentia11k":
        curated = {entry.patient_id: entry for entry in ICENTIA_CURATED_PATIENTS}
        if patient in curated:
            segment = curated[patient].segment_id
            note = curated[patient].note
    return MultiPatientEntry(patient, segment, note)


def _available_afppgecg_entries(
    loader: Any,
    explicit_segment_id: str | None,
) -> list[MultiPatientEntry]:
    entries: list[MultiPatientEntry] = []
    for patient_id in loader.available_patient_ids():
        patient = loader.normalize_patient_id(patient_id)
        if explicit_segment_id is None:
            try:
                loader.get_patient_segments(patient)
            except OSError as exc:
                print(f"Skipping AFPPGECG patient {patient}: {exc}")
                continue
        entries.append(MultiPatientEntry(patient, explicit_segment_id, "afppgecg"))
    return entries


def _run_one_patient(
    *,
    patient: MultiPatientEntry,
    dataset: str,
    loader: Any,
    streamer: Any,
    args: argparse.Namespace,
    pipeline_factory: Callable[[], Any],
    output_dir: Path,
) -> dict[str, Any]:
    label = f"{patient.patient_id}" + (f" seg{patient.segment_id}" if patient.segment_id else "")
    print(f"\n{'-' * 55}")
    print(f"Patient {label} [{patient.note}]")
    print(f"{'-' * 55}")

    try:
        segment_id, windows = _patient_windows(
            patient=patient,
            dataset=dataset,
            loader=loader,
            streamer=streamer,
            max_segments=args.max_segments,
            max_windows=args.max_windows,
        )
        limited_windows = windows
        if args.max_windows is not None:
            limited_windows = (
                window for index, window in enumerate(windows)
                if index < args.max_windows
            )
        pipeline = pipeline_factory()
        if args.eval_target == "intervention":
            artifacts = evaluate_intervention_replay(
                limited_windows,
                pipeline,
                pre_onset_tolerance_windows=args.pre_onset_tolerance_windows,
            )
        else:
            artifacts = evaluate_transition_detection(
                limited_windows,
                pipeline,
                debounce_windows=args.debounce,
            )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return {
            "patient_id": patient.patient_id,
            "segment_id": patient.segment_id or "?",
            "note": patient.note,
            "eval_target": args.eval_target,
            "f1": "ERROR",
        }

    report = artifacts.report
    af_sens, n_spec, n_af, n_n = _af_sensitivity(
        [record.__dict__ if hasattr(record, "__dict__") else record for record in artifacts.window_records]
    )
    _print_patient_summary(report, af_sens, n_spec, n_af, n_n, args.eval_target)

    suffix = "intervention" if args.eval_target == "intervention" else f"d{args.debounce}"
    out_name = f"{patient.patient_id}_seg{patient.segment_id or segment_id}_{suffix}.json"
    out_path = output_dir / out_name
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"  Saved to {out_path}")

    row = {
        "patient_id": patient.patient_id,
        "segment_id": patient.segment_id or segment_id,
        "note": patient.note,
        "eval_target": args.eval_target,
        "windows_processed": report.windows_processed,
        "truth_af_windows": n_af,
        "truth_n_windows": n_n,
        "af_sensitivity": f"{af_sens:.4f}" if af_sens is not None else "",
        "n_specificity": f"{n_spec:.4f}" if n_spec is not None else "",
        "precision": f"{report.precision:.4f}",
        "recall": f"{report.recall:.4f}",
        "f1": f"{report.f1:.4f}",
        "latency_windows_mean": (
            f"{report.latency_windows_mean:.2f}"
            if report.latency_windows_mean is not None
            else ""
        ),
        "latency_windows_median": (
            f"{report.latency_windows_median:.2f}"
            if report.latency_windows_median is not None
            else ""
        ),
        "latency_seconds_mean": (
            f"{report.latency_seconds_mean:.1f}"
            if report.latency_seconds_mean is not None
            else ""
        ),
        "truth_unknown_windows": report.truth_unknown_window_count,
        "predicted_unknown_windows": getattr(report, "predicted_unknown_window_count", ""),
    }
    if args.eval_target == "intervention":
        row.update(
            {
                "gt_episodes": report.ground_truth_episode_total,
                "intervention_events": report.intervention_event_total,
                "matched_episodes": report.matched_episode_total,
                "missed_episodes": report.missed_episode_total,
                "false_alerts": report.false_alert_total,
                "duplicate_alerts": report.duplicate_alert_total,
            }
        )
    else:
        row.update(
            {
                "gt_transitions": report.ground_truth_transition_total,
                "pred_transitions": report.predicted_transition_total,
            }
        )
    return row


def _patient_windows(
    *,
    patient: MultiPatientEntry,
    dataset: str,
    loader: Any,
    streamer: Any,
    max_segments: int | None,
    max_windows: int | None,
) -> tuple[str, Iterable[Any]]:
    if patient.segment_id is not None:
        segment_id = loader.normalize_segment_id(patient.segment_id)
        raw = loader.read_segment(patient_id=patient.patient_id, segment_id=segment_id)
        return segment_id, streamer.stream_segment(raw)

    windows = streamer.stream_patient(
        loader,
        patient_id=patient.patient_id,
        max_segments=max_segments,
        max_windows=max_windows,
    )
    segments = loader.get_patient_segments(patient.patient_id)
    segment_id = segments[0].segment_id if segments else ("ecg" if dataset == "afppgecg" else "00")
    return segment_id, windows


def _make_pipeline(llm_mode: str):
    from agent.proactive.pipeline import ProactivePipeline

    if llm_mode == "env":
        return ProactivePipeline()

    from agent.proactive.llm_judge import ProactiveLLMJudge

    return ProactivePipeline(
        llm_judge=ProactiveLLMJudge(force_enabled=(llm_mode == "live"))
    )


def _af_sensitivity(window_records: list[dict[str, Any]]) -> tuple[float | None, float | None, int, int]:
    truth_af = [record for record in window_records if record.get("truth_label") == "AF"]
    truth_n = [record for record in window_records if record.get("truth_label") == "N"]
    tp_af = sum(1 for record in truth_af if record.get("predicted_label") == "AF")
    tp_n = sum(1 for record in truth_n if record.get("predicted_label") == "N")
    af_sens = tp_af / len(truth_af) if truth_af else None
    n_spec = tp_n / len(truth_n) if truth_n else None
    return af_sens, n_spec, len(truth_af), len(truth_n)


def _print_patient_summary(
    report: Any,
    af_sens: float | None,
    n_spec: float | None,
    n_af: int,
    n_n: int,
    eval_target: str,
) -> None:
    print(
        f"  Windows: {report.windows_processed} "
        f"(truth AF={n_af}, N={n_n}, unknown={report.truth_unknown_window_count})"
    )
    print(f"  AF sensitivity: {_fmt(af_sens, pct=True)}  N specificity: {_fmt(n_spec, pct=True)}")
    if eval_target == "intervention":
        print(
            f"  GT episodes: {report.ground_truth_episode_total}  "
            f"Intervention events: {report.intervention_event_total}"
        )
        print(
            "  Matched / Missed / FalseAlert / Duplicate: "
            f"{report.matched_episode_total} / {report.missed_episode_total} / "
            f"{report.false_alert_total} / {report.duplicate_alert_total}"
        )
    else:
        print(
            f"  GT transitions: {report.ground_truth_transition_total}  "
            f"Predicted: {report.predicted_transition_total}"
        )
    print(f"  P / R / F1: {_fmt(report.precision)} / {_fmt(report.recall)} / {_fmt(report.f1)}")
    if report.latency_windows_mean is not None:
        print(
            f"  Latency: mean={_fmt(report.latency_windows_mean)}w  "
            f"median={_fmt(report.latency_windows_median)}w"
        )


def _print_multi_patient_table(rows: list[dict[str, Any]], eval_target: str) -> None:
    gt_label = "GT ep" if eval_target == "intervention" else "GT trans"
    gt_key = "gt_episodes" if eval_target == "intervention" else "gt_transitions"
    print(f"\n{'Patient':>12}  {'Note':>22}  {'AF sens':>8}  {'N spec':>7}  {gt_label:>9}  {'F1':>6}")
    print("-" * 78)
    for row in rows:
        af_s = f"{float(row['af_sensitivity']) * 100:.1f}%" if row.get("af_sensitivity") else "-"
        n_s = f"{float(row['n_specificity']) * 100:.1f}%" if row.get("n_specificity") else "-"
        gt = row.get(gt_key, "-")
        f1 = row.get("f1", "-")
        pid_label = f"{row['patient_id']}/s{row.get('segment_id', '')}"
        print(f"  {pid_label:>12}  {row['note']:>22}  {af_s:>8}  {n_s:>7}  {str(gt):>9}  {f1:>6}")


def _fmt(value: float | int | None, *, pct: bool = False, decimals: int = 3) -> str:
    if value is None:
        return "-"
    if pct:
        return f"{value * 100:.1f}%"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{decimals}f}"
