"""Command-line entry point and dataset selection for proactive evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.evaluation.proactive.config import (
    default_afppgecg_dataset_root,
    default_icentia11k_dataset_root,
)
from agent.evaluation.proactive.multi_patient import (
    discover_icentia11k_patient_ids,
    run_multi_patient_evaluation,
)
from agent.evaluation.proactive.metrics import (
    InterventionReplayReport,
    TransitionEvalReport,
    evaluate_intervention_replay,
    evaluate_transition_detection,
)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate proactive rhythm-transition detection on ECG windows."
    )
    parser.add_argument(
        "--dataset",
        choices=("icentia11k", "afppgecg"),
        default="icentia11k",
        help="Dataset adapter to use.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Local dataset root. Defaults to dataset/icentia11k_subset/1.0 for "
            "Icentia11k or the configured AFPPGECG dataset root."
        ),
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help=(
            "Optional patient id, for example 12 or 00012. If omitted, "
            "each discoverable patient is evaluated separately."
        ),
    )
    parser.add_argument(
        "--segment-id",
        type=str,
        default=None,
        help="Optional segment id. If omitted, the first available segment is used.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=1,
        help="How many segments to stream when --segment-id is omitted.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Maximum number of windows to evaluate. Defaults to no window limit.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=10.0,
        help="Window length passed to ECGStreamer.",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=None,
        help="Stride passed to ECGStreamer. Defaults to non-overlapping windows.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the aggregate report JSON.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional path for per-window/per-event debug records.",
    )
    parser.add_argument(
        "--debounce",
        type=int,
        default=3,
        help="Minimum consecutive windows before confirming a rhythm transition (0=disabled).",
    )
    parser.add_argument(
        "--eval-target",
        choices=("transition", "intervention"),
        default="transition",
        help="Evaluate rhythm transitions or emitted proactive interventions.",
    )
    parser.add_argument(
        "--pre-onset-tolerance-windows",
        type=int,
        default=0,
        help=(
            "For intervention replay, count an alert this many windows before "
            "episode onset as a hit."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for per-patient reports and multi_patient_summary.csv. "
            "Defaults to results/proactive when --patient-id is omitted."
        ),
    )
    parser.add_argument(
        "--llm-mode",
        choices=("env", "live", "disabled"),
        default="env",
        help=(
            "LLM judge mode. 'env' respects LLM_PROACTIVE_ENABLED, 'live' forces "
            "the LLM judge on, and 'disabled' runs rule-only."
        ),
    )
    return parser


def _select_windows(args: argparse.Namespace):
    from agent.data.streaming import ECGStreamer

    if args.patient_id is None:
        raise ValueError("Single-patient evaluation requires --patient-id.")

    if args.dataset == "afppgecg":
        from agent.data.afppgecg_adapter import AFPPGECGSegmentLoader

        loader = AFPPGECGSegmentLoader(dataset_root=args.dataset_root)
        streamer = ECGStreamer(
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
        )
        patient_id = loader.normalize_patient_id(args.patient_id)

        if args.segment_id is not None:
            segment_id = loader.normalize_segment_id(args.segment_id)
            segments = loader.get_patient_segments(patient_id)
            sampto = None
            if args.max_windows is not None and segments:
                sampto = _sampto_for_window_limit(
                    max_windows=args.max_windows,
                    window_seconds=args.window_seconds,
                    stride_seconds=args.stride_seconds,
                    fs=segments[0].fs,
                    segment_total_samples=segments[0].n_samples,
                )
                if sampto <= 0:
                    return patient_id, segment_id, iter(())
            raw = loader.read_segment(
                patient_id=patient_id,
                segment_id=segment_id,
                sampto=sampto,
            )
            return patient_id, segment_id, streamer.stream_segment(raw)

        segments = loader.get_patient_segments(patient_id)
        first_segment = segments[0].segment_id if segments else "unknown"
        if args.max_segments is not None and args.max_segments <= 0:
            return patient_id, first_segment, iter(())
        if args.max_windows is not None and segments:
            sampfrom = 0
            if args.eval_target == "transition":
                sampfrom = _afppgecg_transition_context_start_sample(
                    loader=loader,
                    patient_id=patient_id,
                    max_windows=args.max_windows,
                    window_seconds=args.window_seconds,
                    stride_seconds=args.stride_seconds,
                    fs=segments[0].fs,
                )
            sampto = _sampto_for_window_limit(
                max_windows=args.max_windows,
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
                fs=segments[0].fs,
                segment_total_samples=segments[0].n_samples,
                sampfrom=sampfrom,
            )
            if sampto <= 0:
                return patient_id, first_segment, iter(())
            raw = loader.read_segment(
                patient_id=patient_id,
                segment_id=first_segment,
                sampfrom=sampfrom,
                sampto=sampto,
            )
            return (
                patient_id,
                first_segment,
                streamer.stream_segment(raw, segment_time_offset_s=sampfrom / segments[0].fs),
            )

        windows = streamer.stream_patient(
            loader,
            patient_id=patient_id,
            max_segments=args.max_segments,
            max_windows=args.max_windows,
        )
        return patient_id, first_segment, windows

    from agent.data.icentia11k_loader import Icentia11kLoader

    loader = Icentia11kLoader(local_root=args.dataset_root)
    streamer = ECGStreamer(
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )
    patient_id = loader.normalize_patient_id(args.patient_id)

    if args.segment_id is not None:
        segment_id = loader.normalize_segment_id(args.segment_id)
        raw = loader.read_segment(patient_id=patient_id, segment_id=segment_id)
        return patient_id, segment_id, streamer.stream_segment(raw)

    windows = streamer.stream_patient(
        loader,
        patient_id=patient_id,
        max_segments=args.max_segments,
        max_windows=args.max_windows,
    )
    segments = loader.get_patient_segments(patient_id)
    first_segment = segments[0].segment_id if segments else "unknown"
    return patient_id, first_segment, windows


def _discover_icentia11k_patient_ids(dataset_root: Path) -> list[str]:
    return discover_icentia11k_patient_ids(dataset_root)


def _sampto_for_window_limit(
    *,
    max_windows: int,
    window_seconds: float,
    stride_seconds: float | None,
    fs: int,
    segment_total_samples: int,
    sampfrom: int = 0,
) -> int:
    window_samples = int(round(float(window_seconds) * int(fs)))
    stride_samples = int(
        round(float(stride_seconds if stride_seconds is not None else window_seconds) * int(fs))
    )
    if max_windows <= 0 or window_samples <= 0 or stride_samples <= 0:
        return 0
    requested_samples = window_samples + (int(max_windows) - 1) * stride_samples
    return min(int(segment_total_samples), int(sampfrom) + requested_samples)


def _afppgecg_transition_context_start_sample(
    *,
    loader: Any,
    patient_id: str,
    max_windows: int,
    window_seconds: float,
    stride_seconds: float | None,
    fs: int,
) -> int:
    from agent.data.afppgecg_adapter import build_afppgecg_rhythm_change_annotation

    record = loader.ppg_loader.read_ecg_record(patient_id, include_signal=False)
    changes = build_afppgecg_rhythm_change_annotation(
        qrs_index=record.qrs_index,
        af_annotation=record.af_annotation,
    )
    if changes is None or len(changes.sample) < 2:
        return 0

    stride_s = stride_seconds if stride_seconds is not None else window_seconds
    stride_samples = int(round(float(stride_s) * int(fs)))
    if stride_samples <= 0:
        return 0

    pre_context_windows = min(max(int(max_windows) // 4, 1), 50)
    first_transition_sample = int(changes.sample[1])
    return max(0, first_transition_sample - pre_context_windows * stride_samples)


def _save_report_json(report: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def _save_debug_jsonl(artifacts: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in artifacts.iter_debug_records():
            handle.write(json.dumps(record) + "\n")


def _print_report(report: TransitionEvalReport) -> None:
    print(f"Windows processed: {report.windows_processed}")
    print(f"GT transitions: {report.ground_truth_transition_total}")
    print(f"Predicted transitions: {report.predicted_transition_total}")
    print(
        f"Precision / Recall / F1: "
        f"{report.precision:.3f} / {report.recall:.3f} / {report.f1:.3f}"
    )
    print(
        f"Latency (windows): mean={_fmt(report.latency_windows_mean)} "
        f"median={_fmt(report.latency_windows_median)} "
        f"p90={_fmt(report.latency_windows_p90)} "
        f"max={_fmt(report.latency_windows_max)}"
    )
    print(
        f"Latency (seconds): mean={_fmt(report.latency_seconds_mean)} "
        f"median={_fmt(report.latency_seconds_median)} "
        f"p90={_fmt(report.latency_seconds_p90)} "
        f"max={_fmt(report.latency_seconds_max)}"
    )
    print(
        f"Unknown windows: truth={report.truth_unknown_window_count} "
        f"predicted={report.predicted_unknown_window_count} "
        f"skipped_gt_pairs={report.skipped_ground_truth_transition_count}"
    )

    print("Top transition pairs:")
    for pair, stats in sorted(
        report.per_transition_pair.items(),
        key=lambda item: item[1]["ground_truth_total"],
        reverse=True,
    )[:10]:
        print(
            f"  {pair}: gt={stats['ground_truth_total']} pred={stats['predicted_total']} "
            f"matched={stats['matched_total']} f1={stats['f1']:.3f}"
        )

    print("Unknown raw annotation labels:")
    if report.unknown_raw_label_counts:
        for label, count in report.unknown_raw_label_counts.items():
            print(f"  {label}: {count}")
    else:
        print("  none")


def _print_intervention_report(report: InterventionReplayReport) -> None:
    print(f"Windows processed: {report.windows_processed}")
    print(f"GT abnormal episodes: {report.ground_truth_episode_total}")
    print(f"Intervention events: {report.intervention_event_total}")
    print(
        f"Matched / Missed: "
        f"{report.matched_episode_total} / {report.missed_episode_total}"
    )
    print(
        f"False alerts / Duplicate alerts: "
        f"{report.false_alert_total} / {report.duplicate_alert_total}"
    )
    print(
        f"Precision / Recall / F1: "
        f"{report.precision:.3f} / {report.recall:.3f} / {report.f1:.3f}"
    )
    print(
        f"Latency (windows): mean={_fmt(report.latency_windows_mean)} "
        f"median={_fmt(report.latency_windows_median)} "
        f"p90={_fmt(report.latency_windows_p90)} "
        f"max={_fmt(report.latency_windows_max)}"
    )
    print(
        f"Latency (seconds): mean={_fmt(report.latency_seconds_mean)} "
        f"median={_fmt(report.latency_seconds_median)} "
        f"p90={_fmt(report.latency_seconds_p90)} "
        f"max={_fmt(report.latency_seconds_max)}"
    )
    print(f"Unknown truth windows: {report.truth_unknown_window_count}")

    print("Episode labels:")
    if report.per_label:
        for label, stats in sorted(report.per_label.items()):
            print(
                f"  {label}: gt={stats['ground_truth_total']} "
                f"matched={stats['matched_total']} "
                f"missed={stats['missed_total']} "
                f"duplicates={stats['duplicate_alert_total']} "
                f"recall={stats['recall']:.3f}"
            )
    else:
        print("  none")

    print("Alert rules:")
    if report.per_rule:
        for rule, stats in sorted(report.per_rule.items()):
            print(
                f"  {rule}: alerts={stats['alert_total']} "
                f"matched={stats['matched_episode_total']} "
                f"false={stats['false_alert_total']} "
                f"duplicates={stats['duplicate_alert_total']} "
                f"precision={stats['precision']:.3f}"
            )
    else:
        print("  none")

    print("Unknown raw annotation labels:")
    if report.unknown_raw_label_counts:
        for label, count in report.unknown_raw_label_counts.items():
            print(f"  {label}: {count}")
    else:
        print("  none")


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def _make_pipeline(llm_mode: str):
    from agent.proactive.pipeline import ProactivePipeline

    if llm_mode == "env":
        return ProactivePipeline()

    from agent.proactive.llm_judge import ProactiveLLMJudge

    return ProactivePipeline(
        llm_judge=ProactiveLLMJudge(force_enabled=(llm_mode == "live"))
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    multi_patient_mode = args.patient_id is None or args.output_dir is not None
    if args.output_dir is not None and args.output_json is not None:
        parser.error("--output-dir cannot be combined with --output-json.")
    if args.output_dir is not None and args.output_jsonl is not None:
        parser.error("--output-dir cannot be combined with --output-jsonl.")
    if args.patient_id is None and args.output_json is not None:
        parser.error("--output-json requires --patient-id. Use --output-dir for all-patient evaluation.")
    if args.patient_id is None and args.output_jsonl is not None:
        parser.error("--output-jsonl requires --patient-id. Use --output-dir for all-patient evaluation.")

    if args.dataset_root is None:
        args.dataset_root = (
            default_afppgecg_dataset_root()
            if args.dataset == "afppgecg"
            else default_icentia11k_dataset_root()
        )

    if not args.dataset_root.exists():
        parser.error(f"Dataset root does not exist: {args.dataset_root}")

    try:
        if multi_patient_mode:
            return run_multi_patient_evaluation(args)
        pipeline = _make_pipeline(args.llm_mode)
    except ModuleNotFoundError as exc:
        print(f"Missing dependency while importing proactive pipeline: {exc}")
        print(
            "Try running with: "
            "uv run --with wfdb --with neurokit2 python -m agent.evaluation.proactive"
        )
        return 1

    patient_id, first_segment, windows = _select_windows(args)
    limited_windows = windows
    if args.max_windows is not None:
        limited_windows = (
            window for index, window in enumerate(windows)
            if index < args.max_windows
        )

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

    print(f"Patient: {patient_id}")
    print(f"First segment: {first_segment}")
    if args.eval_target == "intervention":
        _print_intervention_report(artifacts.report)
    else:
        _print_report(artifacts.report)

    if args.output_json is not None:
        _save_report_json(artifacts.report, args.output_json)
        print(f"Saved report JSON to {args.output_json}")
    if args.output_jsonl is not None:
        _save_debug_jsonl(artifacts, args.output_jsonl)
        print(f"Saved debug JSONL to {args.output_jsonl}")

    return 0

