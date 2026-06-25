"""
Generate VitalBench dataset.

Supported sources:
    1. icentia11k
       Real Icentia11k ECG segment -> MonitoringState -> VitalBench.

    2. af_ppg_ecg
       AFPPGECG wrist PPG + aligned ECG AF annotation -> MonitoringState
       -> VitalBench.

    3. wesad
       WESAD synchronized subject pickle -> clean 30s/60s stress-aware
       MonitoringState -> VitalBench.

    4. ppg_dalia
       Original UCI PPG-DaLiA full-session subject pickle -> wearable PPG
       ECG-reference-HR/activity-context MonitoringState -> VitalBench.

Examples:
    python scripts/vitalbench/generate_vitalbench.py \
        --source icentia11k \
        --patient-id 00000 \
        --segment-id 00 \
        --duration-sec 1800 \
        --output dataset/vitalbench/full/icentia11k_provisional_qa.jsonl

    python scripts/vitalbench/generate_vitalbench.py \
        --source icentia11k \
        --local-root /path/to/icentia11k-continuous-ecg/1.0 \
        --patient-id 00000 \
        --segment-id 00 \
        --duration-sec 1800

    python scripts/vitalbench/generate_vitalbench.py \
        --source afppgecg \
        --dataset-root dataset/zenodo_af_monitoring_v11242869/raw \
        --patient-id 001 \
        --current-window-sec 300 \
        --ppg-rhythm-threshold 0.5 \
        --ppg-min-annotation-coverage-ratio 0.8 \
        --max-windows 20 \
        --output dataset/vitalbench/full/afppgecg_candidate_qa.jsonl

    python scripts/vitalbench/generate_vitalbench.py \
        --source wesad \
        --wesad-pickle dataset/wesad_sample/WESAD/S11/S11.pkl \
        --wesad-window-secs 30 60 \
        --output dataset/vitalbench/full/wesad_all_qa.jsonl

    python scripts/vitalbench/generate_vitalbench.py \
        --source ppg_dalia \
        --ppg-dalia-pickle dataset/PPG_FieldStudy/S1/S1.pkl \
        --ppg-dalia-window-secs 30 60 \
        --max-windows 20 \
        --output dataset/vitalbench/full/ppg_dalia_all_qa.jsonl

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from pprint import pprint


# ---------------------------------------------------------------------------
# Make repo root importable when running:
#     python scripts/vitalbench/generate_vitalbench.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.benchmarks.vitalbench.state_to_qa_generator import (  # noqa: E402
    generate_qa_from_states,
    summarize_samples,
    write_jsonl,
)


DEFAULT_ICENTIA_OUTPUT = (
    PROJECT_ROOT / "dataset" / "vitalbench" / "full" / "icentia11k_provisional_qa.jsonl"
)
DEFAULT_PPG_OUTPUT = (
    PROJECT_ROOT / "dataset" / "vitalbench" / "full" / "afppgecg_candidate_qa.jsonl"
)
DEFAULT_WESAD_OUTPUT = (
    PROJECT_ROOT / "dataset" / "vitalbench" / "full" / "wesad_all_qa.jsonl"
)
DEFAULT_PPG_DALIA_OUTPUT = (
    PROJECT_ROOT / "dataset" / "vitalbench" / "full" / "ppg_dalia_all_qa.jsonl"
)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate VitalBench JSONL dataset from supported real datasets."
    )

    parser.add_argument(
        "--source",
        choices=["icentia11k", "afppgecg", "wesad", "ppg_dalia"],
        required=True,
        help="Data source to generate from.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. If omitted, a source-specific default path "
            "under dataset/vitalbench/full/ is used."
        ),
    )

    parser.add_argument(
        "--states-output",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for exporting the intermediate MonitoringState "
            "records before QA generation."
        ),
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of generated samples to print as preview. Default: 3.",
    )

    # Icentia11k-specific args
    parser.add_argument(
        "--local-root",
        type=Path,
        default=None,
        help=(
            "Optional local Icentia11k root. If omitted, wfdb will read from "
            "PhysioNet remotely. Only used when --source icentia11k."
        ),
    )

    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Patient id. Required when --source icentia11k or --source afppgecg.",
    )

    parser.add_argument(
        "--segment-id",
        type=str,
        default=None,
        help="Icentia11k segment id. Required when --source icentia11k.",
    )

    parser.add_argument(
        "--sampfrom",
        type=int,
        default=0,
        help="Start sample within the Icentia11k segment. Default: 0.",
    )

    parser.add_argument(
        "--sampto",
        type=int,
        default=None,
        help=(
            "End sample within the Icentia11k segment. If omitted, read to "
            "segment end unless --duration-sec is provided."
        ),
    )

    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help=(
            "Optional duration in seconds. If provided, overrides --sampto as "
            "sampfrom + duration_sec * sampling_rate."
        ),
    )

    parser.add_argument(
        "--current-window-sec",
        type=int,
        default=300,
        help="Current monitoring window length in seconds. Default: 300.",
    )

    parser.add_argument(
        "--previous-window-sec",
        type=int,
        default=300,
        help="Previous comparison window length in seconds. Default: 300.",
    )

    parser.add_argument(
        "--window-stride-sec",
        type=int,
        default=300,
        help="Stride between generated monitoring windows. Default: 300.",
    )

    parser.add_argument(
        "--max-hr-subwindow-sec",
        type=int,
        default=30,
        help="Subwindow length for max HR estimation. Default: 30.",
    )

    parser.add_argument(
        "--rhythm-min-overlap-sec",
        type=float,
        default=2.0,
        help="Minimum rhythm interval overlap used in rhythm aggregation. Default: 2.0.",
    )

    # PPG-specific args
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "AFPPGECG dataset root. Only used when --source afppgecg. "
            "If omitted, the PPG loader default is used."
        ),
    )

    parser.add_argument(
        "--ppg-rhythm-threshold",
        type=float,
        default=0.5,
        help=(
            "AF burden threshold used to map aligned ECG reference annotation "
            "to PPG rhythm labels. Only used when --source afppgecg. Default: 0.5."
        ),
    )

    parser.add_argument(
        "--ppg-ambiguous-rhythm-margin",
        type=float,
        default=None,
        help=(
            "Optional margin around --ppg-rhythm-threshold where PPG rhythm_class "
            "is left unset. Only used when --source afppgecg."
        ),
    )

    parser.add_argument(
        "--ppg-min-annotation-coverage-ratio",
        type=float,
        default=0.8,
        help=(
            "Minimum fraction of a PPG window that must be covered by aligned "
            "ECG reference annotation. Only used when --source afppgecg. Default: 0.8."
        ),
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help=(
            "Maximum number of generated states. For WESAD this applies per "
            "window duration."
        ),
    )

    # WESAD-specific args
    parser.add_argument(
        "--wesad-pickle",
        type=Path,
        default=None,
        help="Path to one WESAD subject pickle, e.g. dataset/wesad_sample/WESAD/S11/S11.pkl.",
    )

    parser.add_argument(
        "--wesad-window-secs",
        type=int,
        nargs="+",
        default=[30, 60],
        help="Clean WESAD window durations in seconds. Default: 30 60.",
    )

    parser.add_argument(
        "--wesad-min-label-fraction",
        type=float,
        default=1.0,
        help="Minimum dominant protocol-label fraction for a clean WESAD window.",
    )

    parser.add_argument(
        "--wesad-window-stride-sec",
        type=int,
        default=None,
        help=(
            "Stride for WESAD clean-window generation. Defaults to each window "
            "duration, i.e. non-overlapping windows."
        ),
    )

    parser.add_argument(
        "--wesad-include-meditation",
        action="store_true",
        help=(
            "Include WESAD label 4 meditation in generated clean states. "
            "By default the core stress/affect subset uses labels 1/2/3."
        ),
    )

    # PPG-DaLiA-specific args. This targets original UCI full-session subject
    # pickles, not the MATLAB activity-split derivative used for quick inspection.
    parser.add_argument(
        "--ppg-dalia-pickle",
        type=Path,
        default=None,
        help="Path to one original UCI PPG-DaLiA subject pickle, e.g. PPG_FieldStudy/S1/S1.pkl.",
    )

    parser.add_argument(
        "--ppg-dalia-window-secs",
        type=int,
        nargs="+",
        default=[30, 60],
        help="Clean PPG-DaLiA window durations in seconds. Default: 30 60.",
    )

    parser.add_argument(
        "--ppg-dalia-window-stride-sec",
        type=int,
        default=None,
        help=(
            "Stride for PPG-DaLiA window generation. Defaults to each window "
            "duration, i.e. non-overlapping windows."
        ),
    )

    parser.add_argument(
        "--ppg-dalia-min-activity-fraction",
        type=float,
        default=0.8,
        help="Minimum dominant protocol-activity fraction for a clean PPG-DaLiA window.",
    )

    parser.add_argument(
        "--ppg-dalia-activity-fs-hz",
        type=float,
        default=None,
        help=(
            "Optional sampling rate for PPG-DaLiA activity labels. If omitted, "
            "the builder infers it from activity length and HR/BVP duration."
        ),
    )

    parser.add_argument(
        "--ppg-dalia-source-variant",
        type=str,
        default="uci_original_pickle",
        help=(
            "PPG-DaLiA source variant recorded in metadata. Default: "
            "uci_original_pickle."
        ),
    )

    parser.add_argument(
        "--ppg-dalia-hr-label-source",
        type=str,
        default="ecg_derived",
        help=(
            "Provenance of HR labels recorded in metadata. Default: ecg_derived."
        ),
    )

    parser.add_argument(
        "--ppg-dalia-reference-signal",
        type=str,
        default="chest_ecg",
        help=(
            "Reference signal recorded in metadata. Default: chest_ecg."
        ),
    )

    parser.add_argument(
        "--ppg-dalia-include-transition-windows",
        action="store_true",
        help=(
            "Include PPG-DaLiA transition/unknown activity windows. By default "
            "only clear protocol activity-context windows are generated."
        ),
    )

    return parser


def resolve_output_path(source: str, output: Path | None) -> Path:
    if output is not None:
        return output

    if source == "icentia11k":
        return DEFAULT_ICENTIA_OUTPUT

    if source == "afppgecg":
        return DEFAULT_PPG_OUTPUT

    if source == "wesad":
        return DEFAULT_WESAD_OUTPUT

    if source == "ppg_dalia":
        return DEFAULT_PPG_DALIA_OUTPUT

    raise ValueError(f"Unsupported source: {source!r}")


def validate_args(args: argparse.Namespace) -> None:
    if args.max_windows is not None and args.max_windows < 0:
        raise ValueError("--max-windows must be non-negative when provided.")

    if args.source == "afppgecg":
        if not args.patient_id:
            raise ValueError("Missing required argument for --source afppgecg: --patient-id")
        if not 0 <= args.ppg_rhythm_threshold <= 1:
            raise ValueError("--ppg-rhythm-threshold must be in [0, 1].")
        if (
            args.ppg_ambiguous_rhythm_margin is not None
            and args.ppg_ambiguous_rhythm_margin < 0
        ):
            raise ValueError("--ppg-ambiguous-rhythm-margin must be non-negative.")
        if not 0 <= args.ppg_min_annotation_coverage_ratio <= 1:
            raise ValueError(
                "--ppg-min-annotation-coverage-ratio must be in [0, 1]."
            )
        return

    if args.source == "wesad":
        if args.wesad_pickle is None:
            raise ValueError("Missing required argument for --source wesad: --wesad-pickle")
        if not args.wesad_window_secs:
            raise ValueError("--wesad-window-secs must contain at least one value.")
        if any(window_sec <= 0 for window_sec in args.wesad_window_secs):
            raise ValueError("--wesad-window-secs values must be positive.")
        if not 0 < args.wesad_min_label_fraction <= 1:
            raise ValueError("--wesad-min-label-fraction must be in (0, 1].")
        if (
            args.wesad_window_stride_sec is not None
            and args.wesad_window_stride_sec <= 0
        ):
            raise ValueError("--wesad-window-stride-sec must be positive when provided.")
        return

    if args.source == "ppg_dalia":
        if args.ppg_dalia_pickle is None:
            raise ValueError(
                "Missing required argument for --source ppg_dalia: "
                "--ppg-dalia-pickle"
            )
        if not args.ppg_dalia_window_secs:
            raise ValueError("--ppg-dalia-window-secs must contain at least one value.")
        if any(window_sec <= 0 for window_sec in args.ppg_dalia_window_secs):
            raise ValueError("--ppg-dalia-window-secs values must be positive.")
        if (
            args.ppg_dalia_window_stride_sec is not None
            and args.ppg_dalia_window_stride_sec <= 0
        ):
            raise ValueError(
                "--ppg-dalia-window-stride-sec must be positive when provided."
            )
        if not 0 < args.ppg_dalia_min_activity_fraction <= 1:
            raise ValueError("--ppg-dalia-min-activity-fraction must be in (0, 1].")
        if (
            args.ppg_dalia_activity_fs_hz is not None
            and args.ppg_dalia_activity_fs_hz <= 0
        ):
            raise ValueError("--ppg-dalia-activity-fs-hz must be positive when provided.")
        return

    if args.source != "icentia11k":
        return

    missing: list[str] = []

    if not args.patient_id:
        missing.append("--patient-id")

    if not args.segment_id:
        missing.append("--segment-id")

    if missing:
        raise ValueError(
            "Missing required arguments for --source icentia11k: "
            + ", ".join(missing)
        )

    if args.sampfrom < 0:
        raise ValueError("--sampfrom must be >= 0.")

    if args.sampto is not None and args.sampto <= args.sampfrom:
        raise ValueError("--sampto must be greater than --sampfrom.")

    if args.duration_sec is not None and args.duration_sec <= 0:
        raise ValueError("--duration-sec must be positive when provided.")


def build_icentia11k_states(args: argparse.Namespace):
    # Imported lazily so other sources do not require wfdb unless needed.
    from agent.data.icentia11k_loader import Icentia11kLoader
    from agent.mhealth.state_builders.icentia11k_state_builder import (
        IcentiaWindowConfig,
        build_monitoring_states_from_icentia_raw,
    )

    loader = Icentia11kLoader(local_root=args.local_root)

    header = loader.read_segment_header(
        patient_id=args.patient_id,
        segment_id=args.segment_id,
    )

    sampto = args.sampto
    if args.duration_sec is not None:
        sampto = args.sampfrom + int(round(args.duration_sec * header.fs))

    raw = loader.read_segment(
        patient_id=args.patient_id,
        segment_id=args.segment_id,
        sampfrom=args.sampfrom,
        sampto=sampto,
    )

    config = IcentiaWindowConfig(
        sampling_rate_hz=raw.fs,
        current_window_sec=args.current_window_sec,
        previous_window_sec=args.previous_window_sec,
        window_stride_sec=args.window_stride_sec,
        max_hr_subwindow_sec=args.max_hr_subwindow_sec,
        rhythm_min_overlap_sec=args.rhythm_min_overlap_sec,
    )

    states = build_monitoring_states_from_icentia_raw(
        raw,
        config=config,
    )

    return states, raw


def build_ppg_states(args: argparse.Namespace):
    from agent.mhealth.state_builders.ppg_state_builder import (
        PPGWindowConfig,
        build_monitoring_states_from_ppg_patient,
    )

    config = PPGWindowConfig(
        window_sec=args.current_window_sec,
        max_hr_subwindow_sec=args.max_hr_subwindow_sec,
        max_windows=args.max_windows,
        rhythm_threshold=args.ppg_rhythm_threshold,
        ambiguous_rhythm_margin=args.ppg_ambiguous_rhythm_margin,
        min_annotation_coverage_ratio=args.ppg_min_annotation_coverage_ratio,
    )

    states = build_monitoring_states_from_ppg_patient(
        dataset_root=args.dataset_root,
        patient_id=args.patient_id,
        config=config,
    )

    return states


def build_wesad_states(args: argparse.Namespace):
    from agent.mhealth.state_builders.wesad_state_builder import (
        WESADWindowConfig,
        build_monitoring_states_from_wesad_subject,
    )

    allowed_labels = (1, 2, 3, 4) if args.wesad_include_meditation else (1, 2, 3)

    config = WESADWindowConfig(
        window_secs=tuple(args.wesad_window_secs),
        window_stride_sec=args.wesad_window_stride_sec,
        allowed_protocol_label_ids=allowed_labels,
        min_label_fraction=args.wesad_min_label_fraction,
        max_windows_per_duration=args.max_windows,
    )

    return build_monitoring_states_from_wesad_subject(
        args.wesad_pickle,
        config=config,
    )


def build_ppg_dalia_states(args: argparse.Namespace):
    from agent.mhealth.state_builders.ppg_dalia_state_builder import (
        PPGDaLiAWindowConfig,
        build_monitoring_states_from_ppg_dalia_subject,
    )

    config = PPGDaLiAWindowConfig(
        window_secs=tuple(args.ppg_dalia_window_secs),
        window_stride_sec=args.ppg_dalia_window_stride_sec,
        activity_label_fs_hz=args.ppg_dalia_activity_fs_hz,
        source_variant=args.ppg_dalia_source_variant,
        hr_label_source=args.ppg_dalia_hr_label_source,
        reference_signal=args.ppg_dalia_reference_signal,
        min_activity_fraction=args.ppg_dalia_min_activity_fraction,
        max_windows_per_duration=args.max_windows,
        include_transition_windows=args.ppg_dalia_include_transition_windows,
    )

    return build_monitoring_states_from_ppg_dalia_subject(
        args.ppg_dalia_pickle,
        config=config,
    )


def print_preview(samples, n: int) -> None:
    """Print a few generated QA samples for inspection."""
    if n <= 0:
        return

    print()
    print("=" * 80)
    print(f"Preview: first {min(n, len(samples))} samples")
    print("=" * 80)

    for sample in samples[:n]:
        record = sample.to_jsonl_record()
        print()
        print(json.dumps(record, ensure_ascii=False, indent=2))


def write_states_jsonl(states, output_path: Path) -> None:
    """Write MonitoringState objects as one JSON record per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for state in states:
            handle.write(json.dumps(state.to_dict(), ensure_ascii=False))
            handle.write("\n")


def print_run_header(
    *,
    source: str,
    states_count: int,
    samples_count: int,
    output_path: Path,
    raw=None,
) -> None:
    print()
    print("=" * 80)
    print("VitalBench generation complete")
    print("=" * 80)

    print(f"Source      : {source}")
    print(f"Input states: {states_count}")
    print(f"QA samples  : {samples_count}")
    print(f"Output path : {output_path}")

    if raw is not None:
        print()
        print("Icentia11k source segment")
        print("-" * 80)
        print(f"Patient ID  : {raw.patient_id}")
        print(f"Segment ID  : {raw.segment_id}")
        print(f"Raw samples : {raw.n_samples}")
        print(f"Sampling Hz : {raw.fs}")
        print(f"Duration    : {raw.n_samples / raw.fs:.1f}s")
        print(f"Sample range: [{raw.sample_start}, {raw.sample_end})")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    output_path = resolve_output_path(args.source, args.output)

    raw = None

    if args.source == "icentia11k":
        states, raw = build_icentia11k_states(args)

    elif args.source == "afppgecg":
        states = build_ppg_states(args)

    elif args.source == "wesad":
        states = build_wesad_states(args)

    elif args.source == "ppg_dalia":
        states = build_ppg_dalia_states(args)

    else:
        raise ValueError(f"Unsupported source: {args.source!r}")

    if args.states_output is not None:
        write_states_jsonl(states, args.states_output)

    samples = generate_qa_from_states(states)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(samples, output_path)

    print_run_header(
        source=args.source,
        states_count=len(states),
        samples_count=len(samples),
        output_path=output_path,
        raw=raw,
    )

    print()
    print("=" * 80)
    print("Dataset summary")
    print("=" * 80)
    pprint(summarize_samples(samples), sort_dicts=False)

    print_preview(samples, args.preview)


if __name__ == "__main__":
    main()

