"""Build the release VitalBench benchmark subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from agent.mhealth.schemas import canonicalize_dataset_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "vitalbench"
FULL_ROOT = DATASET_ROOT / "full"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "eval_subsets"
SEED = "42"
SPLIT_SEED = "dev_test_split_v3"

STRATUM_LIMITS: "OrderedDict[tuple[str, str], int]" = OrderedDict(
    [
        (("afppgecg", "A"), 278),
        (("afppgecg", "B"), 278),
        (("icentia11k", "A"), 278),
        (("icentia11k", "B"), 278),
        (("ppg_dalia", "A"), 200),
        (("ppg_dalia", "B"), 200),
        (("wesad", "A"), 150),
        (("wesad", "B"), 200),
    ]
)

def _canonical_or_legacy_path(
    full_root: Path,
    canonical_name: str,
    legacy_name: str,
) -> Path:
    canonical_path = full_root / canonical_name
    return canonical_path if canonical_path.exists() else full_root / legacy_name


def source_qa_paths(full_root: Path) -> dict[str, Path]:
    return {
        "afppgecg": _canonical_or_legacy_path(
            full_root,
            "afppgecg_candidate_qa.jsonl",
            "af_ppg_ecg_candidate_qa.jsonl",
        ),
        "icentia11k": full_root / "icentia11k_provisional_qa.jsonl",
        "ppg_dalia": full_root / "ppg_dalia_all_qa.jsonl",
        "wesad": full_root / "wesad_all_qa.jsonl",
    }


def source_state_paths(full_root: Path) -> dict[str, Path]:
    return {
        "afppgecg": _canonical_or_legacy_path(
            full_root,
            "afppgecg_candidate_states.jsonl",
            "af_ppg_ecg_candidate_states.jsonl",
        ),
        "icentia11k": full_root / "icentia11k_provisional_states.jsonl",
        "ppg_dalia": full_root / "ppg_dalia_all_states.jsonl",
        "wesad": full_root / "wesad_all_states.jsonl",
    }


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(f"{value}\n")


def stable_key(seed: str, question_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()
    return digest, question_id


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def count_by(field_name: str) -> dict[str, int]:
        return dict(
            sorted(Counter(str(record.get(field_name)) for record in records).items())
        )

    return {
        "sample_count": len(records),
        "by_dataset": count_by("dataset"),
        "by_tier": count_by("tier"),
        "by_question_type": count_by("question_type"),
        "by_target": count_by("target"),
        "by_template_id": count_by("template_id"),
    }


def select_stratified_by_target(
    records: list[dict[str, Any]],
    *,
    seed: str,
    max_samples: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("target"))].append(record)

    if not groups or max_samples <= 0:
        return []

    targets = sorted(groups)
    base = max_samples // len(targets)
    remainder = max_samples % len(targets)
    selected: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        quota = base + (1 if index < remainder else 0)
        if quota <= 0:
            continue
        ordered = sorted(
            groups[target],
            key=lambda record: stable_key(seed, str(record["question_id"])),
        )
        selected.extend(ordered[:quota])
    return selected


def collect_stratum_candidates(
    source_path: Path,
    *,
    dataset: str,
    tier: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in iter_jsonl(source_path):
        if canonicalize_dataset_name(str(record.get("dataset"))) != dataset:
            continue
        if str(record.get("tier")).upper() != tier.upper():
            continue
        candidates.append({**record, "dataset": dataset})
    return candidates


def create_parts(
    output_root: Path,
    *,
    source_qa: dict[str, Path],
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    parts_root = output_root / "parts"
    if parts_root.exists():
        shutil.rmtree(parts_root)

    parts: list[Path] = []
    stats: dict[str, dict[str, Any]] = {}

    for (dataset, tier), cap in STRATUM_LIMITS.items():
        stratum = f"{dataset}:{tier}"
        if cap == 0:
            stats[stratum] = {
                "requested": cap,
                "actual": 0,
                "candidate_count": 0,
                "part_jsonl": None,
                "skipped": True,
            }
            print(f"SKIP: stratum {stratum} requested=0; no part file generated")
            continue

        candidates = collect_stratum_candidates(
            source_qa[dataset],
            dataset=dataset,
            tier=tier,
        )
        selected = select_stratified_by_target(candidates, seed=SEED, max_samples=cap)
        actual = len(selected)
        if actual < cap:
            print(f"WARN: stratum {stratum} capped at ACTUAL={actual}, requested={cap}")

        part_dir = output_root / "parts" / dataset
        part_path = part_dir / f"{dataset}_{tier}_n{cap}_seed{SEED}.jsonl"
        qids_path = part_path.with_suffix(".question_ids.txt")
        manifest_path = part_path.with_suffix(".manifest.json")

        write_jsonl(part_path, selected)
        write_text_lines(qids_path, (str(record["question_id"]) for record in selected))
        write_json(
            manifest_path,
            {
                "version": "vitalbench_release_part",
                "created_at": utc_now(),
                "pool_version": "release",
                "source_qa_jsonl": rel(source_qa[dataset]),
                "output_jsonl": rel(part_path),
                "question_ids_output": rel(qids_path),
                "selection": {
                    "method": (
                        "per-stratum sha256(seed:question_id) ascending "
                        "stratified by target"
                    ),
                    "seed": SEED,
                    "dataset": dataset,
                    "tier": tier,
                    "requested_cap": cap,
                    "actual_samples": actual,
                    "stratify_by": "target",
                },
                "pool_summary": summarize(candidates),
                "subset_summary": summarize(selected),
            },
        )

        parts.append(part_path)
        stats[stratum] = {
            "requested": cap,
            "actual": actual,
            "candidate_count": len(candidates),
            "part_jsonl": rel(part_path),
            "skipped": False,
        }
        print(
            f"PART: {stratum:<16} requested={cap:>3} "
            f"actual={actual:>3} candidates={len(candidates)}"
        )

    return parts, stats


def referenced_state_ids(qa_records: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(record["source_state_id"])
        for record in qa_records
        if record.get("source_state_id")
    }


def load_referenced_states(
    qa_records: list[dict[str, Any]],
    *,
    source_states: dict[str, Path],
) -> list[dict[str, Any]]:
    needed = referenced_state_ids(qa_records)
    states: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dataset in source_states:
        for state in iter_jsonl(source_states[dataset]):
            state_id = state.get("state_id") or state.get("id")
            state_id_text = str(state_id) if state_id else ""
            if not state_id_text or state_id_text not in needed or state_id_text in seen:
                continue
            states.append(
                {
                    **state,
                    "dataset": canonicalize_dataset_name(str(state.get("dataset", dataset))),
                }
            )
            seen.add(state_id_text)
    missing = needed - seen
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise ValueError(f"Missing {len(missing)} referenced states: {preview}")
    return states


def combine_parts(
    output_root: Path,
    parts: list[Path],
    stratum_stats: dict[str, dict[str, Any]],
    *,
    source_states: dict[str, Path],
) -> Path:
    final_dir = output_root / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    qa_records: list[dict[str, Any]] = []
    for part in parts:
        qa_records.extend(read_jsonl(part))
    states = load_referenced_states(qa_records, source_states=source_states)

    qa_path = final_dir / "qa.jsonl"
    states_path = final_dir / "states.jsonl"
    qids_path = final_dir / "question_ids.txt"
    manifest_path = final_dir / "manifest.json"

    write_jsonl(qa_path, qa_records)
    write_jsonl(states_path, states)
    write_text_lines(qids_path, (str(record["question_id"]) for record in qa_records))

    write_json(
        manifest_path,
        {
            "version": "vitalbench_release",
            "created_at": utc_now(),
            "pool_version": "release",
            "output_jsonl": rel(qa_path),
            "states_output": rel(states_path),
            "question_ids_output": rel(qids_path),
            "selection": {
                "seed": SEED,
                "method": (
                    "per-dataset per-tier sha256(seed:question_id) ascending "
                    "stratified by target with non-uniform release caps"
                ),
                "datasets": list(dict.fromkeys(dataset for dataset, _ in STRATUM_LIMITS)),
                "samples_per_stratum": {
                    key: value["actual"] for key, value in stratum_stats.items()
                },
                "requested_samples_per_stratum": {
                    key: value["requested"] for key, value in stratum_stats.items()
                },
                "shortfalls": {
                    key: {
                        "requested": value["requested"],
                        "actual": value["actual"],
                    }
                    for key, value in stratum_stats.items()
                    if value["actual"] < value["requested"]
                },
                "release_name": "VitalBench",
            },
            "parts": [rel(part) for part in parts],
            "sample_count": len(qa_records),
            "unique_state_count": len(referenced_state_ids(qa_records)),
        },
    )

    print(f"COMBINE: wrote {len(qa_records)} QA rows and {len(states)} states")
    return final_dir


def stratum_key(record: dict[str, Any]) -> str:
    return f"{record['dataset']}:{record['tier']}"


def split_records(
    qa_records: list[dict[str, Any]],
    *,
    dev_ratio: float,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in qa_records:
        by_stratum[stratum_key(record)].append(record)

    dev_ids: set[str] = set()
    strata: dict[str, dict[str, int]] = {}
    for stratum in sorted(by_stratum):
        records = by_stratum[stratum]
        target_dev_count = int(len(records) * dev_ratio)
        ordered = sorted(
            records,
            key=lambda record: stable_key(seed, str(record["question_id"])),
        )
        selected = ordered[:target_dev_count]
        dev_ids.update(str(record["question_id"]) for record in selected)
        strata[stratum] = {
            "total": len(records),
            "target_dev_count": target_dev_count,
            "dev_count": len(selected),
            "test_count": len(records) - len(selected),
        }

    dev_records = [
        record for record in qa_records if str(record["question_id"]) in dev_ids
    ]
    test_records = [
        record for record in qa_records if str(record["question_id"]) not in dev_ids
    ]
    return dev_records, test_records, strata


def composition(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset_tier = Counter(stratum_key(record) for record in records)
    by_tier = Counter(str(record["tier"]) for record in records)
    return {
        "sample_count": len(records),
        "by_tier": dict(sorted(by_tier.items())),
        "by_dataset_tier": dict(sorted(by_dataset_tier.items())),
    }


def write_split_dir(
    split_dir: Path,
    qa_records: list[dict[str, Any]],
    all_states: list[dict[str, Any]],
) -> None:
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    state_ids = referenced_state_ids(qa_records)
    split_states = [
        state
        for state in all_states
        if (state.get("state_id") or state.get("id")) in state_ids
    ]
    write_jsonl(split_dir / "qa.jsonl", qa_records)
    write_jsonl(split_dir / "states.jsonl", split_states)
    write_text_lines(
        split_dir / "question_ids.txt",
        (str(record["question_id"]) for record in qa_records),
    )


def build_split_manifest(
    final_dir: Path,
    *,
    dev_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    strata: dict[str, dict[str, int]],
    dev_ratio: float,
) -> dict[str, Any]:
    qa_path = final_dir / "qa.jsonl"
    states_path = final_dir / "states.jsonl"
    source_manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        "version": "vitalbench_release_dev_test_split",
        "created_at": utc_now(),
        "source_manifest": source_manifest,
        "source_qa_jsonl": rel(qa_path),
        "source_states_jsonl": rel(states_path),
        "dev_ratio": dev_ratio,
        "seed": SPLIT_SEED,
        "method": (
            "stratified by dataset+tier; dev count is floor(stratum_size * "
            "dev_ratio); dev slots filled by sha256(seed:question_id) ascending"
        ),
        "splits": {
            "dev": {
                **composition(dev_records),
                "qa_jsonl": rel(final_dir / "dev" / "qa.jsonl"),
                "states_jsonl": rel(final_dir / "dev" / "states.jsonl"),
                "question_ids": rel(final_dir / "dev" / "question_ids.txt"),
                "unique_state_count": len(referenced_state_ids(dev_records)),
            },
            "test": {
                **composition(test_records),
                "qa_jsonl": rel(final_dir / "test" / "qa.jsonl"),
                "states_jsonl": rel(final_dir / "test" / "states.jsonl"),
                "question_ids": rel(final_dir / "test" / "question_ids.txt"),
                "unique_state_count": len(referenced_state_ids(test_records)),
            },
        },
        "strata": strata,
    }


def write_split_manifests(final_dir: Path, manifest: dict[str, Any]) -> None:
    for split_name in ("dev", "test"):
        split_manifest = {**manifest, "active_split": split_name}
        write_json(final_dir / split_name / "manifest.json", split_manifest)
    write_json(final_dir / "split_manifest.json", manifest)


def create_dev_test_split(final_dir: Path, *, dev_ratio: float) -> dict[str, Any]:
    qa_records = read_jsonl(final_dir / "qa.jsonl")
    states = read_jsonl(final_dir / "states.jsonl")
    dev_records, test_records, strata = split_records(
        qa_records,
        dev_ratio=dev_ratio,
        seed=SPLIT_SEED,
    )
    write_split_dir(final_dir / "dev", dev_records, states)
    write_split_dir(final_dir / "test", test_records, states)
    manifest = build_split_manifest(
        final_dir,
        dev_records=dev_records,
        test_records=test_records,
        strata=strata,
        dev_ratio=dev_ratio,
    )
    write_split_manifests(final_dir, manifest)
    print(
        f"SPLIT: dev={len(dev_records)} test={len(test_records)} "
        f"total={len(qa_records)}"
    )
    return manifest


def verify_outputs(
    final_dir: Path,
    *,
    stratum_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    final_records = read_jsonl(final_dir / "qa.jsonl")
    dev_records = read_jsonl(final_dir / "dev" / "qa.jsonl")
    test_records = read_jsonl(final_dir / "test" / "qa.jsonl")
    actual_by_stratum = Counter(stratum_key(record) for record in final_records)
    dev_by_stratum = Counter(stratum_key(record) for record in dev_records)
    test_by_stratum = Counter(stratum_key(record) for record in test_records)

    rows = []
    for (dataset, tier), cap in STRATUM_LIMITS.items():
        key = f"{dataset}:{tier}"
        total = actual_by_stratum.get(key, 0)
        expected = stratum_stats[key]["actual"]
        if total != expected:
            raise ValueError(f"Count mismatch for {key}: final={total}, part={expected}")
        rows.append(
            {
                "stratum": key,
                "requested": cap,
                "total": total,
                "dev": dev_by_stratum.get(key, 0),
                "test": test_by_stratum.get(key, 0),
            }
        )

    summary = {
        "rows": rows,
        "total": len(final_records),
        "dev_total": len(dev_records),
        "test_total": len(test_records),
        "shortfalls": {
            key: {
                "requested": value["requested"],
                "actual": value["actual"],
            }
            for key, value in stratum_stats.items()
            if value["actual"] < value["requested"]
        },
    }
    write_json(final_dir / "verification_summary.json", summary)
    return summary


def print_final_summary(summary: dict[str, Any]) -> None:
    print()
    print("# VitalBench release summary")
    print("stratum                  dev  test  total  requested")
    for row in summary["rows"]:
        print(
            f"{row['stratum']:<22} "
            f"{row['dev']:>4} {row['test']:>5} {row['total']:>6} {row['requested']:>10}"
        )
    print()
    print(f"Total dev : {summary['dev_total']}")
    print(f"Total test: {summary['test_total']}")
    print(f"Total     : {summary['total']}")
    if summary["shortfalls"]:
        print("Shortfalls:")
        for key, value in summary["shortfalls"].items():
            print(f"- {key}: actual={value['actual']} requested={value['requested']}")
    else:
        print("Shortfalls: none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--full-root", type=Path, default=FULL_ROOT)
    parser.add_argument("--dev-ratio", type=float, default=0.30)
    args = parser.parse_args()

    if not 0 <= args.dev_ratio <= 1:
        raise ValueError("--dev-ratio must be in [0, 1].")

    source_qa = source_qa_paths(args.full_root)
    source_states = source_state_paths(args.full_root)

    parts, stratum_stats = create_parts(args.output_root, source_qa=source_qa)
    final_dir = combine_parts(
        args.output_root,
        parts,
        stratum_stats,
        source_states=source_states,
    )
    create_dev_test_split(final_dir, dev_ratio=args.dev_ratio)
    summary = verify_outputs(
        final_dir,
        stratum_stats=stratum_stats,
    )
    print_final_summary(summary)


if __name__ == "__main__":
    main()
