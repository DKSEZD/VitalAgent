from __future__ import annotations

import pytest

from scripts.vitalbench import generate_vitalbench


def test_ppg_generation_args_expose_reproducible_builder_parameters() -> None:
    args = generate_vitalbench.build_parser().parse_args(
        [
            "--source",
            "afppgecg",
            "--patient-id",
            "001",
            "--ppg-rhythm-threshold",
            "0.4",
            "--ppg-ambiguous-rhythm-margin",
            "0.05",
            "--ppg-min-annotation-coverage-ratio",
            "0.9",
        ]
    )

    generate_vitalbench.validate_args(args)

    assert args.ppg_rhythm_threshold == pytest.approx(0.4)
    assert args.ppg_ambiguous_rhythm_margin == pytest.approx(0.05)
    assert args.ppg_min_annotation_coverage_ratio == pytest.approx(0.9)


def test_ppg_generation_args_validate_annotation_coverage_ratio() -> None:
    args = generate_vitalbench.build_parser().parse_args(
        [
            "--source",
            "afppgecg",
            "--patient-id",
            "001",
            "--ppg-min-annotation-coverage-ratio",
            "1.1",
        ]
    )

    with pytest.raises(ValueError, match="ppg-min-annotation-coverage-ratio"):
        generate_vitalbench.validate_args(args)
