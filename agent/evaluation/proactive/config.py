"""Configuration helpers for proactive evaluation."""

from __future__ import annotations

from pathlib import Path

from agent.config import get_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def default_icentia11k_dataset_root() -> Path:
    return get_config().datasets.icentia11k_root


def default_afppgecg_dataset_root() -> Path:
    return get_config().datasets.afppgecg_root
