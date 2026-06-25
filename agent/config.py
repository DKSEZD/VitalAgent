"""
Global configuration for VitalAgent.

All settings are loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Auto-load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LLMRole = Literal["agent", "planner", "proactive"]
_ALLOWED_LLM_ROLES: tuple[LLMRole, ...] = (
    "agent", "planner", "proactive"
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError:
        return default
    return values or default


def _resolve_project_path(
    env_name: str,
    default: str,
    *,
    legacy_names: tuple[str, ...] = (),
) -> Path:
    """Resolve a canonical dataset/runtime path relative to the project root."""
    project_root = Path(__file__).resolve().parent.parent
    raw = os.getenv(env_name)
    if raw is None:
        raw = next(
            (value for name in legacy_names if (value := os.getenv(name)) is not None),
            default,
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _resolve_dataset_root() -> Path:
    """Resolve ECG dataset path relative to project root, not process CWD."""
    project_root = Path(__file__).resolve().parent.parent
    raw = os.getenv(
        "ECG_DATASET_ROOT",
        "./dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
    )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _resolve_upload_root() -> Path:
    """Resolve the local persistent cache for uploaded single-lead ECG files."""
    project_root = Path(__file__).resolve().parent.parent
    raw = os.getenv("ECG_UPLOAD_ROOT", "./.runtime/apple_watch_uploads")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


@dataclass
class LLMConfig:
    """LLM provider configuration (OpenAI-compatible API)."""

    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.3))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 2048))
    connect_timeout_s: float = field(
        default_factory=lambda: _env_float("LLM_CONNECT_TIMEOUT_S", 5.0)
    )
    read_timeout_s: float = field(
        default_factory=lambda: _env_float("LLM_READ_TIMEOUT_S", 600.0)
    )
    write_timeout_s: float = field(
        default_factory=lambda: _env_float("LLM_WRITE_TIMEOUT_S", 600.0)
    )
    pool_timeout_s: float = field(
        default_factory=lambda: _env_float("LLM_POOL_TIMEOUT_S", 600.0)
    )
    sdk_max_retries: int = field(
        default_factory=lambda: _env_int("LLM_SDK_MAX_RETRIES", 2)
    )
    transient_retry_attempts: int = field(
        default_factory=lambda: _env_int("LLM_TRANSIENT_RETRY_ATTEMPTS", 3)
    )
    transient_retry_backoff_s: tuple[float, ...] = field(
        default_factory=lambda: _env_float_tuple(
            "LLM_TRANSIENT_RETRY_BACKOFF_S",
            (1.0, 2.0, 4.0),
        )
    )

    def __post_init__(self) -> None:
        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not set — LLM calls will fail")

@dataclass
class ECGConfig:
    """ECG data-related settings."""

    # Path to the PTB-XL dataset root directory
    dataset_root: Path = field(default_factory=_resolve_dataset_root)
    # Sampling frequency in Hz (100 or 500 for PTB-XL)
    sampling_rate: int = 100
    # Number of leads
    num_leads: int = 12
    # Persistent storage for uploaded Apple Watch / single-lead ECG records
    upload_root: Path = field(default_factory=_resolve_upload_root)


@dataclass
class PipelineConfig:
    """Reactive pipeline behaviour."""

    # Maximum number of re-plan retries when validation fails
    max_retries: int = field(
        default_factory=lambda: _env_int("PIPELINE_MAX_RETRIES", 1)
    )
    # Whether to log intermediate steps
    verbose: bool = field(
        default_factory=lambda: _env_flag("PIPELINE_VERBOSE", True)
    )
    # Whether to perform validation checks on tool results before answer generation
    disable_validation: bool = field(
        default_factory=lambda: _env_flag("PIPELINE_DISABLE_VALIDATION", False)
    )


@dataclass
class ToolsConfig:
    """Feature flags for optional tools and integrations."""

    enable_ecg_diagnosis: bool = field(
        default_factory=lambda: _env_flag("ENABLE_ECG_DIAGNOSIS", True)
    )


@dataclass
class DatasetConfig:
    """Canonical local dataset and learned-artifact paths."""

    icentia11k_root: Path = field(
        default_factory=lambda: _resolve_project_path(
            "DATASET_ICENTIA11K_ROOT",
            "./dataset/icentia11k_subset/1.0",
            legacy_names=("ICENTIA11K_DATASET_ROOT", "PROACTIVE_ICENTIA11K_ROOT", "ICENTIA11K_ROOT"),
        )
    )
    afppgecg_root: Path = field(
        default_factory=lambda: _resolve_project_path(
            "DATASET_AFPPGECG_ROOT",
            "./dataset/zenodo_af_monitoring_v11242869/raw",
            legacy_names=(
                "DATASET_AF_PPG_ECG_ROOT",
                "PROACTIVE_ZENODO_DATASET_ROOT",
            ),
        )
    )
    ppg_dalia_root: Path = field(
        default_factory=lambda: _resolve_project_path(
            "DATASET_PPG_DALIA_ROOT",
            "./dataset/ppg_dalia/PPG_FieldStudy",
        )
    )
    wesad_root: Path = field(
        default_factory=lambda: _resolve_project_path(
            "DATASET_WESAD_ROOT",
            "./dataset/WESAD",
        )
    )
    wesad_stress_model_path: Path = field(
        default_factory=lambda: _resolve_project_path(
            "WESAD_STRESS_MODEL_PATH",
            "./.runtime/wesad_stress_classifier.pkl",
        )
    )
    vitalbench_root: Path = field(
        default_factory=lambda: _resolve_project_path(
            "DATASET_VITALBENCH_ROOT",
            "./dataset/vitalbench/final",
        )
    )


@dataclass
class AppConfig:
    """Top-level application config that bundles all sub-configs."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    ecg: ECGConfig = field(default_factory=ECGConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Return (and lazily create) the global AppConfig singleton."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def _role_env_name(role: LLMRole, suffix: str) -> str:
    return f"LLM_{role.upper()}_{suffix}"


def _role_text_value(role: LLMRole, suffix: str, default: str) -> str:
    return os.getenv(_role_env_name(role, suffix), default)


def _role_float_value(role: LLMRole, suffix: str, default: float) -> float:
    return _env_float(_role_env_name(role, suffix), default)


def _role_int_value(role: LLMRole, suffix: str, default: int) -> int:
    return _env_int(_role_env_name(role, suffix), default)


def _role_flag_value(role: LLMRole, suffix: str, default: bool) -> bool:
    return _env_flag(_role_env_name(role, suffix), default)


def get_llm_profile(role: LLMRole):
    """Return the resolved LLM profile for the requested role."""
    if role not in _ALLOWED_LLM_ROLES:
        raise ValueError(f"Unsupported LLM role: {role!r}")

    from agent.llm import LLMProfile

    cfg = get_config().llm
    return LLMProfile(
        role=role,
        api_key=_role_text_value(role, "API_KEY", cfg.api_key),
        base_url=_role_text_value(role, "BASE_URL", cfg.base_url),
        model=_role_text_value(role, "MODEL", cfg.model),
        temperature=_role_float_value(role, "TEMPERATURE", cfg.temperature),
        max_tokens=_role_int_value(role, "MAX_TOKENS", cfg.max_tokens),
        disable_reasoning=_role_flag_value(role, "DISABLE_REASONING", role == "agent"),
    )


def resolved_config_snapshot() -> dict[str, object]:
    """Return a secret-free snapshot of runtime settings for experiment artifacts."""
    cfg = get_config()
    profiles: dict[str, object] = {}
    for role in ("agent", "planner", "proactive"):
        profile = get_llm_profile(role)
        profiles[role] = {
            "model": profile.model,
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
            "disable_reasoning": profile.disable_reasoning,
        }
    return {
        "llm_profiles": profiles,
        "llm_transport": {
            "connect_timeout_s": cfg.llm.connect_timeout_s,
            "read_timeout_s": cfg.llm.read_timeout_s,
            "write_timeout_s": cfg.llm.write_timeout_s,
            "pool_timeout_s": cfg.llm.pool_timeout_s,
            "sdk_max_retries": cfg.llm.sdk_max_retries,
            "transient_retry_attempts": cfg.llm.transient_retry_attempts,
            "transient_retry_backoff_s": list(cfg.llm.transient_retry_backoff_s),
        },
        "pipeline": {
            "max_retries": cfg.pipeline.max_retries,
            "verbose": cfg.pipeline.verbose,
            "disable_validation": cfg.pipeline.disable_validation,
        },
        "tools": {
            "enable_ecg_diagnosis": cfg.tools.enable_ecg_diagnosis,
        },
        "datasets": {
            "icentia11k_root": str(cfg.datasets.icentia11k_root),
            "afppgecg_root": str(cfg.datasets.afppgecg_root),
            "ppg_dalia_root": str(cfg.datasets.ppg_dalia_root),
            "wesad_root": str(cfg.datasets.wesad_root),
            "wesad_stress_model_path": str(cfg.datasets.wesad_stress_model_path),
            "vitalbench_root": str(cfg.datasets.vitalbench_root),
        },
    }


def reset_config() -> None:
    """Clear the cached config so environment overrides are re-read."""
    global _config
    _config = None


__all__ = [
    "AppConfig",
    "ECGConfig",
    "DatasetConfig",
    "LLMConfig",
    "LLMRole",
    "PipelineConfig",
    "ToolsConfig",
    "get_config",
    "get_llm_profile",
    "resolved_config_snapshot",
    "reset_config",
]


