"""ECGFounder runtime/setup helpers for encoder and tool layers."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

ECGFounderMode = Literal["12_lead", "single_lead"]
_DEFAULT_MODE: ECGFounderMode = "12_lead"


class ECGFounderSetupError(RuntimeError):
    """Raised when ECGFounder assets cannot be prepared automatically."""


def ecgfounder_repo_root() -> Path:
    return Path(__file__).resolve().parents[2] / "external" / "ECGFounder"


def _normalize_mode(mode: ECGFounderMode | str) -> ECGFounderMode:
    if mode not in {"12_lead", "single_lead"}:
        raise ValueError(
            f"Unsupported ECGFounder mode {mode!r}. Expected '12_lead' or 'single_lead'."
        )
    return mode


def ecgfounder_checkpoint_path(mode: ECGFounderMode = _DEFAULT_MODE) -> Path:
    normalized_mode = _normalize_mode(mode)
    if normalized_mode == "single_lead":
        return ecgfounder_checkpoint_1lead_path()
    return ecgfounder_repo_root() / "checkpoint" / "12_lead_ECGFounder.pth"


def ecgfounder_checkpoint_1lead_path() -> Path:
    return ecgfounder_repo_root() / "checkpoint" / "1_lead_ECGFounder.pth"


def ecgfounder_repo_url() -> str:
    return os.getenv(
        "ECGFOUNDER_REPO_URL",
        "https://github.com/PKUDigitalHealth/ECGFounder.git",
    )


def ecgfounder_checkpoint_url() -> str:
    # Keep backward compatibility with existing env var name.
    return os.getenv(
        "ECGFOUNDER_CHECKPOINT_URL",
        "https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth",
    )


def ecgfounder_checkpoint_12lead_url() -> str:
    return os.getenv("ECGFOUNDER_CHECKPOINT_12LEAD_URL", ecgfounder_checkpoint_url())


def ecgfounder_checkpoint_1lead_url() -> str:
    return os.getenv(
        "ECGFOUNDER_CHECKPOINT_1LEAD_URL",
        "https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/1_lead_ECGFounder.pth",
    )


def _download_ecgfounder_checkpoint(destination_path: Path, url: str) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_suffix(destination_path.suffix + ".tmp")

    try:
        with urllib.request.urlopen(url, timeout=120) as response, temp_path.open("wb") as f:
            shutil.copyfileobj(response, f)
        if temp_path.stat().st_size <= 0:
            raise ECGFounderSetupError(
                f"Downloaded checkpoint is empty from {url}."
            )
        temp_path.replace(destination_path)
        logger.info("ECGFounder checkpoint downloaded to %s", destination_path)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise ECGFounderSetupError(
            "Failed to auto-download ECGFounder checkpoint from "
            f"{url}: {exc}"
        ) from exc


def ensure_ecgfounder_available(mode: ECGFounderMode = _DEFAULT_MODE) -> None:
    normalized_mode = _normalize_mode(mode)
    repo_root = ecgfounder_repo_root()
    checkpoint_12lead_path = ecgfounder_checkpoint_path("12_lead")
    checkpoint_1lead_path = ecgfounder_checkpoint_1lead_path()
    tasks_path = repo_root / "tasks.txt"

    if not repo_root.exists():
        git_executable = shutil.which("git")
        if not git_executable:
            raise ECGFounderSetupError(
                "ECGFounder is missing and Git is not installed/available in PATH. "
                f"Please install Git or manually clone to {repo_root}."
            )

        repo_root.parent.mkdir(parents=True, exist_ok=True)
        clone_cmd = [
            git_executable,
            "clone",
            "--depth",
            "1",
            ecgfounder_repo_url(),
            str(repo_root),
        ]
        try:
            subprocess.run(clone_cmd, check=True, capture_output=True, text=True)
            logger.info("ECGFounder repository cloned to %s", repo_root)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or repr(exc)
            raise ECGFounderSetupError(
                f"Failed to auto-download ECGFounder from {ecgfounder_repo_url()}: {detail}"
            ) from exc

    if not tasks_path.exists():
        raise ECGFounderSetupError(
            "ECGFounder task list not found after setup: "
            f"{tasks_path}. Please verify repository contents."
        )

    if normalized_mode == "single_lead":
        checkpoint_targets = [
            (checkpoint_1lead_path, ecgfounder_checkpoint_1lead_url(), "1-lead"),
        ]
    else:
        checkpoint_targets = [
            (checkpoint_12lead_path, ecgfounder_checkpoint_12lead_url(), "12-lead"),
        ]

    for checkpoint_path, checkpoint_url, checkpoint_name in checkpoint_targets:
        if not checkpoint_path.exists() or checkpoint_path.stat().st_size <= 0:
            _download_ecgfounder_checkpoint(checkpoint_path, checkpoint_url)

        if not checkpoint_path.exists() or checkpoint_path.stat().st_size <= 0:
            raise ECGFounderSetupError(
                "ECGFounder checkpoint not found after attempted auto-download: "
                f"{checkpoint_path} ({checkpoint_name})."
            )


def resample_signal_to_length(signal: np.ndarray, target_length: int = 5000) -> np.ndarray:
    """Resample a multi-lead signal to the fixed length expected by ECGFounder."""
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal array, got shape {signal.shape}")
    num_leads, current_length = signal.shape
    if current_length == target_length:
        return signal.astype(np.float32, copy=False)
    if current_length <= 1:
        raise ValueError("Signal is too short to resample.")

    x_old = np.linspace(0.0, 1.0, num=current_length, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    resampled = np.empty((num_leads, target_length), dtype=np.float32)
    for idx in range(num_leads):
        resampled[idx] = np.interp(x_new, x_old, signal[idx]).astype(np.float32)
    return resampled


def z_score_normalize(signal: np.ndarray) -> np.ndarray:
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    return ((signal - mean) / (std + 1e-8)).astype(np.float32)


@lru_cache(maxsize=2)
def load_ecgfounder_runtime(mode: ECGFounderMode = _DEFAULT_MODE) -> dict[str, Any]:
    normalized_mode = _normalize_mode(mode)
    repo_root = ecgfounder_repo_root()
    checkpoint_path = ecgfounder_checkpoint_path(normalized_mode)
    tasks_path = repo_root / "tasks.txt"

    ensure_ecgfounder_available(normalized_mode)

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    import torch  # type: ignore[import-not-found]
    from net1d import Net1D  # type: ignore[import-not-found]

    tasks = [line.strip() for line in tasks_path.read_text().splitlines() if line.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 1 if normalized_mode == "single_lead" else 12
    model = Net1D(
        in_channels=in_channels,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=len(tasks),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_result = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return {
        "mode": normalized_mode,
        "torch": torch,
        "device": device,
        "model": model,
        "tasks": tasks,
        "in_channels": in_channels,
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
    }
