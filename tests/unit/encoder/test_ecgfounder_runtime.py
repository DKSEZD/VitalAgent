from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from agent.encoder import ecgfounder_runtime


@pytest.fixture(autouse=True)
def clear_runtime_cache() -> None:
    ecgfounder_runtime.load_ecgfounder_runtime.cache_clear()
    yield
    ecgfounder_runtime.load_ecgfounder_runtime.cache_clear()


def test_ensure_ecgfounder_available_downloads_only_requested_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    repo_root = tmp_path / "ECGFounder"
    repo_root.mkdir()
    (repo_root / "tasks.txt").write_text("TASK_A\n")
    downloads: list[str] = []

    monkeypatch.setattr(ecgfounder_runtime, "ecgfounder_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        ecgfounder_runtime,
        "_download_ecgfounder_checkpoint",
        lambda destination_path, url: downloads.append(destination_path.name)
        or destination_path.parent.mkdir(parents=True, exist_ok=True)
        or destination_path.write_bytes(b"checkpoint"),
    )

    ecgfounder_runtime.ensure_ecgfounder_available("single_lead")
    assert downloads == ["1_lead_ECGFounder.pth"]

    downloads.clear()
    ecgfounder_runtime.ensure_ecgfounder_available("12_lead")
    assert downloads == ["12_lead_ECGFounder.pth"]


def test_load_ecgfounder_runtime_caches_per_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    repo_root = tmp_path / "ECGFounder"
    repo_root.mkdir()
    (repo_root / "tasks.txt").write_text("TASK_A\nTASK_B\n")
    ensure_calls: list[str] = []
    model_channels: list[int] = []

    class FakeModel:
        def __init__(self, in_channels: int, **kwargs):
            model_channels.append(in_channels)
            self.in_channels = in_channels

        def load_state_dict(self, state_dict, strict: bool = False):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_torch = SimpleNamespace(
        device=lambda name: name,
        cuda=SimpleNamespace(is_available=lambda: False),
        load=lambda path, map_location=None, weights_only=False: {"state_dict": {}},
    )

    monkeypatch.setattr(ecgfounder_runtime, "ecgfounder_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        ecgfounder_runtime,
        "ensure_ecgfounder_available",
        lambda mode="12_lead": ensure_calls.append(mode),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "net1d", SimpleNamespace(Net1D=FakeModel))

    runtime_12_a = ecgfounder_runtime.load_ecgfounder_runtime("12_lead")
    runtime_12_b = ecgfounder_runtime.load_ecgfounder_runtime("12_lead")
    runtime_1 = ecgfounder_runtime.load_ecgfounder_runtime("single_lead")

    assert runtime_12_a is runtime_12_b
    assert ensure_calls == ["12_lead", "single_lead"]
    assert model_channels == [12, 1]
    assert runtime_12_a["in_channels"] == 12
    assert runtime_1["in_channels"] == 1
    assert runtime_12_a["mode"] == "12_lead"
    assert runtime_1["mode"] == "single_lead"
