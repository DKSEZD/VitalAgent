from __future__ import annotations

import pytest

from agent.config import reset_config
from agent.data.ecg_loader import ECGDatasetNotFoundError
from agent.tools import registry


@pytest.fixture(autouse=True)
def reset_tool_registry():
    original_tools = registry._TOOLS.copy()
    registry._TOOLS.clear()
    reset_config()
    try:
        yield
    finally:
        registry._TOOLS.clear()
        registry._TOOLS.update(original_tools)
        reset_config()


def test_register_and_get_tool() -> None:
    @registry.register_tool("demo_tool", "demo description", {"value": {"type": "integer"}})
    def demo_tool(value: int) -> dict[str, object]:
        return {"success": True, "value": value}

    tool = registry.get_tool("demo_tool")

    assert tool is demo_tool
    assert registry.list_tools() == [
        {
            "name": "demo_tool",
            "description": "demo description",
            "parameters": {"value": {"type": "integer"}},
        }
    ]


def test_register_tool_overwrites_existing_name() -> None:
    @registry.register_tool("demo_tool", "first")
    def first_tool() -> dict[str, object]:
        return {"success": True, "value": "first"}

    @registry.register_tool("demo_tool", "second")
    def second_tool() -> dict[str, object]:
        return {"success": True, "value": "second"}

    assert registry.get_tool("demo_tool") is second_tool
    assert registry.call_tool("demo_tool") == {"success": True, "value": "second"}
    assert first_tool is not second_tool


def test_call_tool_returns_error_for_unknown_tool() -> None:
    assert registry.call_tool("missing_tool") == {
        "error": "Tool 'missing_tool' not found.",
        "success": False,
    }


def test_call_tool_wraps_exceptions() -> None:
    @registry.register_tool("failing_tool", "fails")
    def failing_tool() -> dict[str, object]:
        raise RuntimeError("boom")

    assert registry.call_tool("failing_tool") == {"error": "boom", "success": False}


@pytest.mark.parametrize("legacy_name", ["af_ppg_ecg", "zenodo", "zenodo_af_monitoring"])
def test_call_tool_canonicalizes_legacy_dataset_names(legacy_name: str) -> None:
    @registry.register_tool("dataset_tool", "returns its dataset")
    def dataset_tool(dataset: str) -> dict[str, object]:
        return {"success": True, "dataset": dataset}

    assert registry.call_tool("dataset_tool", dataset=legacy_name) == {
        "success": True,
        "dataset": "afppgecg",
    }



def test_call_tool_reraises_missing_dataset_error() -> None:
    @registry.register_tool("dataset_tool", "fails on missing dataset")
    def dataset_tool() -> dict[str, object]:
        raise ECGDatasetNotFoundError("missing ptbxl_database.csv")

    with pytest.raises(ECGDatasetNotFoundError, match="ptbxl_database.csv"):
        registry.call_tool("dataset_tool")



def test_disabled_tool_is_hidden_and_not_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "false")
    reset_config()

    @registry.register_tool("ecg_diagnosis", "diagnosis helper")
    def diagnosis_tool() -> dict[str, object]:
        return {"success": True, "value": "should not run"}

    assert registry.is_tool_enabled("ecg_diagnosis") is False
    assert registry.get_tool("ecg_diagnosis") is None
    assert all(tool["name"] != "ecg_diagnosis" for tool in registry.list_tools())
    assert "ecg_diagnosis" not in registry.get_tools_prompt()
    assert registry.call_tool("ecg_diagnosis") == {
        "error": "Tool 'ecg_diagnosis' is disabled.",
        "success": False,
    }
    assert diagnosis_tool() == {"success": True, "value": "should not run"}
