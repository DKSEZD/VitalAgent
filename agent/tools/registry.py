"""
Tool registry — lightweight mechanism for registering and discovering
callable tools that the Planner can invoke.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from agent.config import get_config
from agent.data.ecg_loader import ECGDatasetNotFoundError
from agent.mhealth.schemas import canonicalize_dataset_name

logger = logging.getLogger(__name__)

ToolFunc = Callable[..., dict[str, Any]]

# Global registry: name → (function, description, parameter_schema)
_TOOLS: dict[str, tuple[ToolFunc, str, dict[str, Any]]] = {}


def is_tool_enabled(name: str) -> bool:
    """Return whether a registered tool should be exposed at runtime."""
    if name == "ecg_diagnosis":
        return get_config().tools.enable_ecg_diagnosis
    return True


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> Callable[[ToolFunc], ToolFunc]:
    """Decorator to register a tool function."""

    def decorator(func: ToolFunc) -> ToolFunc:
        _TOOLS[name] = (func, description, parameters or {})
        logger.debug("Registered tool: %s", name)
        return func

    return decorator


def get_tool(name: str) -> ToolFunc | None:
    if not is_tool_enabled(name):
        return None
    entry = _TOOLS.get(name)
    return entry[0] if entry else None


def call_tool(name: str, **kwargs: Any) -> dict[str, Any]:
    """Invoke a registered tool by name."""
    if not is_tool_enabled(name):
        return {"error": f"Tool '{name}' is disabled.", "success": False}
    entry = _TOOLS.get(name)
    if entry is None:
        return {"error": f"Tool '{name}' not found.", "success": False}
    func, _, _ = entry
    try:
        if "dataset" in kwargs and kwargs["dataset"] is not None:
            kwargs["dataset"] = canonicalize_dataset_name(kwargs["dataset"])
        return func(**kwargs)
    except ECGDatasetNotFoundError:
        logger.error("Tool '%s' failed due to missing ECG dataset", name)
        raise
    except Exception as e:
        logger.exception("Tool '%s' raised an exception", name)
        return {"error": str(e), "success": False}


def list_tools() -> list[dict[str, Any]]:
    """Return metadata about all registered tools."""
    return [
        {"name": name, "description": desc, "parameters": params}
        for name, (_, desc, params) in _TOOLS.items()
        if is_tool_enabled(name)
    ]


def get_tools_prompt() -> str:
    """Generate a human-readable description of all tools for LLM prompts."""
    lines = ["Available tools:\n"]
    for info in list_tools():
        lines.append(f"- **{info['name']}**: {info['description']}")
        if info["parameters"]:
            for pname, pinfo in info["parameters"].items():
                default = (
                    f" (default: {pinfo['default']})" if "default" in pinfo else ""
                )
                lines.append(
                    f"    - `{pname}` ({pinfo.get('type', 'any')}): "
                    f"{pinfo.get('description', '')}{default}"
                )
    return "\n".join(lines)
