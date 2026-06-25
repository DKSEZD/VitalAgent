from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _missing_dependency(*_args, **_kwargs):
    raise RuntimeError("External dependency stub called unexpectedly in a unit test.")


def _install_stub_module(name: str, attributes: dict[str, object]) -> None:
    if importlib.util.find_spec(name) is not None:
        return

    module = types.ModuleType(name)
    for attr_name, value in attributes.items():
        setattr(module, attr_name, value)
    sys.modules.setdefault(name, module)


_install_stub_module(
    "wfdb",
    {
        "rdheader": _missing_dependency,
        "rdsamp": _missing_dependency,
        "rdann": _missing_dependency,
    },
)

_install_stub_module(
    "neurokit2",
    {
        "ecg_process": _missing_dependency,
        "ecg_delineate": _missing_dependency,
    },
)
