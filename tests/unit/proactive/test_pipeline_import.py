from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_pipeline_import_does_not_require_wfdb() -> None:
    project_root = Path(__file__).resolve().parents[3]
    code = """
import builtins

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "wfdb":
        raise ModuleNotFoundError("No module named 'wfdb'")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from agent.proactive.pipeline import ProactivePipeline

assert ProactivePipeline is not None
print("ok")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
