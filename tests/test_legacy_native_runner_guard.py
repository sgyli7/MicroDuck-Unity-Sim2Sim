"""Historical native gates must not masquerade as the PhysX delivery runner."""

import subprocess
from pathlib import Path


def test_native_runner_refuses_the_physx_main_project_even_in_dry_run():
    root = Path(__file__).parents[1]
    result = subprocess.run([
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(root / "scripts/run-mvp.ps1"), "-Stage", "windows-build", "-DryRun",
    ], cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    assert "REFERENCE_ONLY" in result.stderr
