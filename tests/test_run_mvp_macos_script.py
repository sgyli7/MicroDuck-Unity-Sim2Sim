import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run-mvp-macos.py"
EXPECTED_STAGES = [
    "bootstrap",
    "python-tests",
    "policy-audit",
    "mujoco-rollouts",
    "ppo-smoke",
    "onnx-export",
    "robot-manifest",
    "tuanjie-editmode",
    "tuanjie-playmode",
    "trace-parity",
    "codely-proof",
    "windows-build",
    "macos-build",
    "macos-environment-acceptance",
    "training-prep",
]
CUDA_SKIP_REASON = (
    "requires CUDA (microduck_rl/mjlab Warp); not available on Apple Silicon"
)


def _run(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.pop("TUANJIE_EDITOR", None)
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=merged,
    )


def test_list_stages_exposes_the_complete_ordered_gate_list() -> None:
    completed = _run(["--list-stages", "--json"])
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == EXPECTED_STAGES


def test_dry_run_json_has_no_windows_paths_and_marks_platform_stages(
    tmp_path: Path,
) -> None:
    completed = _run(
        ["--dry-run", "--json", "--artifacts-root", str(tmp_path)],
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    text = completed.stdout
    assert ".exe" not in text
    assert "\\" not in text
    report = json.loads(text)
    assert [item["name"] for item in report["stages"]] == EXPECTED_STAGES
    by_name = {item["name"]: item for item in report["stages"]}
    assert by_name["ppo-smoke"]["status"] == "skipped"
    assert by_name["ppo-smoke"]["reason"] == CUDA_SKIP_REASON
    assert by_name["onnx-export"]["status"] == "skipped"
    assert by_name["onnx-export"]["reason"] == CUDA_SKIP_REASON
    assert by_name["windows-build"]["status"] == "not-applicable"
    assert by_name["windows-build"]["reason"]
    assert by_name["codely-proof"]["status"] == "not-applicable"
    assert by_name["codely-proof"]["reason"]
    for item in report["stages"]:
        if item["status"] != "passed":
            assert item.get("reason"), item
        rendered = json.dumps(item.get("commands", []))
        assert ".exe" not in rendered
        assert "\\" not in rendered
