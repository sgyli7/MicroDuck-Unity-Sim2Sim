"""Run real Unity/Tuanjie + Barracuda physics acceptance; never emulate an editor."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor", type=Path, required=True,
                        help="Installed supported Unity/Tuanjie editor executable")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    editor = args.editor.expanduser().resolve()
    if not editor.is_file():
        parser.error(f"Editor executable not found: {editor}")
    root = Path(__file__).resolve().parents[1]
    project = root / "TuanjieProject"
    if (project / "Temp/UnityLockfile").exists():
        parser.error("Close this project's editor, or run SaiAgent001/Validate Physics and Barracuda from its menu.")
    if not (project / "Assets/SaiAgent001/Generated/stairs-dev40.onnx").exists():
        parser.error("Run python scripts/setup-sai-agent.py first.")
    output = root / "artifacts/sai-editor-test"
    output.mkdir(parents=True, exist_ok=True)
    result = output / "result.json"
    # Preserve previous evidence and prevent a stale pass from satisfying this run.
    if result.exists():
        result.replace(output / "previous-result.json")
    command = [str(editor), "-batchmode", "-nographics", "-projectPath", str(project),
               "-executeMethod", "SaiAgent001.Editor.SaiEditorAcceptance.Run",
               "-logFile", str(output / "editor.log")]
    try:
        completed = subprocess.run(command, cwd=root, timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired:
        print(f"Editor timed out; inspect {output / 'editor.log'}", file=sys.stderr)
        return 1
    if completed.returncode != 0 or not result.exists():
        print(f"Editor failed (exit {completed.returncode}); inspect {output / 'editor.log'}", file=sys.stderr)
        return 1
    report = json.loads(result.read_text())
    physical = report.get("physical") or {}
    passed = report.get("actual_unity_editor") is True and report.get("passed") is True
    passed &= physical.get("passed") is True and len(physical.get("cases", [])) == 12
    print(json.dumps({"passed": passed, "report": str(result),
                      "scope": "Editor + Barracuda physical scenarios; keyboard and rendering still require Play acceptance"}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
