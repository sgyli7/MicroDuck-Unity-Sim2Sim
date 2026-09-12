"""Run real Unity/Tuanjie + Barracuda physics acceptance; never emulate an editor."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import hashlib


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor", type=Path, required=True,
                        help="Installed supported Unity/Tuanjie editor executable")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--build-windows", action="store_true",
                        help="After real editor acceptance passes, build and hash a Windows player")
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
    experimental = report.get("experimental") or {}
    passed = report.get("actual_unity_editor") is True and report.get("passed") is True
    passed &= physical.get("passed") is True and len(physical.get("cases", [])) == 12
    passed &= experimental.get("passed") is True and len(experimental.get("cases", [])) == 2
    if passed and args.build_windows:
        build_marker = output / "windows-build.completed"
        if build_marker.exists():
            build_marker.replace(output / "previous-windows-build.completed")
        build = [str(editor), "-batchmode", "-nographics", "-quit", "-projectPath", str(project),
                 "-executeMethod", "SaiAgent001.Editor.SaiDemoBuilder.BuildWindows64",
                 "-logFile", str(output / "windows-build.log")]
        try:
            finished = subprocess.run(build, cwd=root, timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired:
            print(f"Build timed out; inspect {output / 'windows-build.log'}", file=sys.stderr)
            return 1
        player = root / "Builds/SaiWindows64/Sai_Agent_001.exe"
        native = player.parent / "Sai_Agent_001_Data/Plugins/x86_64/mujoco.dll"
        if finished.returncode != 0 or not build_marker.exists() or not player.is_file() or not native.is_file():
            print(f"Player build failed or expected files missing; inspect {output / 'windows-build.log'}", file=sys.stderr)
            return 1
        expected = json.loads((root / "upstream.lock.json").read_text())["nativeBinaries"]["mujocoWindowsX64"]["sha256"]
        if sha256(native) != expected:
            print("Built player's MuJoCo DLL differs from the supply-chain lock", file=sys.stderr)
            return 1
        binaries = {str(p.relative_to(player.parent)): sha256(p)
                    for p in sorted(player.parent.rglob("*")) if p.is_file()}
        (output / "player-manifest.json").write_text(json.dumps({
            "player": str(player), "files_sha256": binaries,
            "player_execution_verified": False}, indent=2) + "\n")
    print(json.dumps({"passed": passed, "report": str(result),
                      "scope": "Editor + Barracuda physical scenarios; keyboard and rendering still require Play acceptance"}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
