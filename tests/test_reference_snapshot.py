from pathlib import Path

import pytest


def test_snapshot_is_independent_and_detects_changed_baseline(tmp_path: Path) -> None:
    from agenticrobot_bridge.reference_snapshot import create_snapshot, verify_snapshot

    source = tmp_path / "main"
    asset = source / "TuanjieProject/Assets/robot.bin"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"frozen robot and weights")
    target = tmp_path / "reference"

    manifest = create_snapshot(source, target, ["TuanjieProject/Assets"], "source-revision")
    assert manifest["sourceRevision"] == "source-revision"
    assert verify_snapshot(target)["passed"]
    asset.write_bytes(b"changed PhysX main")
    assert (target / "TuanjieProject/Assets/robot.bin").read_bytes() == b"frozen robot and weights"
    assert verify_snapshot(target)["passed"]

    (target / "TuanjieProject/Assets/robot.bin").write_bytes(b"tampered reference")
    result = verify_snapshot(target)
    assert not result["passed"]
    assert result["changed"] == ["TuanjieProject/Assets/robot.bin"]


@pytest.mark.parametrize("root", ["missing", "../outside"])
def test_invalid_inputs_do_not_create_a_partial_snapshot(tmp_path: Path, root: str) -> None:
    from agenticrobot_bridge.reference_snapshot import create_snapshot

    source = tmp_path / "main"
    source.mkdir()
    (tmp_path / "outside").write_text("not in the project")
    target = tmp_path / "reference"
    with pytest.raises((ValueError, FileNotFoundError)):
        create_snapshot(source, target, [root], "revision")
    assert not target.exists()


def test_refuses_nested_target_and_existing_reference(tmp_path: Path) -> None:
    from agenticrobot_bridge.reference_snapshot import create_snapshot

    source = tmp_path / "main"
    source.mkdir()
    (source / "asset").write_text("frozen")
    with pytest.raises(ValueError):
        create_snapshot(source, source / "nested", ["asset"], "revision")
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        create_snapshot(source, target, ["asset"], "revision")


def test_verifier_detects_missing_and_added_files(tmp_path: Path) -> None:
    from agenticrobot_bridge.reference_snapshot import create_snapshot, verify_snapshot

    source = tmp_path / "main"
    (source / "Assets").mkdir(parents=True)
    (source / "Assets/original").write_text("original")
    target = tmp_path / "reference"
    create_snapshot(source, target, ["Assets"], "revision")
    (target / "Assets/original").unlink()
    (target / "Assets/injected").write_text("new behavior")
    result = verify_snapshot(target)
    assert not result["passed"]
    assert result["missing"] == ["Assets/original"]
    assert result["added"] == ["Assets/injected"]
