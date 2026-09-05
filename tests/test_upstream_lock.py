from pathlib import Path

import pytest

from agenticrobot_bridge.upstream import LockFileError, load_upstream_lock


ROOT = Path(__file__).parents[1]


def test_repository_lock_exposes_two_pinned_commits_and_nine_policies() -> None:
    lock = load_upstream_lock(ROOT / "upstream.lock.json")

    assert set(lock.repositories) == {"microduck", "microduck_rl"}
    assert all(len(repo.commit) == 40 for repo in lock.repositories.values())
    assert len(lock.policies) == 9
    assert set(lock.policies) == {
        "alpha_walking.onnx",
        "alpha_stand.onnx",
        "alpha_sitstand.onnx",
        "alpha_ground_pick.onnx",
        "ball_kick_left.onnx",
        "ball_kick_right.onnx",
        "roller.onnx",
        "roller_crouch.onnx",
        "roulade.onnx",
    }


def test_repository_lock_rejects_non_sha_commit(tmp_path: Path) -> None:
    invalid = tmp_path / "upstream.lock.json"
    invalid.write_text(
        '{"schemaVersion":1,"repositories":{"microduck":'
        '{"url":"https://example.test/repo.git","commit":"main"}},"policies":[]}',
        encoding="utf-8",
    )

    with pytest.raises(LockFileError, match="40-character commit SHA"):
        load_upstream_lock(invalid)

