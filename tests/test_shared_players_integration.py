"""Runs actual independent Players: pipeline verification, NOT skill acceptance."""

import copy
import json
import uuid
from pathlib import Path

from agenticrobot_bridge.shared_experiment import make_batch, run_players, write_new


def test_actual_paired_players_use_same_inputs_and_keep_behavior_failures_explicit():
    root = Path(__file__).parents[1]
    batch = make_batch(root / ".cache/upstream/microduck/policies", 92317)
    # Legal non-whole-second horizons must stop at integer native ticks, not at
    # float32 seconds rounded above the intended terminal time.
    for ticks in (68, 148):
        short = copy.deepcopy(batch["cases"][1])
        short.update(id=f"fractional-{ticks}", physicsSteps=ticks)
        batch["cases"].append(short)
    evidence = root / "artifacts/sim2sim" / ("shared-integration-" + uuid.uuid4().hex[:12])
    spec = evidence.with_suffix(".json")
    write_new(spec, batch)
    reference_player = root.parent / "AgenticRobotGame-MuJoCoMeasured-20260909/Builds/MeasuredReference/AgenticRobotGame-MuJoCoReference.exe"
    run_players(spec, evidence, reference_player, root / "Builds/Windows64/AgenticRobotGame.exe", 62103)
    comparison = json.loads((evidence / "comparison.json").read_text(encoding="utf-8"))
    reference = json.loads((evidence / "reference.json").read_text(encoding="utf-8"))
    target = json.loads((evidence / "target.json").read_text(encoding="utf-8"))
    assert comparison["physxAcceptedCount"] == 0
    assert len(comparison["cases"]) == len(reference["episodes"]) == len(target["results"]) == 11
    assert all(item["graphVerified"] for item in reference["models"])
    for case, measured, result in zip(batch["cases"], reference["episodes"], comparison["cases"], strict=True):
        assert measured["completed"] is True
        assert measured["frames"][-1]["physicsSteps"] == case["physicsSteps"]
        assert result["status"] in ("recorded_diagnostic", "incomplete")
        assert result["maximumInitialPositionError"] < 1e-5
        assert result["maximumInitialRotationErrorRadians"] < 1e-5
        assert result["maximumInitialSpeed"] < 1e-6
        assert result["behaviorAccepted"] is False
    print("Independent-player diagnostic evidence:", evidence)
