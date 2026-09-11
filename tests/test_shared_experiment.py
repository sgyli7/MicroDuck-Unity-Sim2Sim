import copy
import hashlib
import json
from pathlib import Path

import pytest

from agenticrobot_bridge.shared_experiment import compare_batch, copy_experiment, make_batch, write_new


def test_seeded_common_inputs_preserve_compound_phase_sequence_and_original_sources():
    root = Path(__file__).parents[1] / ".cache/upstream/microduck/policies"
    first = make_batch(root, 123)
    assert first == make_batch(root, 123)
    assert first != make_batch(root, 124)
    assert [model["slot"] for model in first["models"]] == list(range(1, 10))
    crouch, roulade = first["cases"][7:]
    assert crouch["initialSlot"] == 7 and crouch["physicsSteps"] == 1600
    assert [(event["physicsStep"], event["slot"]) for event in crouch["events"] if event["kind"] == "switch"] == [(600, 8), (1300, 7)]
    assert roulade["initialSlot"] == 2 and roulade["physicsSteps"] == 1000
    assert all(case["terrain"] == "flat" for case in first["cases"])


def paired():
    case = {"id": "same-case", "slot": 2, "physicsSteps": 4, "initialRootPosition": [0, .125, 0], "initialYawDegrees": 0}
    encoded = json.dumps({"cases": [case]})
    digest = hashlib.sha256(encoded.encode()).hexdigest()

    def frame(engine, t, position):
        return {"engine": engine, "timeSeconds": t, "rootPosition": position,
                "rootQuaternionWxyz": [1, 0, 0, 0], "rootRotation": [0, 0, 0, 1],
                "rootVelocity": [0, 0, 0], "rootAngularVelocity": [0, 0, 0],
                "upright": 1, "policyObservation": [0] * 61, "action": [0] * 14}

    reference = {"engine": "MuJoCo", "episodes": [{"slot": 2, "experimentId": "same-case",
                  "experimentSha256": digest, "requestedSeconds": .02, "completed": True,
                  "frames": [frame("MuJoCo", 0, [0, 0, .125]), frame("MuJoCo", .02, [0, 0, .125])]}]}
    target = {"identity": {"engine": "PhysX"}, "results": [{"slot": 2, "experimentId": "same-case",
               "experimentSha256": digest, "requestedSeconds": .02,
               "episodes": [[frame("PhysX", 0, [0, .125, 0]), frame("PhysX", .02, [0, .125, 0])]]}]}
    return encoded, reference, target


def test_shared_hash_and_actual_start_are_checked_without_behavior_promotion():
    encoded, reference, target = paired()
    result = compare_batch(encoded, reference, target)
    assert result["physxAcceptedCount"] == 0
    assert result["cases"][0]["status"] == "recorded_diagnostic"
    assert result["cases"][0]["maximumInitialPositionError"] == 0
    changed = copy.deepcopy(target)
    changed["results"][0]["episodes"][0][0]["rootPosition"][0] = .1
    assert compare_batch(encoded, reference, changed)["cases"][0]["status"] == "initial_condition_mismatch"
    target["results"][0]["experimentSha256"] = "wrong"
    with pytest.raises(ValueError, match="hash mismatch"):
        compare_batch(encoded, reference, target)


def test_declared_position_match_does_not_hide_wrong_initial_heading_or_speed():
    encoded, reference, target = paired()
    initial = target["results"][0]["episodes"][0][0]
    initial["rootRotation"] = [0, 1, 0, 0]
    assert compare_batch(encoded, reference, target)["cases"][0]["status"] == "initial_condition_mismatch"
    initial["rootRotation"] = [0, 0, 0, 1]
    initial["rootVelocity"] = [.1, 0, 0]
    assert compare_batch(encoded, reference, target)["cases"][0]["status"] == "initial_condition_mismatch"


def test_existing_evidence_is_not_overwritten_and_missing_cases_are_not_passed(tmp_path):
    path = tmp_path / "evidence.json"
    write_new(path, {"original": True})
    with pytest.raises(FileExistsError):
        write_new(path, {"original": False})
    encoded, reference, target = paired()
    reference["episodes"] = []
    assert compare_batch(encoded, reference, target)["cases"][0]["status"] == "missing"


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_both_players_receive_identical_bytes_without_windows_newline_translation(tmp_path, newline):
    payload = b"{" + newline + b'"schemaVersion": 1' + newline + b"}"
    source, copied = tmp_path / "input.json", tmp_path / "copy.json"
    source.write_bytes(payload)
    encoded = copy_experiment(source, copied)
    assert copied.read_bytes() == encoded.encode() == payload
