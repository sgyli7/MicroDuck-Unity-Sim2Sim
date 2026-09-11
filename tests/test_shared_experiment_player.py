import json

import pytest

from agenticrobot_bridge.physx_client import PhysXClient


def experiment(client):
    return {"schemaVersion": 1, "coordinateBasis": "Unity-Xright-Yup-Zforward",
            "models": [{"slot": model["slot"], "sha256": model["sourceSha256"]}
                       for model in client.identity["models"]],
            "cases": [{"id": "timeline-contract", "seed": 71, "slot": 3, "initialSlot": 3,
                       "terrain": "flat", "physicsSteps": 12,
                       "initialRootPosition": [0.01, 0.125, -0.01], "initialYawDegrees": 3,
                       "events": [{"physicsStep": 4, "kind": "trigger", "slot": 0, "values": []}]}]}


def test_shared_experiment_drives_initial_pose_and_commands_before_next_observation():
    import hashlib

    with PhysXClient(timeout=10) as client:
        spec = experiment(client)
        encoded = json.dumps(spec)
        initial = client.request("experiment", experimentJson=encoded, caseIndex=0)["results"]
        for frame in initial:
            assert frame["experimentSha256"] == hashlib.sha256(encoded.encode()).hexdigest()
            assert frame["experimentId"] == "timeline-contract"
            assert frame["rootPosition"] == pytest.approx([0.01, 0.125, -0.01], abs=1e-6)
            assert frame["physicsSteps"] == 0
        next_frame = client.step()[0]
        assert next_frame["physicsSteps"] == 4
        assert next_frame["observation"][48:] != initial[0]["observation"][48:]
        client.step()
        client.step()
        with pytest.raises(RuntimeError, match="horizon"):
            client.step()


def test_invalid_common_conditions_fail_before_changing_any_environment():
    with PhysXClient(timeout=10) as client:
        client.reset(2)
        client.step([[0.0] * 14] * 2)
        spec = experiment(client)
        spec["cases"][0]["terrain"] = "unimplemented-terrain"
        with pytest.raises(RuntimeError, match="terrain"):
            client.request("experiment", experimentJson=json.dumps(spec), caseIndex=0)
        assert all(frame["physicsSteps"] == 8 for frame in client.step([[0.0] * 14] * 2))


def test_common_pose_does_not_replace_default_reset_and_hot_swap_keeps_live_state():
    import copy

    with PhysXClient(timeout=10) as client:
        default_before = client.reset(2)
        spec = experiment(client)
        case = spec["cases"][0]
        case["slot"] = case["initialSlot"] = 2
        case["events"] = []
        control = client.request("experiment", experimentJson=json.dumps(spec), caseIndex=0)["results"]
        control = client.step()
        changed = copy.deepcopy(spec)
        changed["cases"][0]["events"] = [{"physicsStep": 4, "kind": "switch", "slot": 9, "values": []}]
        client.request("experiment", experimentJson=json.dumps(changed), caseIndex=0)
        switched = client.step()
        for before, after in zip(control, switched, strict=True):
            assert before["activeSlot"] == 2 and after["activeSlot"] == 9
            for field in ("rootPosition", "rootRotation", "rootVelocity", "jointPosition", "jointVelocity", "targets", "policyObservation", "action"):
                assert after[field] == pytest.approx(before[field], abs=1e-7), field
        default = client.reset(2)
        assert all(frame["rootPosition"] == pytest.approx(before["rootPosition"], abs=1e-7)
                   for frame, before in zip(default, default_before, strict=True))
