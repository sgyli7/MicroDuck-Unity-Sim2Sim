import copy
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def actual_report():
    from agenticrobot_bridge.physx_client import PhysXClient
    from agenticrobot_bridge.shared_experiment import make_batch, sample_target

    with PhysXClient() as client:
        return sample_target(client, json.dumps(make_batch(".cache/upstream/microduck/policies", 92317)))


def test_actual_player_inputs_match_original_onnx_and_training_actor_for_all_nine(actual_report):
    from agenticrobot_bridge.trace_policy_parity import compare_trace

    result = compare_trace(actual_report, Path(".cache/upstream/microduck/policies"), samples_per_slot=24)
    assert result["numericalParityPassed"] is True, result
    assert len(result["models"]) == 9
    assert all(model["samples"] > 0 for model in result["models"])
    assert result["behaviorAccepted"] is False


def test_changed_logged_actions_cannot_be_counted_as_numerical_parity(actual_report):
    from agenticrobot_bridge.trace_policy_parity import compare_trace

    changed = copy.deepcopy(actual_report)
    for row in changed["results"]:
        for frame in row["episodes"][0]:
            if frame.get("inferenceSlot") == 2:
                frame["action"][0] += .01
    result = compare_trace(changed, Path(".cache/upstream/microduck/policies"), samples_per_slot=2)
    assert result["numericalParityPassed"] is False


def test_unknown_model_provenance_is_rejected_not_inferred_from_filenames(actual_report):
    from agenticrobot_bridge.trace_policy_parity import compare_trace

    report = copy.deepcopy(actual_report)
    report["identity"]["models"][0]["sourceSha256"] = "0" * 64
    with pytest.raises(ValueError, match="source"):
        compare_trace(report, Path(".cache/upstream/microduck/policies"))


@pytest.mark.parametrize("inference", ["ONNX Runtime external actor", None])
def test_player_model_inventory_does_not_prove_the_recorded_actions_ran_barracuda(actual_report, inference):
    from agenticrobot_bridge.trace_policy_parity import compare_trace

    report = copy.deepcopy(actual_report)
    if inference is None:
        del report["results"][0]["inference"]
    else:
        report["results"][0]["inference"] = inference
    with pytest.raises(ValueError, match="Barracuda"):
        compare_trace(report, Path(".cache/upstream/microduck/policies"))


@pytest.mark.parametrize("field,value", [("graphSha256", ""), ("convertedOnnxSha256", None),
                                       ("graphVerified", "false")])
def test_all_graph_identity_fields_must_be_verified_not_truthy(actual_report, field, value):
    from agenticrobot_bridge.trace_policy_parity import compare_trace

    report = copy.deepcopy(actual_report)
    report["identity"]["models"][0][field] = value
    with pytest.raises(ValueError, match="source"):
        compare_trace(report, Path(".cache/upstream/microduck/policies"))


def test_actual_frame_backend_overrides_a_false_episode_label(actual_report):
    from agenticrobot_bridge.trace_policy_parity import compare_trace

    report = copy.deepcopy(actual_report)
    report["results"][0]["episodes"][0][1]["inferenceBackend"] = "external actor / real PhysX"
    with pytest.raises(ValueError, match="backend"):
        compare_trace(report, Path(".cache/upstream/microduck/policies"))


def test_cli_writes_hash_bound_diagnostic_not_behavior_acceptance(actual_report, tmp_path, monkeypatch):
    import hashlib
    from agenticrobot_bridge.trace_policy_parity import main

    source, output = tmp_path / "trace.json", tmp_path / "result.json"
    source.write_text(json.dumps(actual_report), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["trace-policy-parity", "--trace", str(source), "--output", str(output)])
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["traceSha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["numericalParityPassed"] is True
    assert report["behaviorAccepted"] is False


@pytest.mark.parametrize("mutation,reason", [("identity", "Player identity"), ("slot", "inference slot"),
                                           ("nonfinite", "Non-finite"), ("engine", "engine identities")])
def test_invalid_trace_is_rejected_before_a_parity_claim(actual_report, mutation, reason):
    from agenticrobot_bridge.trace_policy_parity import compare_trace

    report = copy.deepcopy(actual_report)
    frame = report["results"][0]["episodes"][0][1]
    if mutation == "identity":
        del report["identity"]["buildGuid"]
    elif mutation == "slot":
        del frame["inferenceSlot"]
    elif mutation == "nonfinite":
        frame["action"][0] = float("nan")
    else:
        frame["engine"] = "MuJoCo"
    with pytest.raises(ValueError, match=reason):
        compare_trace(report, Path(".cache/upstream/microduck/policies"))
