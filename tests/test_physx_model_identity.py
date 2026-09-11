import pytest

from agenticrobot_bridge.physx_evaluate import bound_model_identity


def test_missing_or_unverifiable_embedded_graph_is_not_an_actual_onnx_hash():
    assert bound_model_identity({}, 2, "a" * 64) == (None, False)
    model = {"slot": 2, "sourceSha256": "a" * 64, "convertedOnnxSha256": "b" * 64,
             "graphSha256": "c" * 64, "graphVerified": True}
    identity = {"buildGuid": "actual-player", "models": [model]}
    assert bound_model_identity(identity, 2, "a" * 64) == (model, True)
    model["graphVerified"] = False
    assert bound_model_identity(identity, 2, "a" * 64)[1] is False
    model["graphVerified"] = True
    model["graphSha256"] = "invalid"
    assert bound_model_identity(identity, 2, "a" * 64)[1] is False


def test_mismatched_or_duplicate_source_identity_fails_before_rollout():
    model = {"slot": 2, "sourceSha256": "b" * 64}
    with pytest.raises(ValueError, match="differs"):
        bound_model_identity({"models": [model]}, 2, "a" * 64)
    with pytest.raises(ValueError, match="duplicate"):
        bound_model_identity({"models": [model, model]}, 2, "b" * 64)
