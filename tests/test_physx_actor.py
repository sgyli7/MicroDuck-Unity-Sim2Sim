"""Run with the pinned GPU training environment, not the lightweight bridge venv."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from agenticrobot_bridge.physx_actor import ExecutionEnvelope, actor_from_onnx, export_actor


POLICIES = Path(".cache/upstream/microduck/policies")
MODELS = sorted(POLICIES.glob("*.onnx"))


def test_all_nine_official_policies_are_present():
    assert len(MODELS) == 9


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.stem)
def test_original_weights_and_export_match_onnx(path, tmp_path):
    actor = actor_from_onnx(path)
    rng = np.random.default_rng(9173)
    observations = rng.normal(0, 0.2, size=(32, 61)).astype(np.float32)
    observations[:, 5] = -1
    reference = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    expected = np.concatenate([
        reference.run(None, {reference.get_inputs()[0].name: row[None]})[0]
        for row in observations
    ])
    with torch.no_grad():
        actual = actor(torch.from_numpy(observations)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=2e-5)
    destination = tmp_path / (path.stem + "_PhysX.onnx")
    export_actor(actor, destination)
    restored = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    exported = restored.run(None, {"observation": observations})[0]
    np.testing.assert_allclose(exported, expected, atol=1e-4, rtol=2e-5)


def test_export_refuses_to_overwrite_original_policy(tmp_path):
    actor = actor_from_onnx(MODELS[0])
    with pytest.raises(ValueError, match="_PhysX"):
        export_actor(actor, tmp_path / MODELS[0].name)


def test_actor_can_restore_the_same_immutable_bytes_hashed_for_inference():
    payload = MODELS[0].read_bytes()
    from_bytes = actor_from_onnx(payload)
    from_path = actor_from_onnx(MODELS[0])
    with torch.no_grad():
        observed = torch.zeros(2, 61)
        torch.testing.assert_close(from_bytes(observed), from_path(observed), rtol=0, atol=0)


def test_adapted_export_has_the_same_action_execution_envelope_as_training(tmp_path):
    actor = actor_from_onnx(MODELS[0])
    with torch.no_grad():
        actor.mlp[-1].bias.fill_(100)
    bounded = ExecutionEnvelope(actor)
    destination = tmp_path / "bounded_PhysX.onnx"
    export_actor(bounded, destination)
    runtime = ort.InferenceSession(str(destination))
    observation = np.zeros((1, 61), np.float32)
    actual = runtime.run(None, {"observation": observation})[0]
    assert np.abs(actual).max() <= 5
    with torch.no_grad():
        expected = actor(torch.from_numpy(observation)).clamp(-5, 5).numpy()
    np.testing.assert_array_equal(actual, expected)
