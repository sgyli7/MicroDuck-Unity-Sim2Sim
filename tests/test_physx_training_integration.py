"""Real Player + real RSL-RL optimizer + CUDA + checkpoint + ONNX deployment test.

Requires the two-environment Player on port 62101. This proves the training chain,
not behavioral improvement or nine-skill acceptance.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from agenticrobot_bridge.physx_training import train


def test_real_physx_ppo_update_resume_and_export(tmp_path):
    source = Path(__file__).parents[1] / ".cache/upstream/microduck/policies/alpha_stand.onnx"
    args = argparse.Namespace(output=str(tmp_path / "first"), policy=str(source), slot=2,
                              port=62101, device="cuda", seed=94021, learning_rate=1e-5,
                              episode_seconds=0.1, iterations=2, save_interval=1, resume=None)
    assert torch.cuda.is_available(), "GPU training capability is required by this gate"
    train(args)
    checkpoint = Path(args.output) / "checkpoint_000001.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["iteration"] == 1 and saved["slot"] == 2
    metadata = json.loads((Path(args.output) / "run.json").read_text())
    assert metadata["engine"] == metadata["sampling"]["engine"] == "PhysX"
    assert metadata["behavior_accepted"] is False
    assert metadata["sampling"]["resetInitializationDt"] == 0
    records = [json.loads(line) for line in (Path(args.output) / "training.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert any(record["episodes"] for record in records)
    args.resume = str(checkpoint)
    args.output = str(tmp_path / "resumed")
    args.iterations = 1
    args.learning_rate = 2e-5
    train(args)
    resumed = torch.load(Path(args.output) / "checkpoint_000002.pt", map_location="cpu", weights_only=False)
    assert all(group["lr"] == args.learning_rate for group in resumed["optimizer_state_dict"]["param_groups"])
    resume_metadata = json.loads((Path(args.output) / "run.json").read_text())
    assert resume_metadata["resume"]["checkpointSha256"]
    model = ort.InferenceSession(str(Path(args.output) / "alpha_stand_PhysX.onnx"),
                                providers=["CPUExecutionProvider"])
    values = model.run(None, {model.get_inputs()[0].name: np.zeros((1, 61), np.float32)})[0]
    assert values.shape == (1, 14) and np.isfinite(values).all()
    assert np.abs(values).max() <= 5
