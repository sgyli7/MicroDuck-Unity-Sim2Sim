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
import pytest

from agenticrobot_bridge.physx_training import train


def test_actual_reference_reward_trains_only_on_independent_physx_samples(tmp_path):
    from agenticrobot_bridge.shared_experiment import make_batch, run_players, write_new

    root = Path(__file__).parents[1]
    batch = make_batch(root / '.cache/upstream/microduck/policies', 92318)
    batch['cases'] = [batch['cases'][0]]
    batch['cases'][0]['physicsSteps'] = 40
    spec = tmp_path / 'reference-spec.json'
    write_new(spec, batch)
    evidence = tmp_path / 'paired'
    run_players(spec, evidence,
                root.parent / 'AgenticRobotGame-MuJoCoMeasured-20260909/Builds/MeasuredReference/AgenticRobotGame-MuJoCoReference.exe',
                root / 'Builds/Windows64/AgenticRobotGame.exe', 62103)
    args = argparse.Namespace(output=str(tmp_path / 'motion-ppo'),
                              policy=str(root / '.cache/upstream/microduck/policies/alpha_walking.onnx'),
                              slot=1, port=62101, device='cuda', seed=94023, learning_rate=1e-6,
                              episode_seconds=.1, iterations=2, save_interval=1, resume=None,
                              reference_trace=str(evidence / 'reference.json'),
                              reference_case=batch['cases'][0]['id'])
    train(args)
    metadata = json.loads((Path(args.output) / 'run.json').read_text())
    assert metadata['env']['reward_version'] == 'measured-motion-v1'
    assert metadata['env']['reference']['trace_sha256']
    assert metadata['sampling']['engine'] == 'PhysX'
    assert metadata['behavior_accepted'] is False
    args.resume = str(Path(args.output) / 'checkpoint_000001.pt')
    args.output = str(tmp_path / 'changed-horizon')
    args.episode_seconds = .2
    with pytest.raises(ValueError, match='horizon'):
        train(args)


def test_critic_warmup_keeps_original_actor_exact_and_resume_starts_actor_updates(tmp_path):
    from agenticrobot_bridge.physx_actor import actor_from_onnx

    source = Path('.cache/upstream/microduck/policies/alpha_walking.onnx')
    original = actor_from_onnx(source)
    args = argparse.Namespace(output=str(tmp_path / 'warmup'), policy=str(source), slot=1,
                              port=62101, device='cuda', seed=94022, learning_rate=1e-6,
                              critic_learning_rate=1e-4, initial_std=.025,
                              critic_warmup_iterations=2, episode_seconds=.1,
                              iterations=2, save_interval=1, resume=None)
    train(args)
    checkpoint = Path(args.output) / 'checkpoint_000001.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    earlier = torch.load(Path(args.output) / 'checkpoint_000000.pt', map_location='cpu', weights_only=True)
    for name, weight in original.mlp.state_dict().items():
        assert torch.equal(saved['actor_state_dict']['mlp.' + name], weight), name
    assert all(torch.equal(saved['actor_state_dict'][name], value)
               for name, value in earlier['actor_state_dict'].items())
    assert any(not torch.equal(saved['critic_state_dict'][name], value)
               for name, value in earlier['critic_state_dict'].items() if name.startswith('mlp.'))
    metadata = json.loads((Path(args.output) / 'run.json').read_text())
    assert metadata['ppo']['actor']['distribution_cfg']['init_std'] == .025
    records = [json.loads(line) for line in (Path(args.output) / 'training.jsonl').read_text().splitlines()]
    assert all(record['critic_warmup'] for record in records)
    assert all(record['learning_rates'] == [0.0, 1e-4] for record in records)
    args.resume, args.output, args.iterations = str(checkpoint), str(tmp_path / 'after-warmup'), 1
    train(args)
    resumed = torch.load(Path(args.output) / 'checkpoint_000002.pt', map_location='cpu', weights_only=True)
    assert any(not torch.equal(resumed['actor_state_dict']['mlp.' + name], weight)
               for name, weight in original.mlp.state_dict().items())
    assert [group['lr'] for group in resumed['optimizer_state_dict']['param_groups']] == [1e-6, 1e-4]


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
    # Exercise migration of the earlier single parameter-group checkpoint layout.
    groups = saved['optimizer_state_dict']['param_groups']
    saved['optimizer_state_dict']['param_groups'] = [dict(groups[0], params=groups[0]['params'] + groups[1]['params'])]
    saved.pop('optimizer_group_schema', None)
    legacy_checkpoint = Path(args.output) / 'legacy-layout.pt'
    torch.save(saved, legacy_checkpoint)
    args.resume = str(legacy_checkpoint)
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
