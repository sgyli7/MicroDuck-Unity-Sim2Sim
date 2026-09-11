"""Explicit target: requires a running Player with -physxPort 62101 -physxEnvs 2.

This is deliberately not skipped when the Player is missing; absence is failure.
"""

import pytest

from agenticrobot_bridge.physx_client import PhysXClient


def test_real_player_batch_reset_step_and_transactional_input_rejection():
    with PhysXClient(timeout=10) as client:
        assert client.identity["engine"] == "PhysX"
        assert client.identity["numEnvs"] == 2
        initial = client.reset(2)
        assert len(initial) == 2
        stepped = client.step([[0.1] * 14, [0.0] * 14])
        assert all(frame["physicsSteps"] == 4 for frame in stepped)
        assert all(len(frame["observation"]) == 61 for frame in stepped)
        assert stepped[0]["observation"][34:48] == pytest.approx([0.1] * 14)
        assert stepped[0]["jointPosition"] != initial[0]["jointPosition"]
        with pytest.raises(RuntimeError, match="envelope"):
            client.step([[0.1] * 14, [6.0] * 14])
        stepped = client.step([[0.0] * 14, [0.0] * 14])
        assert all(frame["physicsSteps"] == 8 for frame in stepped)
        reset = client.reset(2, indices=[1])
        assert len(reset) == 1 and reset[0]["physicsSteps"] == 0
        stepped = client.step([[0.0] * 14, [0.0] * 14])
        assert [frame["physicsSteps"] for frame in stepped] == [12, 4]


def test_short_training_episode_preserves_terminal_observation_before_auto_reset():
    import torch
    from agenticrobot_bridge.physx_training import PhysXVecEnv

    with PhysXClient(timeout=10) as client:
        env = PhysXVecEnv(client, 2, device="cpu", episode_seconds=0.02)
        next_obs, _, dones, extras = env.step(torch.ones(2, 14) * 0.1)
        assert dones.all() and extras["time_outs"].all()
        assert torch.all(extras["terminal_observations"]["policy"][:, 34:48] == 0.1)
        assert torch.all(next_obs["policy"][:, 34:48] == 0)
        assert torch.all(env.episode_length_buf == 0)


def test_internal_barracuda_steps_without_external_actions_and_rejects_overrides():
    with PhysXClient(timeout=10) as client:
        client.reset(2, external=False)
        stepped = client.step()
        assert all(frame["physicsSteps"] == 4 for frame in stepped)
        assert all(len(frame["policyObservation"]) == 61 for frame in stepped)
        with pytest.raises(RuntimeError, match="takes no actions"):
            client.step([[0.0] * 14] * 2)


def test_reset_reports_home_targets_not_the_previous_episode_targets():
    with PhysXClient(timeout=10) as client:
        client.reset(2)
        client.step([[0.2] * 14] * 2)
        reset = client.reset(2)
        for frame in reset:
            assert frame["targets"] == pytest.approx(frame["jointPosition"], abs=1e-6)


def test_internal_evaluation_does_not_attribute_an_unverified_source_hash():
    from pathlib import Path
    from agenticrobot_bridge.physx_evaluate import rollout

    policy = Path(__file__).parents[1] / ".cache/upstream/microduck/policies/alpha_stand.onnx"
    with PhysXClient(timeout=10) as client:
        result = rollout(client, 2, policy, 0.02, internal=True)
    assert result["modelSha256"] is None
    assert result["expectedSourceSha256"]
    assert result["modelIdentityVerified"] is False


def test_critic_gets_current_physx_privileged_state_without_changing_actor_inputs():
    import torch
    from agenticrobot_bridge.physx_training import PhysXVecEnv

    with PhysXClient(timeout=10) as client:
        env = PhysXVecEnv(client, 1, device="cpu", episode_seconds=6)
        observed = env.get_observations()
        assert observed["policy"].shape == (2, 61)
        assert observed["critic"].shape == (2, 68)
        assert torch.equal(observed["critic"][:, :61], observed["policy"])
        assert torch.all(observed["critic"][:, -1] == 0)
        observed, _, _, _ = env.step(torch.zeros(2, 14))
        assert torch.all(observed["critic"][:, -1] > 0)
