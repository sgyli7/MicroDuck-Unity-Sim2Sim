"""Actual RSL-RL storage must receive V(final state), exactly once, on truncation."""

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from agenticrobot_bridge import physx_training


def observations(first_value):
    values = torch.zeros(2, 61)
    values[:, 0] = first_value
    return TensorDict({"policy": values}, batch_size=[2])


def test_terminal_value_is_bootstrapped_once_and_failure_is_not_bootstrapped():
    cfg = physx_training.ppo_config()
    algorithm = PPO.construct_algorithm(observations(2),
                                        SimpleNamespace(num_envs=2, num_actions=14), cfg, "cpu")
    algorithm.critic.obs_normalization = False
    algorithm.critic.obs_normalizer = torch.nn.Identity()
    algorithm.critic.mlp = torch.nn.Linear(61, 1, bias=False)
    with torch.no_grad():
        algorithm.critic.mlp.weight.zero_()
        algorithm.critic.mlp.weight[0, 0] = 1
        algorithm.act(observations(2))
    # Environment 0 times out at value 10; environment 1 falls at the same value.
    # The returned observation has already reset to 0 and must not be used here.
    assert hasattr(physx_training, "record_transition"), "Missing final-state timeout adapter"
    with torch.no_grad():
        physx_training.record_transition(
            algorithm, observations(0), torch.ones(2), torch.ones(2, dtype=torch.bool),
            {"time_outs": torch.tensor([True, False]),
             "terminal_observations": observations(10)},
        )
    assert float(algorithm.storage.rewards[0, 0, 0]) == pytest.approx(10.9)
    assert float(algorithm.storage.rewards[0, 1, 0]) == pytest.approx(1.0)
