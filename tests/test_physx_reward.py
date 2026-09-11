import numpy as np
import pytest

from agenticrobot_bridge.physx_training import locomotion_reward


def frame(height=0.125, upright=1.0, velocity=(0, 0, 0)):
    return {"rootPosition": [0, height, 0], "rootRotation": [0, 0, 0, 1], "upright": upright,
            "rootVelocity": list(velocity), "observation": [0.0] * 61, "healthy": True}


def test_standing_is_rewarded_above_falling_or_drifting():
    actions = np.zeros((1, 14))
    stable, fallen = locomotion_reward([frame()], actions, 2)
    bad, terminal = locomotion_reward([frame(0.04, 0.1)], actions, 2)
    drift, _ = locomotion_reward([frame(velocity=(1, 0, 0))], actions, 2)
    assert not fallen[0] and terminal[0]
    assert stable[0] > drift[0] > bad[0]


def test_walk_reward_uses_actual_forward_velocity_and_rejects_unimplemented_roles():
    good, _ = locomotion_reward([frame(velocity=(0, 0, 0.2))], np.zeros((1, 14)), 1)
    backward, _ = locomotion_reward([frame(velocity=(0, 0, -0.2))], np.zeros((1, 14)), 1)
    assert good[0] > backward[0]
    with pytest.raises(ValueError, match="not implemented"):
        locomotion_reward([frame()], np.zeros((1, 14)), 9)


def test_velocity_tracking_is_yaw_rotation_invariant():
    straight = frame(velocity=(0, 0, 0.2))
    turned = frame(velocity=(0.2, 0, 0))
    turned["rootRotation"] = [0, 2 ** -0.5, 0, 2 ** -0.5]
    a, _ = locomotion_reward([straight], np.zeros((1, 14)), 1)
    b, _ = locomotion_reward([turned], np.zeros((1, 14)), 1)
    assert a == pytest.approx(b)
