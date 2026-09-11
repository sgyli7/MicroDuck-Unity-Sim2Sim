import numpy as np
import pytest

from agenticrobot_bridge.physx_training import PhysXVecEnv, locomotion_reward


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


def test_zero_turn_command_penalizes_heading_drift_from_the_episode_initial_heading():
    straight = frame(velocity=(0, 0, 0.2))
    veered = frame(velocity=(0.2, 0, 0))
    veered["rootRotation"] = [0, 2 ** -0.5, 0, 2 ** -0.5]
    good, _ = locomotion_reward([straight], np.zeros((1, 14)), 1, initial_yaw=np.array([0.0]))
    bad, _ = locomotion_reward([veered], np.zeros((1, 14)), 1, initial_yaw=np.array([0.0]))
    rotated_start, _ = locomotion_reward([veered], np.zeros((1, 14)), 1,
                                       initial_yaw=np.array([np.pi / 2]))
    assert good[0] > bad[0] + 1
    assert good == pytest.approx(rotated_start)


def test_heading_is_the_unity_forward_projection_even_with_nonzero_roll():
    rolled = frame(upright=0.5)
    # Unity yaw 45 degrees followed by local roll 60 degrees, XYZW.
    rolled["rootRotation"] = [0.191341716, 0.331413574, 0.461939766, 0.800103145]
    assert PhysXVecEnv._yaw([rolled])[0] == pytest.approx(np.pi / 4, abs=1e-8)
