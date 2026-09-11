import copy

import numpy as np
import pytest


def reference():
    return {'engine': 'MuJoCo', 'coordinateBasis': 'right-handed X-forward Y-left Z-up; quaternion WXYZ',
            'buildGuid': 'test-build', 'models': [
        {'slot': 1, 'sourceSha256': 'a' * 64, 'convertedOnnxSha256': 'b' * 64,
         'graphSha256': 'c' * 64, 'graphVerified': True}], 'episodes': [
        {'slot': 1, 'experimentId': 'unit-motion', 'experimentSha256': 'd' * 64,
         'completed': True, 'inference': 'Barracuda', 'requestedSeconds': .04,
         'frames': [{'engine': 'MuJoCo', 'physicsSteps': tick, 'timeSeconds': tick * .005,
                     'inferenceBackend': 'MuJoCo 3.12 + Barracuda 3.0.1 CPU',
                     'healthy': True, 'upright': 1., 'fault': '',
                     'activeSlot': 1, 'inferenceSlot': 1 if tick else 0,
                     'rootPosition': [tick * .005 * .08, 0, .115],
                     'rootQuaternionWxyz': [1, 0, 0, 0], 'rootVelocity': [.08, 0, 0],
                     'jointPosition': [0] * 14, 'jointVelocity': [0] * 14}
                    for tick in range(9)]}]}


def frame():
    observation = np.zeros(61)
    observation[5] = -1
    return {'engine': 'PhysX', 'rootPosition': [0, .115, .0016],
            'rootVelocity': [0, 0, .08], 'rootRotation': [0, 0, 0, 1],
            'jointPosition': [0] * 14, 'jointVelocity': [0] * 14,
            'observation': observation.tolist(), 'upright': 1., 'healthy': True}


def test_reference_reward_targets_measured_motion_not_conflicting_command_speed():
    from agenticrobot_bridge.motion_reference import RecordedMotionReference

    source = reference()
    unchanged = copy.deepcopy(source)
    motion = RecordedMotionReference(source, 1, 'a' * 64, 'unit-motion')
    actual = frame()
    fast = copy.deepcopy(actual)
    fast['rootVelocity'][2] = .2
    good, failed = motion.reward([actual], [1], np.zeros((1, 3)), np.zeros(1))
    bad, _ = motion.reward([fast], [1], np.zeros((1, 3)), np.zeros(1))
    assert not failed[0] and good[0] > bad[0] + .3
    assert source == unchanged


def test_reference_reward_is_invariant_to_episode_world_translation_and_yaw():
    from agenticrobot_bridge.motion_reference import RecordedMotionReference

    motion = RecordedMotionReference(reference(), 1, 'a' * 64, 'unit-motion')
    straight = frame()
    rotated = frame()
    rotated.update(rootPosition=[2.0016, .115, 3], rootVelocity=[.08, 0, 0],
                   rootRotation=[0, 2 ** -.5, 0, 2 ** -.5])
    a, _ = motion.reward([straight], [1], np.zeros((1, 3)), np.zeros(1))
    b, _ = motion.reward([rotated], [1], np.array([[2, 0, 3]]), np.array([np.pi / 2]))
    np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.parametrize('problem', ['engine', 'source', 'incomplete', 'missing_tick', 'nan', 'basis', 'unhealthy', 'inference_slot'])
def test_invalid_reference_cannot_silently_drive_training_rewards(problem):
    from agenticrobot_bridge.motion_reference import RecordedMotionReference

    source = reference()
    if problem == 'engine':
        source['engine'] = 'PhysX'
    elif problem == 'source':
        source['models'][0]['sourceSha256'] = 'f' * 64
    elif problem == 'incomplete':
        source['episodes'][0]['completed'] = False
    elif problem == 'missing_tick':
        source['episodes'][0]['frames'].pop(4)
    elif problem == 'basis':
        source['coordinateBasis'] = 'Unity-Xright-Yup-Zforward'
    elif problem == 'unhealthy':
        source['episodes'][0]['frames'][3]['healthy'] = False
    elif problem == 'inference_slot':
        source['episodes'][0]['frames'][3]['inferenceSlot'] = 2
    else:
        source['episodes'][0]['frames'][2]['jointPosition'][0] = float('nan')
    with pytest.raises(ValueError):
        RecordedMotionReference(source, 1, 'a' * 64, 'unit-motion')


def test_falls_and_exhausted_reference_are_not_clamped_to_a_successful_last_pose():
    from agenticrobot_bridge.motion_reference import RecordedMotionReference

    motion = RecordedMotionReference(reference(), 1, 'a' * 64, 'unit-motion')
    fallen = frame()
    fallen.update(upright=0., rootPosition=[0, .02, 0])
    _, failed = motion.reward([fallen], [1], np.zeros((1, 3)), np.zeros(1))
    assert failed[0]
    with pytest.raises(ValueError, match='horizon'):
        motion.reward([frame()], [3], np.zeros((1, 3)), np.zeros(1))


def test_end_of_finite_reference_is_terminal_but_earlier_cutoff_is_a_timeout():
    import torch
    from agenticrobot_bridge.physx_training import PhysXVecEnv
    from agenticrobot_bridge.motion_reference import RecordedMotionReference

    class Client:
        identity = {'numEnvs': 1}

        def reset(self, *args, **kwargs):
            return [frame()]

        def step(self, *args):
            return [frame()]

    motion = RecordedMotionReference(reference(), 1, 'a' * 64, 'unit-motion')
    whole = PhysXVecEnv(Client(), 1, 'cpu', .04, motion)
    whole.step(torch.zeros(1, 14))
    _, _, done, extra = whole.step(torch.zeros(1, 14))
    assert done.item() is True
    assert extra['time_outs'].item() is False
    shorter = PhysXVecEnv(Client(), 1, 'cpu', .02, motion)
    _, _, done, extra = shorter.step(torch.zeros(1, 14))
    assert done.item() is True and extra['time_outs'].item() is True
