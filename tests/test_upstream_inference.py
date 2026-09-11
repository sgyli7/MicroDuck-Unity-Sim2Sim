import hashlib
from pathlib import Path

import numpy as np
import pytest


def test_original_upstream_controller_loads_on_windows_without_changing_policy_methods():
    from agenticrobot_bridge.upstream_inference import load_controller

    module, provenance = load_controller(Path('.'))
    source = Path('.cache/upstream/microduck_rl/scripts/infer_policy.py').read_bytes()
    assert provenance['scriptSha256'] == hashlib.sha256(source).hexdigest()
    assert provenance['removedImports'] == ['termios', 'tty']
    assert provenance['policyClassAstUnchanged'] is True
    assert len(module.DEFAULT_POSE) == 14


def test_official_stand_uses_cli_torque_limit_and_unfiltered_original_actions():
    from agenticrobot_bridge.upstream_inference import run_case
    from agenticrobot_bridge.shared_experiment import make_batch

    case = make_batch('.cache/upstream/microduck/policies', 92317)['cases'][1]
    case['physicsSteps'] = 40
    result = run_case(Path('.'), case)
    assert result['engine'] == 'MuJoCo'
    assert result['role'] == 'official-source-headless-adapter'
    assert result['coordinateBasis'] == 'Unity-Xright-Yup-Zforward'
    assert result['behaviorAccepted'] is False
    assert result['completed'] is True and result['frames'][-1]['physicsSteps'] == 40
    assert .63 < result['settings']['actuatorTorqueLimit'] < .65
    assert result['settings']['actionScale'] == 1.0
    frame = result['frames'][1]
    assert len(frame['policyObservation']) == 61 and len(frame['action']) == 14
    home = result['settings']['home']
    np.testing.assert_allclose(frame['targets'], np.asarray(home) + frame['action'], atol=1e-7)


def test_source_phase_ends_at_point_seven_and_returns_to_standing():
    from agenticrobot_bridge.upstream_inference import run_case
    from agenticrobot_bridge.shared_experiment import make_batch

    case = make_batch('.cache/upstream/microduck/policies', 92317)['cases'][3]
    result = run_case(Path('.'), case)
    assert result['completed'] is True
    slots = {frame['inferenceSlot'] for frame in result['frames']}
    assert slots == {2, 4}
    assert result['frames'][-1]['inferenceSlot'] == 2
    assert result['settings']['phasePeriod'] == 4.0


@pytest.mark.parametrize('slot', range(1, 10))
def test_nine_original_models_execute_their_upstream_lifecycle_without_claiming_acceptance(slot):
    from agenticrobot_bridge.upstream_inference import run_case
    from agenticrobot_bridge.shared_experiment import make_batch

    case = make_batch('.cache/upstream/microduck/policies', 92317)['cases'][slot - 1]
    result = run_case(Path('.'), case)
    assert result['completed'] is True and result['fault'] is None
    assert result['frames'][-1]['physicsSteps'] == case['physicsSteps']
    assert slot in {frame['inferenceSlot'] for frame in result['frames']}
    assert len(result['provenance']['configuredCompiledModelSha256']) == 64
    assert len(result['provenance']['robotTree']) == 40
    assert result['behaviorAccepted'] is False
    assert all(np.isfinite(frame['action']).all() for frame in result['frames'])
