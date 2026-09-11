import copy

import pytest

from agenticrobot_bridge import shared_experiment


def interval():
    def frame(tick):
        return {'engine': 'PhysX', 'physicsSteps': tick, 'timeSeconds': tick * .005,
                'inferenceSlot': 2, 'rootPosition': [0, .125, 0],
                'rootRotation': [0, 0, 0, 1], 'rootVelocity': [0.] * 3,
                'rootAngularVelocity': [0.] * 3, 'mouthTipPosition': [0.] * 3,
                'jointPosition': [0.] * 14, 'jointVelocity': [0.] * 14,
                'passiveWheelVelocity': [], 'upright': 1., 'ballActive': False,
                'ballPosition': [0.] * 3, 'ballVelocity': [0.] * 3, 'contacts': []}
    return {**frame(4),
            'physicsTrace': [frame(tick) for tick in range(1, 5)]}


def test_dense_trace_requires_actual_contiguous_physics_samples_not_interpolation():
    validate = getattr(shared_experiment, 'validated_microsteps', None)
    assert callable(validate), 'Missing fail-closed dense trace validation'
    source = interval()
    assert [f['physicsSteps'] for f in validate(source, 0)] == [1, 2, 3, 4]
    for missing in range(4):
        bad = copy.deepcopy(source)
        del bad['physicsTrace'][missing]
        with pytest.raises(ValueError, match='Dense physics'):
            validate(bad, 0)
    for key, value in [('physicsSteps', 42), ('timeSeconds', 99), ('inferenceSlot', 9),
                       ('engine', 'MuJoCo'), ('jointPosition', [float('nan')] * 14)]:
        bad = copy.deepcopy(source)
        bad['physicsTrace'][1][key] = value
        with pytest.raises(ValueError, match='Dense physics'):
            validate(bad, 0)
    source['physicsTrace'][-1]['rootPosition'] = [1, 2, 3]
    with pytest.raises(ValueError, match='Dense physics'):
        validate(source, 0)


@pytest.mark.parametrize('field,value', [('rootVelocity', [float('inf')] * 3),
                                       ('rootRotation', [0.] * 4), ('upright', float('nan')),
                                       ('contacts', [{'point': [0, 0, 0]}])])
def test_dense_trace_rejects_invalid_measurements_beyond_root_position(field, value):
    source = interval()
    source['physicsTrace'][0][field] = value
    with pytest.raises(ValueError, match='Dense physics'):
        shared_experiment.validated_microsteps(source, 0)


@pytest.mark.parametrize('field', ['jointPosition', 'jointVelocity', 'rootVelocity',
                                  'rootAngularVelocity', 'rootRotation'])
def test_dense_trace_endpoint_must_match_the_same_control_boundary(field):
    source = interval()
    source[field][0] = .2
    with pytest.raises(ValueError, match='Dense physics endpoint'):
        shared_experiment.validated_microsteps(source, 0)


def test_roller_measurement_requires_all_four_passive_wheels():
    source = interval()
    source['inferenceSlot'] = 7
    for item in source['physicsTrace']:
        item['inferenceSlot'] = 7
    with pytest.raises(ValueError, match='passive wheel'):
        shared_experiment.validated_microsteps(source, 0)


def test_endpoint_vectors_cannot_broadcast_a_single_value_over_fourteen_joints():
    source = interval()
    source['jointPosition'] = [0.]
    with pytest.raises(ValueError, match='Dense physics'):
        shared_experiment.validated_microsteps(source, 0)
