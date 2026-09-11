import pytest
import numpy as np

from agenticrobot_bridge.sim2sim_comparison import compare_reports


def frame(t, position, engine):
    result = {"engine": engine, "timeSeconds": t, "rootPosition": position,
              "upright": 1.0, "policyObservation": [0.0] * 61, "action": [0.0] * 14}
    return result


def reports():
    reference = {"engine": "MuJoCo", "episodes": [{"slot": 2, "completed": True,
        "requestedSeconds": 0.02, "frames": [frame(0, [0, 0, .125], "MuJoCo"),
        frame(.005, [.005, 0, .125], "MuJoCo"), frame(.02, [.02, 0, .125], "MuJoCo")]}]}
    target = {"identity": {"engine": "PhysX"}, "results": [{"slot": 2,
        "requestedSeconds": .02, "fault": None, "modelIdentityVerified": False,
        "episodes": [[frame(0, [-.45, .125, 0], "PhysX"),
                       frame(.02, [-.45, .125, .02], "PhysX")]]}]}
    return reference, target


def test_compare_aligns_simulation_time_and_coordinate_basis_not_frame_index():
    result = compare_reports(*reports())
    stand = result["skills"][1]
    assert stand["matchedSamples"] == 2
    assert stand["maximumRelativeRootDistance"] == pytest.approx(0)
    assert stand["maximumPolicyInputDifference"] == 0
    assert stand["behaviorAccepted"] is False
    assert "experiment_identity_not_verified" in stand["limitations"]
    assert result["physxAcceptedCount"] == 0
    assert sum(skill["status"] == "missing" for skill in result["skills"]) == 8


def test_reference_failure_or_target_fault_never_counts_as_a_complete_pair():
    reference, target = reports()
    target["results"][0]["fault"] = {"reason": "controller failed"}
    result = compare_reports(reference, target)["skills"][1]
    assert result["status"] == "incomplete"
    assert result["behaviorAccepted"] is False


def test_wrong_engine_and_duplicate_slots_are_rejected():
    reference, target = reports()
    target["identity"]["engine"] = "MuJoCo"
    with pytest.raises(ValueError, match="PhysX"):
        compare_reports(reference, target)
    reference, target = reports()
    reference["episodes"].append(reference["episodes"][0])
    with pytest.raises(ValueError, match="duplicate"):
        compare_reports(reference, target)


def test_missing_time_sample_is_reported_not_interpolated_as_exact_parity():
    reference, target = reports()
    reference["episodes"][0]["frames"][-1]["timeSeconds"] = .025
    stand = compare_reports(reference, target)["skills"][1]
    assert stand["unmatchedTargetSamples"] == 1
    assert stand["status"] == "incomplete"


def test_missing_initial_frame_cannot_define_a_different_comparison_origin():
    reference, target = reports()
    target["results"][0]["episodes"][0].pop(0)
    stand = compare_reports(reference, target)["skills"][1]
    assert stand["status"] == "incomplete"
    assert "missing_initial_state" in stand["limitations"]
    assert stand.get("maximumRelativeRootDistance") is None


def test_sixty_second_float_timestamp_quantization_is_not_a_missing_tick():
    reference, target = reports()
    reference["episodes"][0]["requestedSeconds"] = 60
    target["results"][0]["requestedSeconds"] = 60
    source, destination = [], []
    for i in range(3001):
        seconds = i * .02
        source.append(frame(seconds, [0, 0, .125], "MuJoCo"))
        sample = frame(float(np.float32(i * 4) * np.float32(.005)), [0, .125, 0], "PhysX")
        sample["physicsSteps"] = i * 4
        destination.append(sample)
    reference["episodes"][0]["frames"] = source
    target["results"][0]["episodes"] = [destination]
    stand = compare_reports(reference, target)["skills"][1]
    assert stand["matchedSamples"] == 3001
    assert stand["unmatchedTargetSamples"] == 0


def test_nonfinite_upright_is_rejected_before_error_aggregation():
    reference, target = reports()
    target["results"][0]["episodes"][0][0]["upright"] = float("nan")
    with pytest.raises(ValueError, match="upright"):
        compare_reports(reference, target)
