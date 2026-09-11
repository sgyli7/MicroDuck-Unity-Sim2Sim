"""Diagnostic comparison of independent simulations; never a behavior pass gate."""

import argparse
import json
from pathlib import Path

import numpy as np

POLICIES = ["alpha_walking", "alpha_stand", "alpha_sitstand", "alpha_ground_pick",
            "ball_kick_left", "ball_kick_right", "roller", "roller_crouch", "roulade"]


def _indexed(rows):
    result = {}
    for row in rows:
        slot = row["slot"]
        if slot not in range(1, 10) or slot in result:
            raise ValueError("Invalid or duplicate skill slot")
        result[slot] = row
    return result


def _vector(value, length):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"Expected finite vector of length {length}")
    return array


def _time_index(frames, engine):
    ticks = []
    for frame in frames:
        if frame.get("engine") != engine:
            raise ValueError("Mixed-engine trace")
        value = float(frame["timeSeconds"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Simulation timestamps must be finite and strictly increasing")
        tick = frame.get("physicsSteps", round(value / 0.005))
        if not isinstance(tick, int) or tick < 0 or (ticks and tick <= ticks[-1]):
            raise ValueError("Physics ticks must be nonnegative, integral and strictly increasing")
        # Unity publishes float timestamps; use integer physics ticks for identity,
        # allowing only the timestamp's own float32 representation error.
        tolerance = max(1e-8, abs(float(np.spacing(np.float32(value)))) * 1.1)
        if abs(value - tick * 0.005) > tolerance:
            raise ValueError("Timestamp does not match the 200 Hz physics tick")
        if not np.isfinite(float(frame["upright"])):
            raise ValueError("Non-finite upright measurement")
        _vector(frame["rootPosition"], 3)
        ticks.append(tick)
    return ticks


def _target_position(frame):
    x, y, z = _vector(frame["rootPosition"], 3)
    return np.asarray([z, -x, y])


def compare_reports(reference, target):
    if reference.get("engine") != "MuJoCo" or target.get("identity", {}).get("engine") != "PhysX":
        raise ValueError("Require independent MuJoCo reference and PhysX target reports")
    native = _indexed(reference.get("episodes", []))
    physical = _indexed(target.get("results", []))
    skills = []
    for slot, name in enumerate(POLICIES, 1):
        result = {"slot": slot, "policy": name, "behaviorAccepted": False,
                  "limitations": ["diagnostic_only_not_frozen_task_acceptance",
                                  "experiment_identity_not_verified",
                                  "only_first_target_environment_compared"]}
        skills.append(result)
        if slot not in native or slot not in physical:
            result["status"] = "missing"
            continue
        source, destination = native[slot], physical[slot]
        a = source.get("frames") or []
        episodes = destination.get("episodes") or []
        b = episodes[0] if episodes else []
        if not a or not b:
            result["status"] = "missing"
            continue
        source_ticks = _time_index(a, "MuJoCo")
        target_ticks = _time_index(b, "PhysX")
        if source_ticks[0] != 0 or target_ticks[0] != 0:
            result["status"] = "incomplete"
            result["limitations"].append("missing_initial_state")
            continue
        source_by_tick = dict(zip(source_ticks, a, strict=True))
        missing_control_samples = 0
        for previous, current in zip(target_ticks, target_ticks[1:]):
            if (current - previous) % 4:
                raise ValueError("Target samples do not match the 50 Hz control cadence")
            missing_control_samples += (current - previous) // 4 - 1
        a_origin = _vector(a[0]["rootPosition"], 3)
        b_origin = _target_position(b[0])
        root_errors, input_errors, action_errors, upright_errors = [], [], [], []
        first_input_difference = None
        unmatched = 0
        for tick, sample in zip(target_ticks, b, strict=True):
            time = tick * 0.005
            if tick not in source_by_tick:
                unmatched += 1
                continue
            other = source_by_tick[tick]
            root_errors.append(float(np.linalg.norm(
                (_target_position(sample) - b_origin)
                - (_vector(other["rootPosition"], 3) - a_origin))))
            # Reset contains no previous policy invocation, so t=0 input/action are
            # excluded from policy-parity diagnostics rather than treating stale zeros
            # or previous-episode buffers as a valid inference record.
            if time > 1e-9:
                difference = float(np.max(np.abs(_vector(sample["policyObservation"], 61)
                                                - _vector(other["policyObservation"], 61))))
                input_errors.append(difference)
                action_errors.append(float(np.max(np.abs(_vector(sample["action"], 14)
                                                     - _vector(other["action"], 14)))))
                if first_input_difference is None and difference > 1e-3:
                    first_input_difference = time
            upright_errors.append(abs(float(sample["upright"]) - float(other["upright"])))
        complete = (source.get("completed") and not destination.get("fault") and unmatched == 0
                    and missing_control_samples == 0
                    and source_ticks[-1] >= round(source["requestedSeconds"] / 0.005)
                    and target_ticks[-1] >= round(destination["requestedSeconds"] / 0.005))
        result.update(
            status="recorded_diagnostic" if complete else "incomplete",
            targetFault=destination.get("fault"), referenceError=source.get("error"),
            referenceDuration=a[-1]["timeSeconds"], targetDuration=b[-1]["timeSeconds"],
            matchedSamples=len(root_errors), unmatchedTargetSamples=unmatched,
            missingTargetControlSamples=missing_control_samples,
            maximumRelativeRootDistance=max(root_errors, default=None),
            maximumPolicyInputDifference=max(input_errors, default=None),
            maximumRawActionDifference=max(action_errors, default=None),
            maximumUprightDifference=max(upright_errors, default=None),
            policyInputDifferenceThreshold=0.001,
            firstPolicyInputDifferenceTime=first_input_difference,
            targetFinalUpright=b[-1]["upright"], referenceFinalUpright=a[-1]["upright"],
            targetFinalHeight=b[-1]["rootPosition"][1], referenceFinalHeight=a[-1]["rootPosition"][2],
            modelIdentityVerified=destination.get("modelIdentityVerified", False),
        )
    return {"schemaVersion": 1, "referenceEngine": "MuJoCo", "targetEngine": "PhysX",
            "purpose": "diagnosis_only", "physxAcceptedCount": 0,
            "translationAlignment": "relative to each independent initial root position",
            "timeAlignment": "200 Hz integer physics ticks; float32 timestamp precision checked; no interpolation",
            "skills": skills}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Existing evidence is not overwritten")
    result = compare_reports(json.loads(Path(args.reference).read_text(encoding="utf-8")),
                             json.loads(Path(args.target).read_text(encoding="utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"physxAcceptedCount": 0, "skillsRecorded": len(result["skills"])}))


if __name__ == "__main__":
    main()
