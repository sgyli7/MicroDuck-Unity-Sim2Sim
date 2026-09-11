"""Shared inputs and independent-player A/B diagnostics, never an acceptance shortcut."""

import argparse
import hashlib
import json
import random
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

from .physx_client import PhysXClient
from .physx_evaluate import POLICIES, bound_model_identity
from .sim2sim_comparison import compare_reports


def make_batch(policy_root, seed):
    rng = random.Random(seed)
    models = [{"slot": i, "sha256": hashlib.sha256((Path(policy_root) / (name + ".onnx")).read_bytes()).hexdigest()}
              for i, name in enumerate(POLICIES, 1)]
    cases = []
    for slot, seconds in enumerate([6, 4, 8, 4, 3, 3, 6, 8, 5], 1):
        initial_slot = 7 if slot == 8 else 2 if slot == 9 else slot
        events = []

        def event(tick, kind, selected=0, values=None):
            events.append({"physicsStep": tick, "kind": kind, "slot": selected, "values": values or []})

        if slot == 1:
            event(0, "twist", values=[0.2, 0, 0])
        if slot in (7, 8):
            event(0, "twist", values=[0.3, 0, 0])
        if slot == 3:
            event(200, "trigger")
            event(900, "trigger")
        if slot == 4:
            event(0, "trigger")
        if slot == 8:
            event(600, "switch", 8)
            event(600, "trigger")
            event(1300, "switch", 7)
            event(1300, "twist", values=[0.3, 0, 0])
        if slot == 9:
            event(200, "switch", 9)
            event(200, "trigger")
            event(600, "switch", 2)
        cases.append({"id": f"seed-{seed}-slot-{slot}", "seed": seed, "slot": slot,
                      "initialSlot": initial_slot, "terrain": "flat", "physicsSteps": seconds * 200,
                      "initialRootPosition": [rng.uniform(-0.01, 0.01),
                                              0.1385 if slot in (7, 8) else 0.125,
                                              rng.uniform(-0.01, 0.01)],
                      "initialYawDegrees": rng.uniform(-3, 3), "events": events})
    return {"schemaVersion": 1, "coordinateBasis": "Unity-Xright-Yup-Zforward",
            "seedUsage": "deterministic initial-position/yaw generation, not simulated noise",
            "evaluationRole": "development diagnostic; not final held-out acceptance",
            "models": models, "cases": cases}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def copy_experiment(source, destination):
    payload = Path(source).read_bytes()
    encoded = payload.decode("utf-8")
    json.loads(encoded)
    with Path(destination).open("xb") as stream:
        stream.write(payload)
    return encoded


def validated_microsteps(interval, previous_tick):
    def finite_vector(record, name, length):
        array = np.asarray(record.get(name), dtype=float)
        if array.shape != (length,) or not np.isfinite(array).all():
            raise ValueError('Dense physics trace contains invalid physical state: ' + name)
        return array

    trace = interval.get('physicsTrace')
    if not isinstance(trace, list) or len(trace) != 4 or interval['physicsSteps'] != previous_tick + 4:
        raise ValueError('Dense physics trace is missing actual substeps')
    for offset, frame in enumerate(trace, 1):
        tick = previous_tick + offset
        if (frame.get('engine') != 'PhysX' or frame.get('physicsSteps') != tick
                or frame.get('inferenceSlot') != interval['inferenceSlot']
                or not np.isfinite(frame['timeSeconds'])
                or abs(frame['timeSeconds'] - tick * .005) > max(1e-8, tick * .005 * 2e-7)):
            raise ValueError('Dense physics trace tick/engine/executed policy mismatch')
        for name, length in [('rootPosition', 3), ('rootVelocity', 3), ('rootAngularVelocity', 3),
                             ('jointPosition', 14), ('jointVelocity', 14), ('mouthTipPosition', 3),
                             ('ballPosition', 3), ('ballVelocity', 3)]:
            finite_vector(frame, name, length)
        rotation = finite_vector(frame, 'rootRotation', 4)
        if abs(np.linalg.norm(rotation) - 1) > 1e-4 or not np.isfinite(frame.get('upright', np.nan)):
            raise ValueError('Dense physics trace contains invalid orientation')
        wheels = frame.get('passiveWheelVelocity')
        wheel_count = 4 if frame['inferenceSlot'] in (7, 8) else 0
        if not isinstance(wheels, list) or len(wheels) != wheel_count or not np.isfinite(wheels).all():
            raise ValueError('Dense physics trace contains invalid passive wheel measurement')
        if type(frame.get('ballActive')) is not bool:
            raise ValueError('Dense physics trace missing ball identity')
        if not isinstance(frame.get('contacts'), list):
            raise ValueError('Dense physics trace missing contact callback journal')
        for contact in frame['contacts']:
            for name in ('point', 'normal', 'pairImpulse'):
                finite_vector(contact, name, 3)
            if (not np.isfinite(contact.get('separation', np.nan))
                    or contact.get('eventKind') not in ('enter', 'stay')
                    or any(not isinstance(contact.get(key), str) or not contact[key]
                           for key in ('observer', 'collider', 'otherCollider'))):
                raise ValueError('Dense physics trace contains invalid contact callback')
    for field in ('rootPosition', 'rootRotation', 'rootVelocity', 'rootAngularVelocity',
                  'jointPosition', 'jointVelocity', 'passiveWheelVelocity'):
        endpoint = finite_vector(interval, field, len(trace[-1][field]))
        if not np.allclose(trace[-1][field], endpoint, rtol=0, atol=1e-6):
            raise ValueError('Dense physics endpoint differs from actual control boundary: ' + field)
    return trace


def sample_target(client, encoded):
    if client.identity.get('physicsTraceSchema') != 'physx-microstep-v1-passive-contact-callbacks':
        raise ValueError('Target Player does not support verified dense physics measurements')
    batch = json.loads(encoded)
    input_hash = hashlib.sha256(encoded.encode()).hexdigest()
    results = []
    for index, case in enumerate(batch["cases"]):
        current = client.request("experiment", experimentJson=encoded, caseIndex=index)["results"]
        episodes = [[frame] for frame in current]
        physics_episodes = [[] for _ in current]
        fault = None
        for _ in range(case["physicsSteps"] // 4):
            try:
                previous = current
                current = client.step(record_physics_trace=True)
            except RuntimeError as error:
                fault = {"reason": str(error), "physicsSteps": current[0]["physicsSteps"]}
                break
            for frames, physical, before, frame in zip(episodes, physics_episodes, previous, current, strict=True):
                if frame["experimentSha256"] != input_hash or frame["experimentId"] != case["id"]:
                    raise ValueError("Target changed experiment identity during sampling")
                physical.extend(validated_microsteps(frame, before['physicsSteps']))
                # Store each microstep only once, separate from the 50 Hz inference log.
                frame.pop('physicsTrace')
                frames.append(frame)
        source = next(item["sha256"] for item in batch["models"] if item["slot"] == case["slot"])
        identity, verified = bound_model_identity(client.identity, case["slot"], source)
        results.append({"slot": case["slot"], "experimentId": case["id"], "experimentSha256": input_hash,
                        "seed": case["seed"], "requestedSeconds": case["physicsSteps"] * 0.005,
                        "modelIdentity": identity, "modelIdentityVerified": verified,
                        "modelSha256": None, "inference": "Barracuda", "behaviorAccepted": False,
                        "fault": fault, "episodes": episodes, 'physicsEpisodes': physics_episodes,
                        'contactsSemantics': 'per-step enter/stay callbacks; empty does not exclude sleeping contact; '
                                             'pair impulse repeated per point and body observer'})
    return {"identity": client.identity, "experimentSha256": input_hash, "results": results,
            "behaviorAccepted": False, "independentSeedsPerSkill": len({case["seed"] for case in batch["cases"]})}


def compare_batch(encoded, reference, target):
    batch = json.loads(encoded)
    input_hash = hashlib.sha256(encoded.encode()).hexdigest()

    def indexed(rows):
        result = {}
        for row in rows:
            key = row.get("experimentId")
            if not key or key in result:
                raise ValueError("Missing or duplicate experiment case identity")
            if row.get("experimentSha256") != input_hash:
                raise ValueError("Experiment input hash mismatch")
            result[key] = row
        return result

    native = indexed(reference["episodes"])
    physical = indexed(target["results"])
    comparisons = []
    for case in batch["cases"]:
        key = case["id"]
        if key not in native or key not in physical:
            comparisons.append({"experimentId": key, "status": "missing", "behaviorAccepted": False})
            continue
        a, b = native[key], physical[key]
        if a["slot"] != case["slot"] or b["slot"] != case["slot"]:
            raise ValueError("Experiment slot mismatch")
        expected_seconds = case["physicsSteps"] * 0.005
        if abs(a["requestedSeconds"] - expected_seconds) > 1e-5 or abs(b["requestedSeconds"] - expected_seconds) > 1e-5:
            raise ValueError("Experiment horizon mismatch")
        report = compare_reports({**reference, "episodes": [a]}, {**target, "results": [b]})
        item = report["skills"][case["slot"] - 1]
        item["experimentId"] = key
        item["experimentSha256"] = input_hash
        item["limitations"].remove("experiment_identity_not_verified")
        item["limitations"].append("terrain_material_and_solver_equivalence_not_asserted")
        # Shared declared conditions are not enough: inspect both actual reset poses.
        if a.get("frames") and b.get("episodes") and b["episodes"][0]:
            native_frame, target_frame = a["frames"][0], b["episodes"][0][0]
            expected = np.asarray(case["initialRootPosition"])
            native_initial = np.asarray(a["frames"][0]["rootPosition"])
            native_unity = native_initial[[1, 2, 0]] * [-1, 1, 1]
            actual_target = np.asarray(b["episodes"][0][0]["rootPosition"])
            item["maximumInitialPositionError"] = float(max(np.max(np.abs(native_unity - expected)),
                                                           np.max(np.abs(actual_target - expected))))
            if item["maximumInitialPositionError"] > 1e-5:
                item["status"] = "initial_condition_mismatch"
            required = ("rootVelocity", "rootAngularVelocity")
            if ("rootQuaternionWxyz" not in native_frame or "rootRotation" not in target_frame
                    or any(key not in frame for frame in (native_frame, target_frame) for key in required)):
                item["status"] = "initial_condition_unverified"
            else:
                def vector(value, count):
                    array = np.asarray(value, dtype=float)
                    if array.shape != (count,) or not np.isfinite(array).all():
                        raise ValueError("Invalid actual initial state measurement")
                    return array

                w, x, y, z = vector(native_frame["rootQuaternionWxyz"], 4)
                native_q = np.asarray([y, -z, -x, w])
                target_q = vector(target_frame["rootRotation"], 4)
                half = np.deg2rad(case["initialYawDegrees"]) / 2
                expected_q = np.asarray([0, np.sin(half), 0, np.cos(half)])
                angles = []
                for q in (native_q, target_q):
                    norm = np.linalg.norm(q)
                    if abs(norm - 1) > 1e-4:
                        raise ValueError("Actual initial quaternion is not normalized")
                    angles.append(2 * np.arccos(np.clip(abs(np.dot(q / norm, expected_q)), 0, 1)))
                item["maximumInitialRotationErrorRadians"] = float(max(angles))
                item["maximumInitialSpeed"] = float(max(np.linalg.norm(vector(frame[key], 3))
                                                       for frame in (native_frame, target_frame) for key in required))
                if max(angles) > 1e-5 or item["maximumInitialSpeed"] > 1e-6:
                    item["status"] = "initial_condition_mismatch"
        comparisons.append(item)
    return {"purpose": "independent_physics_diagnostics", "experimentSha256": input_hash,
            "referenceEngine": "MuJoCo", "targetEngine": "PhysX", "physxAcceptedCount": 0,
            "cases": comparisons}


def run_players(spec_path, output, reference_player, target_player, port):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # All artifacts stay in this fresh run directory. No shell or existing process is controlled.
    spec_copy = output / "experiment.json"
    encoded = copy_experiment(spec_path, spec_copy)
    reference_output = output / "reference.json"
    owned = []
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        target = subprocess.Popen([str(Path(target_player).resolve()), "-batchmode", "-nographics",
                                   "-physxPort", str(port), "-physxEnvs", "1", "-logFile",
                                   str((output / "target.log").resolve())],
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        owned.append(target)
        deadline = time.monotonic() + 45
        while True:
            try:
                client = PhysXClient(port=port, timeout=10)
                break
            except OSError:
                if target.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Independent PhysX Player did not become ready") from None
                time.sleep(0.2)
        reference = subprocess.Popen([str(Path(reference_player).resolve()), "-batchmode", "-nographics",
                                      "-referenceExperiment", str(spec_copy.resolve()), "-referenceOutput",
                                      str(reference_output.resolve()), "-logFile", str((output / "reference.log").resolve())],
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        owned.append(reference)
        with client:
            target_report = sample_target(client, encoded)
        write_new(output / "target.json", target_report)
        reference_code = reference.wait(timeout=180)
        if not reference_output.exists():
            raise RuntimeError(f"Reference produced no report; exit code {reference_code}")
        comparison = compare_batch(encoded, json.loads(reference_output.read_text(encoding="utf-8")), target_report)
        comparison["referenceExitCode"] = reference_code
        write_new(output / "comparison.json", comparison)
        print(json.dumps({"output": str(output.resolve()), "physxAcceptedCount": 0,
                          "cases": [(item["experimentId"], item["status"]) for item in comparison["cases"]]}), flush=True)
    finally:
        for process in reversed(owned):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--policy-root", default=".cache/upstream/microduck/policies")
    generate.add_argument("--seed", type=int, default=71453)
    generate.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--reference-player", required=True)
    run.add_argument("--target-player", default="Builds/Windows64/AgenticRobotGame.exe")
    run.add_argument("--port", type=int, default=62103)
    args = parser.parse_args()
    if args.command == "generate":
        write_new(args.output, make_batch(args.policy_root, args.seed))
    else:
        run_players(args.spec, args.output, args.reference_player, args.target_player, args.port)


if __name__ == "__main__":
    main()
