"""Record actual target rollouts. Metrics are evidence, never an automatic skill pass."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .physx_client import PhysXClient


POLICIES = ["alpha_walking", "alpha_stand", "alpha_sitstand", "alpha_ground_pick",
            "ball_kick_left", "ball_kick_right", "roller", "roller_crouch", "roulade"]
DURATIONS = [6, 4, 8, 4, 3, 3, 6, 3, 3]


def bound_model_identity(identity, slot, expected_source_hash):
    models = [model for model in identity.get("models", []) if model.get("slot") == slot]
    if not models:
        return None, False
    if len(models) != 1:
        raise ValueError("Player reports duplicate model identities for this slot")
    model = models[0]
    if model.get("sourceSha256") != expected_source_hash:
        raise ValueError("Actual Player policy source differs from the requested ONNX")
    verified = bool(identity.get("buildGuid")) and model.get("graphVerified") is True
    for field in ("sourceSha256", "convertedOnnxSha256", "graphSha256"):
        value = model.get(field, "")
        verified = verified and isinstance(value, str) and len(value) == 64
        verified = verified and all(char in "0123456789abcdef" for char in value)
    return model, verified


def rollout(client, slot, policy, seconds, internal=False):
    policy = Path(policy)
    source_bytes = policy.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    model_identity, model_verified = bound_model_identity(client.identity, slot, source_hash) if internal else (None, True)
    # Hash and infer the same immutable bytes, even if training exports a newer file meanwhile.
    runtime = None if internal else ort.InferenceSession(source_bytes,
                                                       providers=["CPUExecutionProvider"])
    current = client.reset(slot, external=not internal)
    frames = [[frame] for frame in current]
    fault = None
    for _ in range(round(seconds / 0.02)):
        actions = None
        if runtime is not None:
            actions = [runtime.run(None, {runtime.get_inputs()[0].name:
                       np.asarray(frame["observation"], np.float32)[None]})[0][0].tolist()
                       for frame in current]
            if not np.isfinite(actions).all() or np.abs(actions).max() > 5:
                fault = {"reason": "actor_outside_execution_envelope", "actions": actions,
                         "timeSeconds": current[0]["timeSeconds"]}
                break
        try:
            current = client.step(actions)
        except RuntimeError as error:
            fault = {"reason": str(error), "timeSeconds": current[0]["timeSeconds"]}
            break
        for destination, frame in zip(frames, current, strict=True):
            destination.append(frame)
        if any(not frame["healthy"] for frame in current):
            fault = {"reason": "controller_fault", "timeSeconds": current[0]["timeSeconds"]}
            break
    metrics = []
    for episode in frames:
        xyz = np.asarray([f["rootPosition"] for f in episode])
        upright = np.asarray([f["upright"] for f in episode])
        below = np.flatnonzero((xyz[:, 1] < 0.065) | (upright < 0.45))
        metrics.append({
            "durationSeconds": episode[-1]["timeSeconds"],
            "minimumUpright": float(upright.min()), "finalUpright": float(upright[-1]),
            "minimumHeight": float(xyz[:, 1].min()), "finalHeight": float(xyz[-1, 1]),
            "heightExcursion": float(np.ptp(xyz[:, 1])),
            "forwardDisplacement": float(xyz[-1, 2] - xyz[0, 2]),
            "sideDisplacement": float(xyz[-1, 0] - xyz[0, 0]),
            "firstLowOrTippedSeconds": None if not len(below) else episode[int(below[0])]["timeSeconds"],
        })
    return {"engine": "PhysX", "role": "target", "slot": slot,
            "model": policy.name, "modelSha256": None if internal else source_hash,
            "expectedSourceSha256": source_hash, "modelIdentityVerified": model_verified,
            "modelIdentity": model_identity,
            "inference": "Barracuda" if internal else "ONNX Runtime external actor",
            "adapted": policy.name.endswith("_PhysX.onnx"), "behaviorAccepted": False,
            "coordinateBasis": "Unity left-handed X-right Y-up Z-forward",
            "independentRandomSeeds": False, "requestedSeconds": seconds,
            "fault": fault, "metrics": metrics, "episodes": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=62101)
    parser.add_argument("--slot", type=int, choices=range(1, 10))
    parser.add_argument("--policy")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        parser.error("Use a new evidence filename; existing experiments are not overwritten")
    if args.policy and (args.slot is None or args.internal):
        parser.error("An external model requires --slot and cannot use --internal")
    started = time.monotonic()
    with PhysXClient(port=args.port) as client:
        results = []
        for slot in [args.slot] if args.slot else range(1, 10):
            path = args.policy or f".cache/upstream/microduck/policies/{POLICIES[slot-1]}.onnx"
            result = rollout(client, slot, path, args.seconds or DURATIONS[slot-1], args.internal)
            results.append(result)
            print(json.dumps({"model": result["model"], "metrics": result["metrics"][0],
                              "fault": result["fault"]}), flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"identity": client.identity, "results": results,
                                      "elapsedSeconds": time.monotonic() - started},
                                     allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
