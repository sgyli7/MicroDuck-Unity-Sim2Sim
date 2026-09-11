"""Same-input numerical checks on actual Player traces, distinct from behavior quality."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .physx_actor import actor_from_onnx
from .physx_evaluate import POLICIES, bound_model_identity
from .shared_experiment import write_new


def compare_trace(report, policy_root, samples_per_slot=64):
    if samples_per_slot < 1:
        raise ValueError("Need positive sample count")
    native = report.get("engine") == "MuJoCo"
    identity = report if native else report.get("identity", {})
    if identity.get("engine") not in ("MuJoCo", "PhysX") or not identity.get("buildGuid"):
        raise ValueError("Actual Player identity is required")
    models = {item["slot"]: item for item in identity.get("models", [])}
    if len(models) != 9 or len(identity["models"]) != 9:
        raise ValueError("Nine unique model identities required")
    grouped = {slot: [] for slot in range(1, 10)}
    for episode in report["episodes" if native else "results"]:
        if episode.get("inference") != "Barracuda":
            raise ValueError("Recorded actions must explicitly come from Barracuda, not an external actor")
        frames = episode.get("frames", []) if native else (episode.get("episodes") or [[]])[0]
        for frame in frames:
            tick = frame["physicsSteps"]
            if tick == 0 or tick % 4:
                continue
            if frame.get("engine") != identity["engine"]:
                raise ValueError("Trace mixes physics engine identities")
            expected_backend = "MuJoCo 3.12 + Barracuda 3.0.1 CPU" if native else "Barracuda 3.0.1 CPU"
            if frame.get("inferenceBackend") != expected_backend:
                raise ValueError("Actual inference backend is not the recorded Barracuda runtime")
            slot = frame.get("inferenceSlot")
            if slot not in grouped:
                raise ValueError("Missing actual inference slot; cannot guess from active skill at a switch")
            grouped[slot].append(frame)
    results = []
    torch.set_num_threads(4)
    for slot, name in enumerate(POLICIES, 1):
        path = Path(policy_root) / (name + ".onnx")
        payload = path.read_bytes()
        source_hash = hashlib.sha256(payload).hexdigest()
        model, verified = bound_model_identity(identity, slot, source_hash)
        if model is None or not verified:
            raise ValueError("Recorded model source differs from evaluated original weights")
        available = grouped[slot]
        if not available:
            results.append({"slot": slot, "samples": 0, "passed": False, "reason": "missing_inference_samples"})
            continue
        selected = [available[index] for index in np.linspace(0, len(available) - 1,
                    min(samples_per_slot, len(available)), dtype=int)]
        obs = np.asarray([frame["policyObservation"] for frame in selected], dtype=np.float32)
        recorded = np.asarray([frame["action"] for frame in selected], dtype=np.float32)
        if obs.shape != (len(selected), 61) or recorded.shape != (len(selected), 14):
            raise ValueError("Malformed inference dimensions")
        if not np.isfinite(obs).all() or not np.isfinite(recorded).all():
            raise ValueError("Non-finite recorded inference")
        runtime = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
        # Original exports may use a static batch of one.
        original = np.concatenate([runtime.run(None, {runtime.get_inputs()[0].name: row[None]})[0] for row in obs])
        actor = actor_from_onnx(payload)
        with torch.no_grad():
            restored = actor(torch.from_numpy(obs)).numpy()
        # Existing repository actor/export tolerance, unchanged for this new corpus.
        tolerance = 1e-4 + 2e-5 * np.abs(original)
        barracuda_error = np.abs(recorded - original)
        actor_error = np.abs(restored - original)
        finite = np.isfinite(original).all() and np.isfinite(restored).all()
        results.append({"slot": slot, "model": path.name, "sourceSha256": source_hash,
                        "graphSha256": model["graphSha256"], "samples": len(selected),
                        "barracudaMaximumAbsoluteError": float(barracuda_error.max()),
                        "restoredTrainingActorMaximumAbsoluteError": float(actor_error.max()),
                        "passed": bool(finite and (barracuda_error <= tolerance).all() and (actor_error <= tolerance).all())})
    return {"purpose": "same-input numerical parity only", "samplePhysicsEngine": identity["engine"],
            "buildGuid": identity["buildGuid"], "absoluteTolerance": 1e-4, "relativeTolerance": 2e-5,
            "actor": "original weights restored into the training architecture; not an adapted checkpoint",
            "numericalParityPassed": all(item["passed"] for item in results),
            "behaviorAccepted": False, "models": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--policy-root", default=".cache/upstream/microduck/policies")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trace_bytes = Path(args.trace).read_bytes()
    report = json.loads(trace_bytes)
    result = compare_trace(report, args.policy_root)
    result["traceSha256"] = hashlib.sha256(trace_bytes).hexdigest()
    write_new(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["numericalParityPassed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
