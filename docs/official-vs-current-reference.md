# Official source versus the preserved current MuJoCo reference

These are two distinct baselines. Neither is PhysX acceptance. The current reference
remains frozen; none of the following findings justify changing it or relaxing the
game's behavior requirements.

## What actually ran

- Official-source adapter: pinned `microduck_rl` revision
  `5946fd9cdbc58956424420153e51975af3b30d77`, original `PolicyInference` class;
  MuJoCo 3.12.0, original nine ONNX files via ONNX Runtime.
- Current reference: preserved native MuJoCo 3.12 + Barracuda Player with passive
  measurement adapter, from `shared-integration-db6f4df79aa3/reference.json`.
- Same development seed 92317, root poses and requested horizons. Official automatic
  skill exits are intentionally retained and differ from current manual slot holds.
- Windows adapter removes only `termios`/`tty` imports and does not run the viewer.
  Timers use simulation time (the interactive source uses wall clock); inference is
  active immediately rather than waiting for the keyboard/standby UI. Thus this is
  **an official-controller headless adapter, not an untouched interactive CLI run**.
  Source timers advance before inference; fixed-step kick/roll inference therefore
  lasts approximately 2.98/1.98 seconds, not a strict full 3/2 seconds after triggering.
- Script bytes are checked against the pinned Git blob. Robot source tree is checked
  clean against its pinned revision. Reports hash configured compiled model bytes,
  source policy files and BAM parameter JSON. Native and target Players remain separate.

## Material configuration differences

| Item | Official inference script defaults | Frozen current reference |
| --- | --- | --- |
| Action scale | 1.0 | walk 1.1; roller/crouch 0.8; others 1.0 |
| Actuator torque cap | 1.75 A × bundled kt = 0.6405236 N m | XML actuator cap 0.96 N m |
| Target smoothing | None in `apply_action` | Head alpha 0.5, other joints alpha 0.7 |
| Ground pick phase | Period 4 s; exits at phase 0.7, returns to base policy | Period 4 s; clamps at 0.8 while slot remains active |
| Roller crouch phase | Uses the source ground-pick API/default 4 s; auto-exits at 0.7 | Period 5 s, clamp 0.7; explicit live slot switching |
| Kick/roll lifecycle | Automatic return after 3 s / 2 s | Skill slot remains selected unless the experiment switches it |

Source: [pinned official inference implementation](https://github.com/pollen-robotics/microduck_rl/blob/5946fd9cdbc58956424420153e51975af3b30d77/scripts/infer_policy.py).
The headless adapter reads the same installed BAM `params/xl330/m6.json` scalar used
by `load_model`; it does not substitute a BAM dynamics model for position actuators.

## Measured nine-skill difference, not a pass table

Displacements below are in metres on the common Unity world axes: forward +Z,
right +X (not heading-normalized task scores). Both native trace coordinate bases
were converted before comparison. This development seed is not a final held-out seed.
Minimum upright is diagnostic: a full roll is supposed to invert temporarily.

| Slot / requested skill | Official Δforward | Official Δright | Official min upright | Current Δforward | Current Δright | Current min upright |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 walk | 0.0105 | -0.0042 | 0.9999 | 0.4919 | 0.0436 | 0.9975 |
| 2 stand | 0.0033 | -0.0000 | 0.9998 | 0.0028 | 0.0001 | 0.9997 |
| 3 sit/stand | -0.0502 | -0.0033 | 0.9928 | -0.0202 | -0.0101 | 0.9932 |
| 4 ground pick | 0.0067 | 0.0016 | 0.8264 | 0.0079 | 0.0016 | 0.8232 |
| 5 left kick | -0.0003 | -0.0081 | 0.9960 | 0.0007 | -0.0144 | 0.9958 |
| 6 right kick | -0.0321 | 0.0176 | 0.9860 | -0.0395 | 0.0255 | 0.9873 |
| 7 roller | 1.6715 | -0.7781 | 0.9763 | 1.0350 | -0.1763 | 0.9846 |
| 8 moving roller crouch | 0.9771 | -0.3585 | 0.9766 | 0.6303 | -0.0677 | 0.9858 |
| 9 roll/recover | 0.5862 | -0.0471 | -1.0000 | 0.5806 | 0.0047 | -0.9979 |

All nine source-adapter recording horizons completed; this does **not** mean all nine
tasks succeeded. In particular default-source walking barely progressed on this plain
XML setup. Ball direction/contact and complete skill metrics cannot be inferred from
body displacement alone. Current reference and PhysX must continue to be assessed
against the frozen task-quality contract, not these summary numbers.

Evidence: `artifacts/sim2sim/official-source-native-v2-20260912.json` and the current
reference report above. Tests: 12 passed, adapter coverage 88%. Reproduction with
the MuJoCo 3.12 light venv (not the training venv's MuJoCo 3.10):

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m agenticrobot_bridge.upstream_inference --output <new.json> --seed 92317
```
