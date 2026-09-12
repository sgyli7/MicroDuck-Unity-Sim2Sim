# Sai_Agent_001 native profile

From the repository root:

```sh
python scripts/setup-sai-agent.py
```

Open `TuanjieProject` in the existing project editor (Tuanjie 2022.3.62t14 / 1.10.2),
choose **SaiAgent001 → Build Demo**, then Play. W/S forward/reverse, A/D yaw, held
Shift crouch, release to stand, R reset. Existing MicroDuck scenes and contracts
remain unchanged.

The setup command fetches Sai commit `6eb18b2abf97ce6de11a18daba93729848736adf`,
checks the ONNX hash, stages licensed meshes and the articulated MJCF under
StreamingAssets, and places the ONNX where Barracuda imports it. The native
controller loads the display MJCF directly, retaining the same physical inertia, arm joints, cargo sliders
and belt equalities; Unity displays native geometry and native body motion.
Display-only meshes have collisions disabled; contact geometry and all physical
parameters match the original model. This preserves the actual blue cargo
shell and SO101 geometry instead of displaying only collision proxies.
There is no Unity Rigidbody simulation layered over the MuJoCo state.

The flat actor observes 82 values and outputs sixteen residual actions. Twelve
leg joints use bounded PD position control; four wheels use bounded velocity
feedback. Arm/cargo joints remain torque-controlled. The policy runs at 50 Hz;
the native 1 ms model is stepped twenty times per policy update.

## Verified and pending

- Setup from the remote pinned GitHub package and ONNX integrity check passed.
- The C# observation/target contract ran under .NET 8.0.425 against 64 Python
  fixtures. Maximum absolute error: 3.88e-7.
- Source API calls checked against the project's pinned official MuJoCo binding.
- **Unity/Tuanjie editor execution, Barracuda ONNX parity and native rendered
  gameplay are not yet verified:** no editor is installed on the current Linux
  ARM machine. The draft PR remains unmerged pending these checks.
- This scene currently provides the flat policy. The Sai package's 20/40 mm
  stair results are MuJoCo/Godot results, not Unity stair acceptance.
- GitHub Actions repeats the portable contract test. A green result checks C#
  numerical formulas, not Unity scene import or engine execution.

Portable contract check:

```sh
dotnet run --project tests/sai_contract_console -- tests/sai_contract_console/python-fixtures.json
```

No hardware build, measured actuator performance, or completed VLA is claimed.
