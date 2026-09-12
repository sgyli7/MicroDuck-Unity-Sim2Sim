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

**SaiAgent001 → Build Stairs** creates separate 20/40 mm ascent/descent scenes
with four real risers and 180 mm treads. The native controller casts the same
24 terrain rays as the Godot profile, excluding robot geometry. It selects the
frozen stair actor, 3.2 s lift reference and slower forward speed when terrain
height varies. The same bounded heading feedback is used for solver transfer.
The sensing is simulated geometry, not camera-based VLA. Reverse/sideways stair
driving and arbitrary terrain have not been accepted.

`SaiNativeWorld.cs` owns model loading, ABI validation, reset, named-joint maps,
height scans, policy timing and motor torque application. The Unity component
supplies Barracuda inference and renders this world's actual geometry. A .NET
acceptance executable uses the exact same world source with ONNX Runtime and
the project's matching MuJoCo 3.12.0 binding/library.

## Verified and pending

- Setup from the remote pinned GitHub package and ONNX integrity check passed.
- The original 64 C# observation/target fixtures pass (maximum error 3.88e-7).
  Additional fixtures cover stair phases, crouch, action limits and heading
  state; the portable CI runs both sets.
- Source API calls checked against the project's pinned official MuJoCo binding.
- **Unity/Tuanjie editor execution, Barracuda ONNX parity and native rendered
  gameplay are not yet verified:** no editor is installed on the current Linux
  ARM machine. The draft PR remains unmerged pending these checks.
- The shared C# native world passed eight physical flat/crouch/reset cases and
  four 20/40 mm stair ascent/descent cases on Linux ARM64. These are real
  MuJoCo 3.12 physics steps with ONNX Runtime inference; they do not establish
  Unity keyboard input, Barracuda numerical parity or rendering.
- Native compilation caught and fixed the generated binding's unsigned model
  counts in loops. The native ABI is checked before any struct dereference.
- GitHub Actions passed portable formulas plus all twelve native physical
  scenarios on Linux and Windows at commit `36fbee4`. See
  [the recorded run](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim/actions/runs/34680326252).
  A green native check does not mean Unity editor execution passed.

Portable contract check:

```sh
dotnet run --project tests/sai_contract_console -- tests/sai_contract_console/python-fixtures.json
dotnet run --project tests/sai_contract_console -- tests/sai_contract_console/python-stair-fixtures.json
```

For the native acceptance test, use a separate Python environment with
`mujoco==3.12.0`, then run:

```sh
python scripts/setup-sai-agent.py
python scripts/setup-sai-native-test.py
dotnet run --project tests/sai_native_console -- .
```

The helper checks the official binding SHA and writes the local native-library
path under ignored `artifacts/`. No editor or global library replacement is
performed. The result is `artifacts/sai-native-test/result.json`.

On Windows and macOS the native acceptance helper now selects the actual
project-bundled MuJoCo binary after checking `upstream.lock.json`; Linux uses
the isolated Python package library. The runtime manifest records that choice.

## Real editor acceptance entry point

On a machine with the supported editor installed, after asset setup:

```sh
python scripts/run-sai-editor-acceptance.py --editor "/absolute/path/to/editor"
```

Alternatively choose **SaiAgent001 → Validate Physics and Barracuda** while
not in Play mode. It loads the real imported ONNX models through the same
`SaiBarracudaActor` as gameplay, then executes the same twelve physical cases
and criteria as the native console. It does not change the current scene.
The result and editor log are under `artifacts/sai-editor-test/`; stale results
cannot satisfy the launcher. This entry point is prepared, **not yet run** on
an actual Unity/Tuanjie editor in this task.

The editor test uses programmatic commands, so it cannot claim keyboard or
rendering acceptance. After it passes, Play each generated scene and check
W/S, A/D, held/released Shift and R with real key events; inspect the four
wheel legs, SO101, cargo and rear display against the approved CAD. The cargo
pickup task has not yet been ported to the Unity component.

For a Windows player, the same helper can build after the real editor physics
test passes (the editor must have Windows build support installed):

```sh
python scripts/run-sai-editor-acceptance.py --editor "/absolute/path/to/editor" --build-windows
```

The executable is `Builds/SaiWindows64/Sai_Agent_001.exe`. The helper requires a
fresh completion marker, verifies the built MuJoCo DLL against its supply-chain
lock and records all build-file hashes. It does not mark player execution or
keyboard/rendering verified. The standalone build entry point is
**SaiAgent001 → Build Windows Player**. These build commands are prepared but
have not run on this Linux ARM workstation.

No hardware build, measured actuator performance, or completed VLA is claimed.
