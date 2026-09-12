# Sai_Agent_001 integration (in progress)

Add the user's SO101 four-wheel-leg cargo robot as an explicit robot profile while
preserving MicroDuck's existing profile and policy semantics. The source package
is being prepared at https://github.com/sgyli7/Sai_Agent_001; no published tag is
claimed by this planning commit.

## Intended behavior

- W/S command forward/reverse body velocity; A/D command yaw; held Shift commands
  crouch and release returns to normal height.
- A versioned package supplies the articulated MJCF, legal display assets, frame
  and actuator manifest, ONNX policy, observation contract and evaluation results.
- The four active wheels use bounded velocity feedback; twelve leg joints use
  bounded position feedback. The SO101 and cargo mechanism remain articulated.
- Continuous stair ascent and descent must be tested separately. A trained
  MuJoCo result is not automatically a Unity acceptance result.

## Implementation/verification checklist

- [ ] Pin package version and hashes; add an import/setup entry point.
- [ ] Profile-aware command, observation, action and native MuJoCo adapters.
- [ ] Preserve existing MicroDuck defaults and regressions.
- [ ] Verify normalization, joint order, axes, torque limits and timestep semantics.
- [ ] Exercise WASD/Shift and stair flights in the game runtime.
- [ ] Document exact tested editor/runtime and clean-install startup.

This branch is being tracked through a draft PR before implementation, as required
by this repository's AGENTS.md. Checkboxes represent outstanding work, not claims
that the integration is already usable.
