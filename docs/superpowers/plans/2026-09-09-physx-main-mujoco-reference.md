# PhysX main and frozen MuJoCo reference

Approved design: preserve the current runnable MuJoCo implementation as a separate
reference project and Player BEFORE removing its dependencies from the game project.
The game, its training observations and final Player must use real PhysX exclusively.
The upstream official MuJoCo baseline is distinct from our current modified reference.

## Completion contract

- Preserve source revision, editor/packages, native binaries, all nine policies,
  generated assets, scenes, physics/controller parameters, launch instructions and hashes.
- Run the current reference and its isolated copy; verify no behavior change before
  changing the main project's physics dependencies. Keep failures in the evidence.
- Make the default Tuanjie project and Windows build PhysX-only. Retain the terrain,
  lighting, sky, camera, input and real interactions with game objects.
- Give both independent Players explicit engine identities and a reproducible A/B entry.
  Never feed reference poses into the target simulator. Align by simulation time.
- Share experiment definitions (policy hash, variant, terrain, seed, initial state,
  command timeline, duration) and engine-labelled traces, not dynamics implementations.
- Align 61 observations, 14 actions, control 50 Hz and physics 200 Hz, coordinate
  handedness, normalization, action history/filtering and joint/actuator semantics.
- Test unmodified policies first. Adapt failed skills using actual PhysX sampling and
  existing RSL-RL/PyTorch PPO; keep original and *_PhysX policies and results separate.
- Test stand, walk, sit/stand, ground pick, left/right kick, roller, roller crouch and
  roulade completely. Include final independent 100 episodes per policy and 60-second
  continuous tests, phase-aware acceptance, switching, terrains and real game collisions.
- Numerical inference parity, software tests and behavior acceptance are separate gates.
  Missing/skipped/timed-out tests never pass. No hidden supports, pose replay, frozen
  roots or reset-on-failure. Only explicit reset may teleport the robot.
- Deliver both runnable builds, A/B evidence, honest nine-policy comparisons, dependency
  locks, commands, checkpoints and reviewed commits/PR. Five hours is an effort budget,
  not permission to claim convergence or pass failed behaviors.

## Execution record

- [x] Inspect repository and approved architecture; main was clean at 3a97ebe.
- [ ] Open draft PR before implementation.
- [ ] Preserve and independently verify the current runnable reference.
- [ ] Establish engine identity and default PhysX build (test first).
- [ ] Implement shared A/B experiment/trace pipeline (test first).
- [ ] Diagnose and correct PhysX transfer failures (test first).
- [ ] Connect real PhysX training and ONNX re-export (test first).
- [ ] Independent nine-model acceptance, recordings and final review.

Work branch: codex/physx-main-mujoco-reference. Each completed slice is committed;
generated caches, credentials and builds remain excluded. Main is merged only after
appropriate review and verification. Existing misleading delivery reports remain
historical evidence, not current PhysX acceptance.
