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
- [x] Open draft PR before implementation: PR #2.
- [x] Preserve and independently verify the current runnable reference (4,662 hashes;
  nine-policy/compound numerical reports identical except timestamp, both scene suites 5/5).
- [x] Establish engine identity and default PhysX build (red/green default-scene and
  collision tests, 68/68 EditMode, real Player sampler integration passed).
- [ ] Implement shared A/B experiment/trace pipeline (test first).
- [ ] Diagnose and correct PhysX transfer failures (test first).
- [ ] Connect real PhysX training and ONNX re-export (test first).
- [ ] Independent nine-model acceptance, recordings and final review.

Work branch: codex/physx-main-mujoco-reference. Each completed slice is committed;
generated caches, credentials and builds remain excluded. Main is merged only after
appropriate review and verification. Existing misleading delivery reports remain
historical evidence, not current PhysX acceptance.

## Verified slices / outstanding behavior (2026-09-09)

Frozen reference: `../AgenticRobotGame-MuJoCoReference-20260909`, original revision
`3a97ebec04455d9e04f725f0b6e5db7042fe94f8`. Its complete Player and required untracked
inputs are included in the hash manifest. Main-project native inputs were moved to
`artifacts/sim2sim/retired-main-inputs` only after reference equivalence was verified.

Main keeps seven terrain modules, real PhysX colliders, environment/camera, Barracuda
and Codely. Same-rig kick switching and selected-terrain reset regressions are covered.
PhysX-only sampler has independent PhysicsScenes, transactional action validation,
partial resets and the same game controller. Network access is loopback-only/opt-in.

**Behavior is NOT accepted.** Initial full PlayMode: 10/13 passed, 3 failed, no skipped
tests. Home-hold, servo step response and the strict policy aggregate fail. The rotor
inertia approximation improved the isolated servo test but regressed walking, and is
disabled by default. Source action magnitude guards remain intact. Historical native
suite thresholds are preservation checks, not the final nine-skill quality standard.

Latest software evidence: `artifacts/sim2sim/physx-split-editmode-final.xml` (68/68),
`batch-green.xml` (4/4 interactions/sampler isolation), and
`tests/test_physx_player_integration.py` against the actual 2-environment Player.
Original ONNX → restored actor → export numerical tests: all nine pass (absolute
tolerance 1e-4 plus relative 2e-5 on seeded inputs). This is numerical parity, not
cross-engine behavior equivalence.

## Sampling correction and renewed evidence (2026-09-12)

- Frozen reference rehashed: no changed, missing or added files. A separate measured
  derivative (`../AgenticRobotGame-MuJoCoMeasured-20260909`) contains additive observers,
  not modifications to the frozen controller/physics. Its passive probe noninterference
  test compares all native qpos before/after (stand, 0.2 s); 2/2 adapter tests pass.
  `current-mujoco-nine-20260912.json` records all nine requested horizons. Completing
  those horizons is NOT a new skill acceptance result.
- Found a concrete reset defect: reactivation with `matchAnchors` changed parent
  anchor rotations by previous joint angles, changing the robot between episodes.
  Freeze authoring anchors before motion, reset reduced coordinates and publish their
  explicit-reset geometry without advancing simulation. Identical reset/action and
  game-vs-replica tests now pass. Persist/reapply solver settings (12/4); they previously
  reverted to 6/1 when loading saved prefabs. The sampling contract suite is 11/11.
- Next-state observation no longer overwrites the actual policy-input record. Scheduled
  sit/stand commands are published before both internal and external actors infer.
  Real Player protocol/reset tests: 4/4, including internal Barracuda inference.
- Real PhysX + CUDA RSL-RL PPO update, checkpoint reload/resume and bounded ONNX export
  are integration-tested: Python 24/24; actor/client/training coverage 89%. Current
  reward support is ONLY stand/walk. This is pipeline verification, not policy quality.
- Renewed unmodified nine-policy rollouts exist for ORT and Barracuda. Stand is stable
  in the short trial; walk still has excessive lateral drift, sit/stand fails, roller
  crouch still needs a proper moving initial condition. Full PlayMode: 24 total,
  21 pass, 3 fail (servo response, home hold, strict behavior aggregate). No skips.
- Prior exploratory PPO outputs generated before the reset correction are NOT accepted
  adaptation products. Final independent seeds/100 episodes, 60-second runs, common
  experiment contract, complete task metrics, A/B video and all-skill adaptation remain.

## Shared experiment and adaptation progress (2026-09-12, later update)

- Shared v1 flat-ground experiments now drive independent measured MuJoCo and PhysX
  Players with identical hashed inputs, verified initial root pose/velocity and integer
  simulation ticks. Nine skills include the moving roller-crouch and stand/roll/recovery
  timelines. These are diagnostic recordings, not accepted behaviors. Other terrain
  types remain unsupported by this shared sampler, although the game retains them.
- Actual Barracuda traces from both engines versus original ORT and restored training
  actor pass same-input numerical checks for all nine (64 samples per skill). Model
  provenance binds original ONNX, converted ONNX and the actual bound Barracuda graph.
- Official pinned upstream controller now runs separately; its default scales, motor
  cap and automatic lifecycles differ from the frozen current reference. See
  `docs/official-vs-current-reference.md`; neither reference replaces target sampling.
- PPO/checkpoint/resume/export uses real PhysX. Four motion experiments remain rejected;
  measured-motion reward v5 is running separately, not installed in the game. Supported
  training objectives are still only stand/walk. See `docs/physx-training-ledger.md`.
- Opt-in PhysX 200 Hz measurements record each real substep, joint/body states, mouth
  tip, passive wheels, ball and enter/stay contact callbacks. They do not apply states
  or forces. Empty contact callbacks do NOT prove no sleeping contact; pair impulse is
  repeated per point/body-side callback and must not be summed blindly. The 50 Hz
  inference record remains separate, preserving actual executed-model identity.
  The 12-test game/sampler parity suite passes, including tracing-on/off dynamics
  equality at 1e-6. Initial actual-player evidence is
  `shared-integration-971bbd4c7517`: all actual ticks recorded up to each endpoint;
  sit/stand stopped at tick 1144 and remains incomplete. Zero PhysX skills are promoted.

Outstanding full-scope gates: nine-skill quality/learning, all-terrain shared sampling,
100 held-out episodes per skill, 60-second controls and transitions, full game object
interactions, sim-time aligned side-by-side video, packaging and final PR review.
