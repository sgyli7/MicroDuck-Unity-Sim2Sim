# PhysX adaptation observation ledger

Owner/goal: implement the user's game-quality Sim2Sim contract, not maximize PPO reward.
Actor contract remains 61 policy inputs, 14 actions, 50 Hz, original frozen ONNX
normalization; physics samples are produced by the independent PhysX Player at 200 Hz.
Original policies remain the rollback artifacts. No adapted model is promoted yet.

## Baseline and decision gates

The fixed strict behavior tests are not relaxed. Walking diagnostic gates already
require 6 s forward displacement 0.35–0.78 m, lateral displacement ≤0.12 m and upright
quality. These are screening gates, not the final independent 100-episode, 60-second,
terrain and switching acceptance. Final seeds have not been consumed for tuning.

After canonical reset correction, original stand ran 60 s with ≈0.0062 m drift and
minimum upright ≈0.99957. Original walk remained upright at 60 s, but direction failed
(≈1.72 m side displacement). Sitting/recovery still fails. No whole-MVP pass is claimed.

## Experiments

1. `walk-canonical-20260912`: 300 PPO iterations, reward locomotion-v1, source original
   walking actor. Candidate stayed upright but 6 s lateral displacement grew from
   ≈0.214 m to ≈1.015 m. **Rejected**. Faster movement is not better control.
2. `walk-heading-v2-20260912`: fixed zero-turn reward additionally measures heading
   relative to the episode start and yaw rate. Intermediate candidate still veered
   ≈0.997 m in 6 s. The final 500-iteration candidate moved backward ≈0.269 m and sideways
   ≈0.582 m; **rejected**. Final assessment is separate from training logs.
3. `walk-critic-v3-20260912`: 600 iterations, current-state privileged critic and
   corrected heading projection. Final 6 s forward displacement ≈0.089 m, lateral
   ≈0.023 m, final height ≈0.072 m. **Rejected**: reduced side drift came with lost
   walking progress and a crouched posture, not successful directional control.

## Next falsifiable experiment: critic state visibility

Hypothesis: the reward now depends on accumulated heading and world velocity, but the
critic sees only the actor's 61 values (no absolute heading/linear velocity). Two states
with identical policy observations can therefore have different direction rewards.
This is a candidate source of noisy value/advantage estimates, not a proven explanation
of all learning failures.

Add a training-only critic view of current PhysX state: episode-relative heading sine/
cosine, heading-relative linear velocity, height and elapsed episode fraction. Keep
the actor input, normalizer, exported architecture and game interface unchanged. Do not
feed future state or any MuJoCo-generated target body state to the actor or simulation.
Test shape, finite values, reset/terminal snapshots, actor-only export and checkpoint
schema compatibility. Start from original weights in a fresh experiment directory.

Review also found a reused quaternion-to-heading formula that was wrong for nonzero
roll/pitch in Unity's Y-up frame. Reward and critic now use the same projected-forward
heading helper, regression-tested with yaw 45° plus roll 60°. This is reward version
`locomotion-v3-heading-projection`; earlier experiments keep their original version tags.

Reject if physical directional metrics regress, even if reward/value loss improve.
The candidate is not production-ready without the user's complete frozen acceptance
matrix. Any additional calibration/learning hypothesis must get its own evidence entry.

## Physics-only recheck: reflected rotor inertia after canonical reset repair

The earlier whole-robot rotor experiment used the now-invalid reset (mutable joint
anchors/contact impulses). Repeat the existing strict behavior scenarios with only
the existing reflected-rotor component enabled, keeping original policy weights and
all thresholds unchanged. Record a separately named diagnostic report, never replace
the default game's configuration or its report. Reject the configuration if it improves
one response but regresses standing, gait, compound skills, or stability. This remains
an approximate PhysX joint-space drive treatment, not proof of MuJoCo solver equivalence.

Canonical-reset recheck (`canonical-rotor-experiment-20260912.xml`) rejected the full
rotor configuration: stand minimum upright fell below zero, walking fell and advanced
only ≈0.127 m, and roller progress became negative. Default components remain disabled.
The calibration test is diagnostic and writes a separate `rotor-experiment` report.

## Hypothesis: separate limited motor torque from the reflected-inertia drive

The rejected approximation capped the *combined* motor, mechanical damping and
reflected-inertia constraint torque at the actuator's 0.96 N m. The source actuator
limit does not cap its generalized inertia or passive joint damping. Test an opt-in
variant with the original explicit position-motor torque separately capped, and an
uncapped implicit velocity drive implementing J/dt plus mechanical damping. This is
still a discretized approximation, not native PhysX armature or equivalence proof.
Run the unchanged whole-skill contracts, record a separate split-motor diagnostic,
and leave generated/default components disabled unless all regressions are resolved.

References: [PhysX implicit articulation drives](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/docs/Articulations.html),
[MuJoCo joint armature and actuator limits](https://mujoco.readthedocs.io/en/stable/XMLreference.html),
[Unity 2022 articulation force API](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/ArticulationBody.SetJointForces.html).
Newer PhysX SDK armature APIs are not assumed available in this Tuanjie version.

Result: **rejected**. The unchanged isolated ankle calibration passed, but the
full-body split-motor strict suite diverged severely (unphysical velocities and
failed standing/walking/roller/compound motion). Evidence: `split-motor-ankle-20260912.xml`
and `split-motor-rotor-experiment-20260912.xml`; separate behavior JSON retains the
failure metrics. Single-joint agreement is insufficient to promote this approximation.

## Next training hypothesis: preserve the actor while fitting the critic first

Previous experiments started a randomly initialized critic and a 0.12 action-noise
scale together with the already-trained actor. Test smaller 0.025 exploration and
150 iterations of critic-only fitting, followed by a conservative actor learning rate
1e-6 with critic rate 1e-4. Upstream RSL-RL PPO remains unchanged; separate optimizer
groups and frozen actor gradients ensure warmup cannot alter policy weights or its
distribution. The actor still receives only 61 actual PhysX observations. Checkpoint
resume uses cumulative iteration numbers and preserves optimizer moments; original
actor/normalizer bytes must be identical throughout warmup. This hypothesis changes
neither game physics nor frozen behavioral gates. Reject any candidate that loses
forward progress, crouches, falls or increases direction error.
