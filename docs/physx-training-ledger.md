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
