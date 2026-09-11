"""RSL-RL PPO adaptation with training samples produced only by the PhysX Player.

Initial supported reward objectives: stand and walk. Other skills fail explicitly
until their task-specific reward and termination contracts are implemented.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .physx_actor import ExecutionEnvelope, RestoredActor, actor_from_onnx, export_actor
from .physx_client import PhysXClient


def heading_yaw(frames):
    """Heading of Unity's +Z-forward axis projected on the XZ ground plane."""
    x, y, z, w = np.asarray([frame["rootRotation"] for frame in frames]).T
    return np.arctan2(2 * (w * y + x * z), 1 - 2 * (x * x + y * y))


def locomotion_reward(frames, actions, slot, initial_yaw=None):
    if slot not in (1, 2):
        raise ValueError(f"Reward for skill slot {slot} is not implemented")
    observation = np.asarray([f["observation"] for f in frames], dtype=np.float32)
    height = np.asarray([f["rootPosition"][1] for f in frames])
    upright = np.asarray([f["upright"] for f in frames])
    velocity = np.asarray([f["rootVelocity"] for f in frames])
    yaw = heading_yaw(frames)
    forward_speed = np.sin(yaw) * velocity[:, 0] + np.cos(yaw) * velocity[:, 2]
    side_speed = np.cos(yaw) * velocity[:, 0] - np.sin(yaw) * velocity[:, 2]
    healthy = np.asarray([f["healthy"] for f in frames])
    target_speed = 0.2 if slot == 1 else 0.0
    speed_error = (forward_speed - target_speed) ** 2 + side_speed ** 2
    reward = (
        1.0 + 2.0 * np.clip(upright, 0, 1)
        + 2.0 * np.exp(-((height - 0.125) / 0.035) ** 2)
        + 2.0 * np.exp(-speed_error / 0.04)
        - 0.002 * np.square(observation[:, 20:34]).sum(axis=1)
        - 0.01 * np.square(actions).sum(axis=1)
        - 0.02 * np.square(observation[:, :3]).sum(axis=1)
    )
    if initial_yaw is not None:
        # These two objectives command zero turn rate. Tracking only body-local
        # velocity rewards a robot that curves indefinitely while walking forward.
        # Preserve the episode's own heading, not an arbitrary world-axis heading.
        heading_error = np.arctan2(np.sin(yaw - initial_yaw), np.cos(yaw - initial_yaw))
        reward += 2.0 * np.exp(-np.square(heading_error / 0.25))
        reward += np.exp(-np.square(observation[:, 2] / 0.2))
    failed = (height < 0.065) | (upright < 0.45) | ~healthy
    reward = np.where(failed, -10.0, reward)
    return reward.astype(np.float32), failed


class PhysXVecEnv(VecEnv):
    def __init__(self, client, slot, device="cuda", episode_seconds=6.0):
        if slot not in (1, 2):
            raise ValueError("Only stand/walk adaptation rewards are currently implemented")
        self.client = client
        self.slot = slot
        self.device = device
        self.num_envs = client.identity["numEnvs"]
        self.num_actions = 14
        self.max_episode_length = int(round(episode_seconds / 0.02))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.cfg = {"physics": "PhysX", "slot": slot, "dt": 0.005,
                    "control_dt": 0.02, "reset_microstep": 0,
                    "critic_observation_version": 1,
                    "critic_extras": ["heading_sin", "heading_cos", "initial_heading_forward_velocity",
                                      "initial_heading_right_velocity", "up_velocity",
                                      "height", "elapsed_fraction"],
                    "action_execution_clip": [-5, 5], "reward_version": "locomotion-v3-heading-projection"}
        self.frames = client.reset(slot, external=True)
        self.initial_yaw = self._yaw(self.frames)
        self.episodes = []
        self.clipped_values = 0

    @staticmethod
    def _yaw(frames):
        return heading_yaw(frames)

    def get_observations(self):
        values = torch.tensor([f["observation"] for f in self.frames],
                              dtype=torch.float32, device=self.device)
        if not torch.isfinite(values).all():
            raise RuntimeError("Non-finite actual PhysX observation")
        relative_yaw = self._yaw(self.frames) - self.initial_yaw
        velocity = np.asarray([frame["rootVelocity"] for frame in self.frames])
        forward = np.sin(self.initial_yaw) * velocity[:, 0] + np.cos(self.initial_yaw) * velocity[:, 2]
        side = np.cos(self.initial_yaw) * velocity[:, 0] - np.sin(self.initial_yaw) * velocity[:, 2]
        extra = np.column_stack((np.sin(relative_yaw), np.cos(relative_yaw),
                                 forward, side, velocity[:, 1],
                                 [frame["rootPosition"][1] for frame in self.frames],
                                 self.episode_length_buf.cpu().numpy() / self.max_episode_length))
        privileged = torch.tensor(extra, dtype=torch.float32, device=self.device)
        if not torch.isfinite(privileged).all():
            raise RuntimeError("Non-finite current PhysX critic state")
        return TensorDict({"policy": values, "critic": torch.cat((values, privileged), dim=-1)},
                          batch_size=[self.num_envs])

    def step(self, actions):
        if tuple(actions.shape) != (self.num_envs, 14) or not torch.isfinite(actions).all():
            raise ValueError("Expected finite [num_envs,14] actor actions")
        self.clipped_values += int((actions.abs() > 5).sum())
        executed = actions.clamp(-5, 5).detach().cpu().numpy()
        self.frames = self.client.step(executed.tolist())
        rewards, failed = locomotion_reward(self.frames, executed, self.slot, self.initial_yaw)
        self.episode_length_buf += 1
        timeouts = (self.episode_length_buf >= self.max_episode_length)
        failure_tensor = torch.tensor(failed, device=self.device)
        # A fall coincident with the horizon is still a terminal failure, not a timeout.
        timeouts = timeouts & ~failure_tensor
        dones = failure_tensor | timeouts
        terminal_observations = self.get_observations()
        indices = dones.nonzero().flatten().tolist()
        if indices:
            for index in indices:
                self.episodes.append({"steps": int(self.episode_length_buf[index]),
                                      "failed": bool(failed[index]),
                                      "timeout": bool(timeouts[index])})
            reset_frames = self.client.reset(self.slot, external=True, indices=indices)
            for index, frame in zip(indices, reset_frames, strict=True):
                self.frames[index] = frame
                self.initial_yaw[index] = self._yaw([frame])[0]
                self.episode_length_buf[index] = 0
        return (self.get_observations(), torch.tensor(rewards, device=self.device),
                dones, {"time_outs": timeouts, "terminal_observations": terminal_observations})


def record_transition(algorithm, observations, rewards, dones, extras):
    """Bootstrap truncations at s_(t+1), not RSL-RL's default V(s_t)."""
    timeouts = extras["time_outs"]
    with torch.no_grad():
        terminal_values = algorithm.critic(extras["terminal_observations"]).squeeze(-1)
        bootstrapped = rewards + algorithm.gamma * terminal_values * timeouts
    # The reward is already corrected. Do not ask RSL-RL to add its legacy timeout
    # correction as well. True task failures remain terminal with no bootstrap.
    algorithm.process_env_step(observations, bootstrapped, dones, {})


def ppo_config(learning_rate=1e-5):
    return {
        "actor": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128],
                  "activation": "elu", "obs_normalization": False,
                  "distribution_cfg": {"class_name": "GaussianDistribution",
                                       "init_std": 0.12, "std_type": "log"}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 128, 64],
                   "activation": "elu", "obs_normalization": True},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 4,
                      "num_mini_batches": 4, "learning_rate": learning_rate,
                      "schedule": "fixed", "desired_kl": 0.01,
                      "entropy_coef": 0.001, "rnd_cfg": None},
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "num_steps_per_env": 32, "multi_gpu": None,
    }


def train(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Use a fresh output directory or explicitly resume a checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    original = Path(args.policy).resolve()
    source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    initial = actor_from_onnx(original).to(args.device)
    config = ppo_config(args.learning_rate)
    started = time.monotonic()
    with PhysXClient(port=args.port) as client:
        env = PhysXVecEnv(client, args.slot, args.device, args.episode_seconds)
        algorithm = PPO.construct_algorithm(env.get_observations(), env,
                                            copy.deepcopy(config), args.device)
        algorithm.actor.mlp.load_state_dict(initial.mlp.state_dict(), strict=True)
        algorithm.actor.obs_normalizer = copy.deepcopy(initial.obs_normalizer)
        with torch.no_grad():
            torch.testing.assert_close(algorithm.actor(env.get_observations()),
                                       initial(env.get_observations()["policy"]))
        first_iteration = 0
        resume_provenance = None
        if args.resume:
            checkpoint = torch.load(args.resume, weights_only=True, map_location=args.device)
            if checkpoint["source_sha256"] != source_hash or checkpoint["slot"] != args.slot:
                raise ValueError("Checkpoint source/skill does not match this adaptation")
            if checkpoint.get("critic_observation_version") != env.cfg["critic_observation_version"]:
                raise ValueError("Checkpoint critic observation schema differs; start a fresh experiment")
            algorithm.load(checkpoint, load_cfg=None, strict=True)
            # The requested LR governs this continuation; preserve optimizer moments,
            # but do not silently let the checkpoint override the run configuration.
            algorithm.learning_rate = args.learning_rate
            for group in algorithm.optimizer.param_groups:
                group["lr"] = args.learning_rate
            resume_provenance = {
                "checkpoint": str(Path(args.resume).resolve()),
                "checkpointSha256": hashlib.sha256(Path(args.resume).read_bytes()).hexdigest(),
                "environmentStateRestored": False,
                "rngStateRestored": False,
                "semantics": "optimizer/model continuation with fresh seeded episodes",
            }
            first_iteration = checkpoint["iteration"] + 1
        metadata = {"engine": "PhysX", "sampling": client.identity, "source": str(original),
                    "source_sha256": source_hash, "seed": args.seed, "slot": args.slot,
                    "env": env.cfg, "ppo": config, "torch": torch.__version__,
                    "resume": resume_provenance,
                    "actual_learning_rates": [group["lr"] for group in algorithm.optimizer.param_groups],
                    "adapted": True, "behavior_accepted": False}
        source_root = Path(__file__).parents[2]
        metadata["training_code_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
        metadata["training_source_hashes"] = {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in Path(__file__).parent.glob("physx_*.py")
        }
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        def save(iteration):
            checkpoint = algorithm.save()
            checkpoint.update(iteration=iteration, source_sha256=source_hash, slot=args.slot)
            checkpoint["critic_observation_version"] = env.cfg["critic_observation_version"]
            torch.save(checkpoint, output / f"checkpoint_{iteration:06d}.pt")
            export_actor(ExecutionEnvelope(RestoredActor(
                         algorithm.actor.obs_normalizer, algorithm.actor.mlp)),
                         output / (original.stem + "_PhysX.onnx"))

        obs = env.get_observations()
        algorithm.train_mode()
        for iteration in range(first_iteration, first_iteration + args.iterations):
            collection_start = time.monotonic()
            mean_reward = 0.0
            with torch.inference_mode():
                for _ in range(config["num_steps_per_env"]):
                    actions = algorithm.act(obs)
                    obs, rewards, dones, extras = env.step(actions)
                    record_transition(algorithm, obs, rewards, dones, extras)
                    mean_reward += float(rewards.mean()) / config["num_steps_per_env"]
                algorithm.compute_returns(obs)
            losses = algorithm.update()
            record = {"iteration": iteration, "seconds": time.monotonic() - started,
                      "iteration_seconds": time.monotonic() - collection_start,
                      "mean_step_reward": mean_reward, "losses": losses,
                      "episodes": env.episodes, "clipped_action_values": env.clipped_values}
            with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps({key: record[key] for key in
                              ("iteration", "seconds", "mean_step_reward")}), flush=True)
            env.episodes = []
            if iteration % args.save_interval == 0:
                save(iteration)
        save(iteration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--slot", type=int, required=True, choices=[1, 2])
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=62102)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--episode-seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=37141)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.iterations < 1 or args.save_interval < 1 or args.episode_seconds <= 0:
        parser.error("Iterations, save interval and episode duration must be positive")
    train(args)


if __name__ == "__main__":
    main()
