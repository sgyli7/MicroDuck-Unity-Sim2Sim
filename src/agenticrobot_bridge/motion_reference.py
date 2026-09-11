"""Offline reward targets; never step MuJoCo or write a PhysX body state."""

import numpy as np

from .physx_evaluate import bound_model_identity


def local_planar(values, yaw):
    values = np.asarray(values)
    return np.column_stack((np.sin(yaw) * values[:, 0] + np.cos(yaw) * values[:, 2],
                            np.cos(yaw) * values[:, 0] - np.sin(yaw) * values[:, 2]))


class RecordedMotionReference:
    """Verified native-reference episode, sampled by integer simulation ticks.

    First reward slice supports only stand/walk. Compound skills require their own
    target-environment lifecycle and termination implementation before enabling them.
    """

    def __init__(self, report, slot, expected_source_hash, case_id):
        if slot not in (1, 2) or report.get('engine') != 'MuJoCo':
            raise ValueError('Require native MuJoCo stand/walk reference')
        if report.get('coordinateBasis') != 'right-handed X-forward Y-left Z-up; quaternion WXYZ':
            raise ValueError('Reference coordinate basis is not native MuJoCo')
        _, verified = bound_model_identity(report, slot, expected_source_hash)
        if not verified:
            raise ValueError('Unverified source model')
        episodes = [row for row in report.get('episodes', [])
                    if row.get('slot') == slot and row.get('experimentId') == case_id]
        if len(episodes) != 1 or episodes[0].get('completed') is not True:
            raise ValueError('Require one complete, explicitly identified reference episode')
        episode = episodes[0]
        if episode.get('inference') != 'Barracuda' or not episode.get('experimentSha256'):
            raise ValueError('Missing native inference/experiment provenance')
        frames = episode['frames']
        if episode.get('error') or any(frame.get('healthy') is not True or frame.get('fault')
                                       or frame.get('upright', -1) < .9 for frame in frames):
            raise ValueError('Faulted/unstable locomotion cannot be a desired motion reference')
        ticks = [frame['physicsSteps'] for frame in frames]
        if (not ticks or ticks != list(range(ticks[-1] + 1)) or ticks[-1] < 4 or ticks[-1] % 4
                or not np.isclose(ticks[-1] * .005, episode['requestedSeconds'], atol=1e-7)):
            raise ValueError('Reference has missing ticks or an incomplete horizon')
        for frame in frames[1:]:
            if (frame.get('engine') != 'MuJoCo'
                    or frame.get('inferenceBackend') != 'MuJoCo 3.12 + Barracuda 3.0.1 CPU'
                    or frame.get('inferenceSlot') != slot or frame.get('activeSlot') != slot):
                raise ValueError('Unexpected reference backend')
        if frames[0].get('activeSlot') != slot:
            raise ValueError('Initial reference policy differs from the requested skill')
        self.max_steps = ticks[-1] // 4
        self.slot = slot
        self.provenance = {'case_id': case_id, 'experiment_sha256': episode['experimentSha256'],
                           'source_sha256': expected_source_hash, 'build_guid': report['buildGuid'],
                           'reward_version': 'measured-motion-v1', 'reference_horizon_steps': self.max_steps,
                           'usage': 'offline reward targets only; all training states are independent PhysX'}

        def values(name, width):
            array = np.asarray([frame[name] for frame in frames], dtype=np.float64)
            if array.shape != (len(frames), width) or not np.isfinite(array).all():
                raise ValueError('Malformed/non-finite reference ' + name)
            return array

        native_position = values('rootPosition', 3)
        native_velocity = values('rootVelocity', 3)
        quaternion = values('rootQuaternionWxyz', 4)
        if not np.allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-5):
            raise ValueError('Invalid reference rotation')
        w, x, y, z = quaternion.T
        yaw = -np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        position = np.column_stack((-native_position[:, 1], native_position[:, 2], native_position[:, 0]))
        velocity = np.column_stack((-native_velocity[:, 1], native_velocity[:, 2], native_velocity[:, 0]))
        self.position = local_planar(position - position[0], yaw[0])
        self.velocity = local_planar(velocity, yaw[0])
        self.height = position[:, 1]
        self.yaw = yaw - yaw[0]
        self.gravity = np.column_stack((2 * (w * y - x * z), -2 * (w * x + y * z),
                                         2 * (x * x + y * y) - 1))
        self.joints = values('jointPosition', 14)
        self.joint_velocity = values('jointVelocity', 14)
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def reward(self, frames, elapsed_steps, initial_positions, initial_yaw):
        steps = np.asarray(elapsed_steps)
        if (steps.shape != (len(frames),) or not np.equal(steps, np.round(steps)).all()
                or (steps < 0).any() or (steps > self.max_steps).any()):
            raise ValueError('Reference horizon exhausted or invalid step index')
        ticks = steps.astype(int) * 4
        position = np.asarray([frame['rootPosition'] for frame in frames])
        velocity = np.asarray([frame['rootVelocity'] for frame in frames])
        q = np.asarray([frame['jointPosition'] for frame in frames])
        qd = np.asarray([frame['jointVelocity'] for frame in frames])
        gravity = np.asarray([frame['observation'][3:6] for frame in frames])
        x, y, z, w = np.asarray([frame['rootRotation'] for frame in frames]).T
        yaw = np.arctan2(2 * (w * y + x * z), 1 - 2 * (x * x + y * y)) - initial_yaw
        yaw_error = yaw - self.yaw[ticks]
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        def tracking(error, scale):
            error = np.asarray(error)
            if error.ndim == 1:
                error = error[:, None]
            return np.exp(-np.square(error / scale).mean(axis=1))

        reward = (1.5 * tracking(q - self.joints[ticks], .25)
                  + .5 * tracking(qd - self.joint_velocity[ticks], 4.)
                  + 2. * tracking(local_planar(velocity, initial_yaw) - self.velocity[ticks], .1)
                  + tracking(position[:, 1] - self.height[ticks], .015)
                  + 1.5 * tracking(gravity - self.gravity[ticks], .15)
                  + tracking(yaw_error, .2)
                  + 1.5 * tracking(local_planar(position - initial_positions, initial_yaw)
                                    - self.position[ticks], .06))
        failed = ((position[:, 1] < .065) | (np.asarray([f['upright'] for f in frames]) < .45)
                  | ~np.asarray([frame['healthy'] for frame in frames]))
        if not np.isfinite(reward).all():
            raise ValueError('Non-finite actual PhysX reward input')
        return np.where(failed, -10., reward).astype(np.float32), failed
