"""Pinned upstream PolicyInference, with Windows/headless IO adaptation only.

This is a distinct official-source diagnostic, not the frozen current Unity
reference and not PhysX behavior acceptance. Timers use simulation time instead
of the interactive upstream viewer's wall clock; no policy methods are rewritten.
"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import subprocess
import types
from pathlib import Path

import mujoco
import numpy as np

from .physx_evaluate import POLICIES
from .shared_experiment import make_batch, write_new


def load_controller(root):
    root = Path(root).resolve()
    repository = root / '.cache/upstream/microduck_rl'
    revision = json.loads((root / 'upstream.lock.json').read_text())['repositories']['microduck_rl']['commit']
    path = repository / 'scripts/infer_policy.py'
    source = path.read_bytes()
    pinned = subprocess.check_output(['git', 'show', revision + ':scripts/infer_policy.py'], cwd=repository)
    if source.replace(b'\r\n', b'\n') != pinned.replace(b'\r\n', b'\n'):
        raise ValueError('Upstream controller differs from the pinned source')
    tree = ast.parse(source)
    policy_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PolicyInference')
    original_ast = ast.dump(policy_class)
    removed = []
    kept = []
    for node in tree.body:
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name in ('termios', 'tty'):
            removed.append(node.names[0].name)
        else:
            kept.append(node)
    tree.body = kept
    module = types.ModuleType('pinned_microduck_headless')
    module.__file__ = str(path)
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module, {'sourceRevision': revision, 'scriptSha256': hashlib.sha256(source).hexdigest(),
                    'removedImports': removed, 'policyClassAstUnchanged': ast.dump(policy_class) == original_ast,
                    'timerAdapter': 'fixed simulation-time dt; upstream viewer uses wall clock',
                    'interactiveStartupAdapter': 'policy active immediately; no keyboard standby'}


def run_case(root, case):
    root = Path(root).resolve()
    if case['terrain'] != 'flat' or case['physicsSteps'] < 4 or case['physicsSteps'] % 4:
        raise ValueError('Only flat, positive control-aligned cases currently supported')
    module, provenance = load_controller(root)
    slot = case['slot']
    policies = root / '.cache/upstream/microduck/policies'
    source_root = root / '.cache/upstream/microduck_rl/src/mjlab_microduck/robot/microduck'
    subtree = 'src/mjlab_microduck/robot/microduck'
    repository = root / '.cache/upstream/microduck_rl'
    subprocess.run(['git', 'diff', '--quiet', provenance['sourceRevision'], '--', subtree],
                   cwd=repository, check=True)
    provenance['robotTree'] = subprocess.check_output(
        ['git', 'rev-parse', provenance['sourceRevision'] + ':' + subtree], cwd=repository, text=True).strip()
    scene = 'scene_rollers.xml' if slot in (7, 8) else 'scene_ball.xml' if slot in (5, 6) else 'scene.xml'
    model = mujoco.MjModel.from_xml_path(str(source_root / scene))
    model.opt.timestep = .005
    # Same bundled parameter file read by upstream bam.model.load_model(xl330,m6).
    # Reading this scalar directly avoids importing a different venv's MuJoCo DLL.
    bam_path = root / '.cache/upstream/microduck_rl/.venv/Lib/site-packages/bam/params/xl330/m6.json'
    bam_bytes = bam_path.read_bytes()
    torque_limit = json.loads(bam_bytes)['kt'] * 1.75
    model.actuator_forcerange[:] = [-torque_limit, torque_limit]
    model.actuator_forcelimited[:] = 1
    if slot in (7, 8):
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and name.startswith('passive_'):
                model.dof_frictionloss[model.jnt_dofadr[j]] = .003
    data = mujoco.MjData(model)
    compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=compiled)
    provenance['configuredCompiledModelSha256'] = hashlib.sha256(compiled.tobytes()).hexdigest()
    kwargs = {'new_cmd_obs': True, 'use_projected_gravity': True}
    bindings = {}
    policy_hashes = {}

    def bind(argument, selected):
        kwargs[argument] = str(policies / (POLICIES[selected - 1] + '.onnx'))
        bindings[argument] = selected
        policy_hashes[argument] = hashlib.sha256(Path(kwargs[argument]).read_bytes()).hexdigest()

    if slot in (1, 7, 8):
        bind('walking_onnx_path', 7 if slot in (7, 8) else 1)
    elif slot == 3:
        bind('sitstand_onnx_path', 3)
    else:
        bind('standing_onnx_path', 2)
    if slot in (4, 8):
        bind('ground_pick_onnx_path', slot)
    if slot in (5, 6, 9):
        bind({5: 'kick_left_onnx_path', 6: 'kick_right_onnx_path', 9: 'roulade_onnx_path'}[slot], slot)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = module.PolicyInference(model, data, **kwargs)
    session_attributes = {'walking_onnx_path': 'walking_session', 'standing_onnx_path': 'standing_session',
                          'sitstand_onnx_path': 'sit_session', 'ground_pick_onnx_path': 'ground_pick_session'}
    sessions = {id(getattr(policy, session_attributes[key])): selected for key, selected in bindings.items()
                if key in session_attributes}
    sessions.update({id(session): {'kick_left': 5, 'kick_right': 6, 'roulade': 9}[name]
                     for name, session in policy.behavior_sessions.items() if session is not None})
    adr = policy._trunk_qpos_adr
    free_joint = model.joint('trunk_base_freejoint')
    velocity_adr = int(free_joint.dofadr[0])
    x, y, z = case['initialRootPosition']
    yaw = -math.radians(case['initialYawDegrees'])
    data.qpos[adr:adr + 7] = [z, -x, y, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    data.qpos[policy.joint_qpos_indices] = module.DEFAULT_POSE
    data.ctrl[:] = module.DEFAULT_POSE
    mujoco.mj_forward(model, data)
    frames = []
    ignored = []
    policy_obs = np.zeros(61, np.float32)
    action = np.zeros(14, np.float32)
    inference_slot = sessions[id(policy.ort_session)]
    fault = None

    def capture(tick):
        q = data.qpos[adr:adr + 7]
        v = data.qvel[velocity_adr:velocity_adr + 6]
        qw, qx, qy, qz = q[3:7]
        frames.append({'engine': 'MuJoCo', 'physicsSteps': tick, 'timeSeconds': tick * .005,
                       'inferenceSlot': inference_slot, 'activeSlot': sessions[id(policy.ort_session)],
                       'policyObservation': policy_obs.tolist(), 'action': action.tolist(),
                       'targets': data.ctrl.tolist(), 'rootPosition': [-q[1], q[2], q[0]],
                       'rootRotation': [qy, -qz, -qx, qw],
                       'rootVelocity': [-v[1], v[2], v[0]],
                       'upright': 1 - 2 * (qx * qx + qy * qy),
                       'jointPosition': data.qpos[policy.joint_qpos_indices].tolist(),
                       'jointVelocity': data.qvel[policy.joint_qvel_indices].tolist(),
                       'contacts': int(data.ncon)})

    capture(0)
    with contextlib.redirect_stdout(io.StringIO()):
        # Selecting a kick slot in the game corresponds to triggering the official behavior.
        if slot in (5, 6):
            policy.trigger_behavior('kick_left' if slot == 5 else 'kick_right')
        for tick in range(0, case['physicsSteps'], 4):
            for event in case['events']:
                if event['physicsStep'] != tick:
                    continue
                if event['kind'] == 'twist':
                    policy.set_vel_cmd(*event['values'])
                elif event['kind'] == 'trigger':
                    if slot == 3:
                        policy.toggle_sit()
                    elif slot in (4, 8):
                        policy.trigger_ground_pick()
                    elif slot == 9:
                        policy.trigger_behavior('roulade')
                elif event['kind'] == 'switch':
                    # Activation uses the trigger immediately following this event.
                    # Return to the base session uses the existing upstream end method.
                    if event['slot'] == 7 and policy.ground_pick_mode:
                        policy._end_ground_pick()
                    elif event['slot'] == 2 and policy.behavior_mode:
                        policy._end_behavior()
                    else:
                        ignored.append({'physicsStep': tick, 'switch': event['slot'],
                                        'reason': 'source trigger/automatic lifecycle owns session selection'})
            policy.update_ground_pick_phase(.02)
            policy.update_behavior(.02)
            policy_obs = policy.get_observations().copy()
            inference_slot = sessions[id(policy.ort_session)]
            action = policy.infer()
            policy.apply_action(action)
            for _ in range(4):
                mujoco.mj_step(model, data)
            capture(tick + 4)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                fault = 'nonfinite_physics'
                break
            if not math.isclose(data.time, (tick + 4) * .005, abs_tol=1e-8):
                fault = 'unexpected_native_time_or_automatic_reset'
                break
    for key in bindings:
        if hashlib.sha256(Path(kwargs[key]).read_bytes()).hexdigest() != policy_hashes[key]:
            raise ValueError('Policy file changed during original-source inference')
    return {'engine': 'MuJoCo', 'version': mujoco.__version__, 'role': 'official-source-headless-adapter',
            'coordinateBasis': 'Unity-Xright-Yup-Zforward',
            'case': case, 'completed': fault is None and frames[-1]['physicsSteps'] == case['physicsSteps'],
            'fault': fault,
            'behaviorAccepted': False, 'provenance': provenance, 'adaptedEvents': ignored,
            'models': [{'slot': selected, 'sha256': policy_hashes[key]}
                       for key, selected in bindings.items()],
            'settings': {'actuatorTorqueLimit': torque_limit, 'currentLimitAmps': 1.75,
                         'bamParametersSha256': hashlib.sha256(bam_bytes).hexdigest(),
                         'actionScale': policy.action_scale, 'targetFilter': 'none',
                         'phasePeriod': policy.ground_pick_period, 'phaseAutoExit': .7,
                         'home': module.DEFAULT_POSE.tolist(), 'scene': scene},
            'frames': frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=92317)
    args = parser.parse_args()
    root = Path(__file__).parents[2]
    batch = make_batch(root / '.cache/upstream/microduck/policies', args.seed)
    results = [run_case(root, case) for case in batch['cases']]
    write_new(args.output, {'purpose': 'official-source diagnostic, distinct from frozen current reference',
                           'behaviorAccepted': False, 'results': results})
    print(json.dumps([{'slot': item['case']['slot'], 'completed': item['completed'],
                       'finalUpright': item['frames'][-1]['upright']} for item in results]))


if __name__ == '__main__':
    main()
