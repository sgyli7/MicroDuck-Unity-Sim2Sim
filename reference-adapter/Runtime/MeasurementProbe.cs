using System;
using System.Collections.Generic;
using AgenticRobot.MicroDuck;
using AgenticRobot.MicroDuck.Mujoco;
using Mujoco;
using UnityEngine;

namespace AgenticRobot.Reference
{
    [Serializable]
    public sealed class ReferenceFrame
    {
        public string engine = "MuJoCo";
        public double timeSeconds;
        public int physicsSteps;
        public int activeSlot;
        public int inferenceSlot;
        public double[] jointPosition;
        public double[] jointVelocity;
        public double[] passiveWheelVelocity;
        public float[] policyObservation;
        public float[] action;
        public float[] targets;
        public double[] qpos;
        public double[] qvel;
        public double[] rootPosition;
        public double[] rootQuaternionWxyz;
        public double[] rootVelocity;
        public double[] rootAngularVelocity;
        public double upright;
        public int contacts;
        public bool healthy;
        public string fault;
    }

    // Passive observer only: never calls mj_step/forward, changes time, or writes model/data.
    public sealed class MeasurementProbe : MonoBehaviour
    {
        private MjScene scene;
        private MujocoDemoController controller;
        public readonly List<ReferenceFrame> Frames = new List<ReferenceFrame>();
        public int SampleCount => Frames.Count;

        private void Awake()
        {
            scene = FindObjectOfType<MjScene>();
            controller = FindObjectOfType<MujocoDemoController>();
            if (scene == null || controller == null)
                throw new InvalidOperationException("Measurement adapter requires the frozen native scene");
            scene.postUpdateEvent += Observe;
        }

        private void Observe(object sender, MjStepArgs args) => Frames.Add(Capture());

        public unsafe ReferenceFrame Capture()
        {
            var model = scene.Model;
            var data = scene.Data;
            int root = MujocoLib.mj_name2id(model, (int)MujocoLib.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint");
            if (root < 0) throw new InvalidOperationException("Reference root joint is missing");
            int address = model->jnt_qposadr[root];
            int dofAddress = model->jnt_dofadr[root];
            var frame = new ReferenceFrame
            {
                timeSeconds = data->time,
                physicsSteps = (int)Math.Round(data->time / 0.005),
                activeSlot = controller.ActivePolicySlot,
                inferenceSlot = data->time == 0 ? 0 : controller.ActivePolicySlot,
                jointPosition = new double[14], jointVelocity = new double[14],
                passiveWheelVelocity = new double[controller.ActivePolicySlot == 7 || controller.ActivePolicySlot == 8 ? 4 : 0],
                policyObservation = (float[])controller.LastObservation.Clone(),
                action = (float[])controller.LastRawAction.Clone(),
                targets = (float[])controller.LastTargets.Clone(),
                qpos = new double[model->nq], qvel = new double[model->nv],
                rootPosition = new[] { data->qpos[address], data->qpos[address + 1], data->qpos[address + 2] },
                rootQuaternionWxyz = new[] { data->qpos[address + 3], data->qpos[address + 4],
                    data->qpos[address + 5], data->qpos[address + 6] },
                rootVelocity = new[] { data->qvel[dofAddress], data->qvel[dofAddress + 1], data->qvel[dofAddress + 2] },
                rootAngularVelocity = new[] { data->qvel[dofAddress + 3], data->qvel[dofAddress + 4], data->qvel[dofAddress + 5] },
                upright = 1 - 2 * (data->qpos[address + 4] * data->qpos[address + 4]
                    + data->qpos[address + 5] * data->qpos[address + 5]),
                // Contacts describe the just-solved interval, not a new collision query
                // at the post-integration pose. Never call mj_forward just for logging.
                contacts = data->ncon, healthy = controller.IsHealthy, fault = controller.Fault,
            };
            for (int i = 0; i < frame.qpos.Length; i++) frame.qpos[i] = data->qpos[i];
            for (int i = 0; i < frame.qvel.Length; i++) frame.qvel[i] = data->qvel[i];
            for (int i = 0; i < 14; i++)
            {
                int joint = MujocoLib.mj_name2id(model, (int)MujocoLib.mjtObj.mjOBJ_JOINT, PolicyContract.ServoNames[i]);
                if (joint < 0) throw new InvalidOperationException("Reference servo joint missing: " + PolicyContract.ServoNames[i]);
                frame.jointPosition[i] = data->qpos[model->jnt_qposadr[joint]];
                frame.jointVelocity[i] = data->qvel[model->jnt_dofadr[joint]];
            }
            for (int i = 0; i < frame.passiveWheelVelocity.Length; i++)
            {
                int joint = MujocoLib.mj_name2id(model, (int)MujocoLib.mjtObj.mjOBJ_JOINT, PolicyContract.RollerPassiveWheelNames[i]);
                if (joint < 0) throw new InvalidOperationException("Reference passive wheel joint missing");
                frame.passiveWheelVelocity[i] = data->qvel[model->jnt_dofadr[joint]];
            }
            return frame;
        }

        private void OnDestroy()
        {
            if (scene != null) scene.postUpdateEvent -= Observe;
        }
    }
}
