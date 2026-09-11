using System;
using AgenticRobot.Experiments;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace AgenticRobot.MicroDuck
{
    [Serializable]
    public sealed class PhysXStepResult
    {
        public string engine = "PhysX";
        public string policy;
        public int activeSlot;
        public int inferenceSlot;
        public string experimentId;
        public string experimentSha256;
        public float timeSeconds;
        public int physicsSteps;
        public float[] observation;
        public float[] policyObservation;
        public float[] action;
        public float[] jointPosition;
        public float[] jointVelocity;
        public float[] passiveWheelVelocity;
        public float[] rootPosition;
        public float[] rootRotation;
        public float[] rootVelocity;
        public float[] rootAngularVelocity;
        public float[] targets;
        public float[] ballPosition;
        public float upright;
        public bool healthy;
        public string fault;
    }

    public sealed class PhysXPolicySession : IDisposable
    {
        private sealed class ExternalRuntime : IPolicyRuntime
        {
            public float[] action = new float[14];
            public string BackendName => "external actor / real PhysX";
            public void Evaluate(float[] observation, float[] destination)
            { Array.Copy(action, destination, 14); }
            public void Dispose() { }
        }

        private readonly MicroDuckDemoController controller;
        private readonly PhysicsScene scene;
        private readonly SimulationMode priorSimulation;
        private ExternalRuntime external;
        private int steps;
        private bool sitTriggered;
        private bool standTriggered;
        private ExperimentCase experiment;
        private string experimentHash;
        private int nextEvent;
        public MicroDuckDemoController Controller => controller;

        public PhysXPolicySession(MicroDuckDemoController controller)
        {
            this.controller = controller ?? throw new ArgumentNullException(nameof(controller));
            scene = controller.gameObject.scene.GetPhysicsScene();
            priorSimulation = Physics.simulationMode;
            Physics.simulationMode = SimulationMode.Script;
            controller.enabled = false;
        }

        public PhysXStepResult Reset(int slot, bool externalActions = false)
        {
            experiment = null;
            experimentHash = null;
            if (!controller.SelectPolicy(slot)) throw new InvalidOperationException(controller.Fault);
            controller.ResetActiveRig();
            steps = 0;
            sitTriggered = standTriggered = false;
            external = externalActions ? new ExternalRuntime() : null;
            if (external != null) controller.UsePolicyRuntime(external);
            if (slot == 1) controller.SetTwist(0.2f, 0f, 0f);
            if (slot == 7) controller.SetTwist(0.3f, 0f, 0f);
            if (slot == 4 || slot == 8) controller.TriggerSkill(0f);
            Physics.SyncTransforms();
            // Do not simulate during reset: tiny timesteps can inject contact impulses.
            // Reduced joint coordinates and the teleported root are already readable.
            return Capture();
        }

        public PhysXStepResult ResetExperiment(ExperimentCase spec, string inputHash)
        {
            spec.Validate();
            // Clear command/phase history exactly as an ordinary reset, then apply
            // the specified pose without changing the saved default game reset pose.
            Reset(spec.initialSlot, false);
            var rig = controller.ActiveRig;
            var originalPosition = rig.ResetPosition;
            var originalRotation = rig.ResetRotation;
            try
            {
                rig.SetResetPose(new Vector3(spec.initialRootPosition[0], spec.initialRootPosition[1],
                    spec.initialRootPosition[2]), Quaternion.Euler(0, spec.initialYawDegrees, 0));
                controller.ResetActiveRig();
            }
            finally { rig.SetResetPose(originalPosition, originalRotation); }
            experiment = spec;
            experimentHash = inputHash;
            nextEvent = 0;
            Physics.SyncTransforms();
            return Capture();
        }

        public PhysXStepResult Step(float[] action = null)
        {
            if (experiment != null && steps >= experiment.physicsSteps)
                throw new InvalidOperationException("Experiment horizon reached");
            if (external != null)
            {
                PolicyContract.ValidateAction(action);
                foreach (float value in action)
                    if (float.IsNaN(value) || float.IsInfinity(value) || Mathf.Abs(value) > 5f)
                        throw new ArgumentException("External action is non-finite or outside [-5, 5]");
                external.action = (float[])action.Clone();
            }
            else if (action != null) throw new InvalidOperationException("Reset in external actor mode first");
            float now = steps * 0.005f;
            if (!controller.TickOnce(now)) throw new InvalidOperationException(controller.Fault);
            for (int i = 0; i < 4; i++)
            {
                controller.ActiveRig.PreparePhysicsStep(0.005f);
                scene.Simulate(0.005f);
                steps++;
            }
            return Capture();
        }

        private PhysXStepResult Capture()
        {
            float now = steps * 0.005f;
            // Preserve the invocation that produced this interval's action before a
            // scheduled switch replaces the control loop for the NEXT invocation.
            var executedObservation = controller.LastObservation;
            var executedAction = controller.LastRawAction;
            int executedSlot = steps == 0 ? 0 : controller.ActivePolicySlot;
            // Update commands before publishing the observation that the next action uses.
            // Both internal and external actors then see exactly the same timeline.
            if (experiment != null)
            {
                while (nextEvent < experiment.events.Length && experiment.events[nextEvent].physicsStep <= steps)
                {
                    var item = experiment.events[nextEvent++];
                    if (item.kind == "switch" && !controller.HotSwapPolicy(item.slot))
                        throw new InvalidOperationException(controller.Fault);
                    if (item.kind == "twist") controller.SetTwist(item.values[0], item.values[1], item.values[2]);
                    if (item.kind == "trigger") controller.TriggerSkill(now);
                }
            }
            else if (controller.ActivePolicySlot == 3)
            {
                if (!sitTriggered && steps >= 200) { controller.TriggerSkill(now); sitTriggered = true; }
                if (!standTriggered && steps >= 900) { controller.TriggerSkill(now); standTriggered = true; }
            }
            var root = controller.ActiveRig.RootBody;
            var rotation = root.transform.rotation;
            return new PhysXStepResult
            {
                policy = controller.ActivePolicyName,
                activeSlot = controller.ActivePolicySlot,
                inferenceSlot = executedSlot,
                experimentId = experiment?.id, experimentSha256 = experimentHash,
                timeSeconds = now, physicsSteps = steps,
                observation = controller.Observe(now),
                policyObservation = executedObservation,
                action = executedAction,
                jointPosition = controller.LastJointPositionRad,
                jointVelocity = controller.LastJointVelocityRadPerSecond,
                passiveWheelVelocity = controller.ActiveRig.ReadPassiveWheelVelocityRadPerSecond(),
                rootPosition = Vector(root.transform.position),
                rootRotation = new[] { rotation.x, rotation.y, rotation.z, rotation.w },
                rootVelocity = Vector(root.velocity),
                rootAngularVelocity = Vector(root.angularVelocity),
                targets = controller.LastTargets,
                ballPosition = controller.SkillBall == null ? new float[3] : Vector(controller.SkillBall.Position),
                upright = Vector3.Dot(root.transform.up, Vector3.up),
                healthy = controller.IsHealthy, fault = controller.Fault,
            };
        }

        private static float[] Vector(Vector3 value) => new[] { value.x, value.y, value.z };
        public void Dispose() { Physics.simulationMode = priorSimulation; }
    }
}
