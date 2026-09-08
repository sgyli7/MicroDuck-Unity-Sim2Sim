using System;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace AgenticRobot.MicroDuck
{
    [Serializable]
    public sealed class PhysXStepResult
    {
        public string engine = "PhysX";
        public string policy;
        public float timeSeconds;
        public int physicsSteps;
        public float[] observation;
        public float[] action;
        public float[] jointPosition;
        public float[] rootPosition;
        public float[] rootRotation;
        public float[] rootVelocity;
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
            // Rebuild articulation transforms after explicit reset, before observations.
            scene.Simulate(0.000001f);
            return Capture();
        }

        public PhysXStepResult Step(float[] action = null)
        {
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
            if (controller.ActivePolicySlot == 3)
            {
                if (!sitTriggered && now >= 1f) { controller.TriggerSkill(now); sitTriggered = true; }
                if (!standTriggered && now >= 4.5f) { controller.TriggerSkill(now); standTriggered = true; }
            }
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
            var root = controller.ActiveRig.RootBody;
            var rotation = root.transform.rotation;
            return new PhysXStepResult
            {
                policy = controller.ActivePolicyName,
                timeSeconds = now, physicsSteps = steps,
                observation = controller.Observe(now),
                action = controller.LastRawAction,
                jointPosition = controller.LastJointPositionRad,
                rootPosition = Vector(root.transform.position),
                rootRotation = new[] { rotation.x, rotation.y, rotation.z, rotation.w },
                rootVelocity = Vector(root.velocity),
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
