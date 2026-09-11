using System;
using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    /// <summary>
    /// Joint-space reflected rotor inertia using the PhysX implicit drive:
    /// tau_rotor = -J * (v_next - v_previous) / dt. This augments damping by J/dt
    /// and its velocity target, without adding fictitious link mass or root forces.
    /// Force saturation and solver discretization still require measured comparison.
    /// </summary>
    [DefaultExecutionOrder(-500)]
    [DisallowMultipleComponent]
    public sealed class RotorInertiaDrive : MonoBehaviour
    {
        [SerializeField] private ArticulationBody joint;
        [SerializeField] private float rotorInertia;
        [SerializeField] private float mechanicalDamping;
        // Opt-in experiment only. The generated game keeps this component disabled.
        public bool SplitMotorDiagnostic
        {
            get => splitMotorDiagnostic;
            set
            {
                if (value == splitMotorDiagnostic) return;
                RestoreDrive();
                splitMotorDiagnostic = value;
            }
        }
        private bool splitMotorDiagnostic;
        private bool prepared;
        private ArticulationDrive originalDrive;

        public void Configure(ArticulationBody body, float inertia, float damping)
        {
            if (body == null || inertia < 0f || damping < 0f)
                throw new ArgumentException("Invalid rotor parameters");
            RestoreDrive();
            joint = body;
            rotorInertia = inertia;
            mechanicalDamping = damping;
        }

        public void PrepareStep(float dt)
        {
            if (joint == null || joint.jointVelocity.dofCount != 1) return;
            if (dt <= 0f) throw new ArgumentOutOfRangeException(nameof(dt));
            float inertiaDamping = rotorInertia / dt;
            float damping = mechanicalDamping + inertiaDamping;
            var drive = joint.xDrive;
            if (!prepared)
            {
                originalDrive = drive;
                prepared = true;
            }
            if (SplitMotorDiagnostic)
            {
                // Source motor torque is capped, but reflected inertia and joint
                // damping are passive forces, not part of that actuator force cap.
                float torque = originalDrive.stiffness *
                    (drive.target * Mathf.Deg2Rad - joint.jointPosition[0]);
                joint.jointForce = new ArticulationReducedSpace(
                    Mathf.Clamp(torque, -originalDrive.forceLimit, originalDrive.forceLimit));
                drive.stiffness = 0f;
                drive.forceLimit = float.MaxValue;
            }
            drive.damping = damping;
            drive.targetVelocity = damping > 0f
                ? joint.jointVelocity[0] * Mathf.Rad2Deg * inertiaDamping / damping : 0f;
            joint.xDrive = drive;
        }

        private void FixedUpdate() { PrepareStep(Time.fixedDeltaTime); }

        private void OnDisable() { RestoreDrive(); }

        private void RestoreDrive()
        {
            if (!prepared || joint == null) return;
            var restored = originalDrive;
            restored.target = joint.xDrive.target;
            joint.xDrive = restored;
            if (splitMotorDiagnostic && joint.jointForce.dofCount == 1)
                joint.jointForce = new ArticulationReducedSpace(0f);
            prepared = false;
        }
    }
}
