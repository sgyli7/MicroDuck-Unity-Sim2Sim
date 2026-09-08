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

        public void Configure(ArticulationBody body, float inertia, float damping)
        {
            if (body == null || inertia < 0f || damping < 0f)
                throw new ArgumentException("Invalid rotor parameters");
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
            drive.damping = damping;
            drive.targetVelocity = damping > 0f
                ? joint.jointVelocity[0] * Mathf.Rad2Deg * inertiaDamping / damping : 0f;
            joint.xDrive = drive;
        }

        private void FixedUpdate() { PrepareStep(Time.fixedDeltaTime); }
    }
}
