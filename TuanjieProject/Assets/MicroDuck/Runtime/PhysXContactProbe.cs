using System;
using System.Collections.Generic;
using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    [Serializable]
    public sealed class PhysXContactMeasurement
    {
        public string observer;
        public string collider;
        public string otherCollider;
        public string eventKind;
        public float[] point;
        public float[] normal;
        public float separation;
        // Collision.impulse is the whole pair impulse, repeated for its points.
        // Do not sum this field across points or duplicate body-side callbacks.
        public float[] pairImpulse;
    }

    // Passive per-step callback journal, not a persistent contact-manifold query.
    // Sleeping bodies may not emit Stay. Empty means no callbacks, not no contact.
    public sealed class PhysXContactProbe : MonoBehaviour
    {
        private readonly List<PhysXContactMeasurement> events = new List<PhysXContactMeasurement>();
        public bool Recording { get; set; }
        public void BeginStep() { events.Clear(); }
        public PhysXContactMeasurement[] Snapshot() => events.ToArray();
        private void OnCollisionEnter(Collision collision) { Record(collision, "enter"); }
        private void OnCollisionStay(Collision collision) { Record(collision, "stay"); }
        private void Record(Collision collision, string kind)
        {
            if (!Recording) return;
            for (int i = 0; i < collision.contactCount; i++)
            {
                var contact = collision.GetContact(i);
                events.Add(new PhysXContactMeasurement {
                    observer = Path(transform), collider = Path(contact.thisCollider.transform),
                    otherCollider = Path(contact.otherCollider.transform), eventKind = kind,
                    point = Vector(contact.point), normal = Vector(contact.normal),
                    separation = contact.separation, pairImpulse = Vector(collision.impulse),
                });
            }
        }
        private static string Path(Transform item) => item.parent == null ? item.name : Path(item.parent) + "/" + item.name;
        internal static float[] Vector(Vector3 value) => new[] { value.x, value.y, value.z };
    }

    [Serializable]
    public sealed class PhysXMicrostepResult
    {
        public string engine = "PhysX";
        public int physicsSteps;
        public float timeSeconds;
        public int inferenceSlot;
        public float[] rootPosition;
        public float[] rootRotation;
        public float[] rootVelocity;
        public float[] rootAngularVelocity;
        public float[] jointPosition;
        public float[] jointVelocity;
        public float[] passiveWheelVelocity;
        public float[] mouthTipPosition;
        public bool ballActive;
        public float[] ballPosition;
        public float[] ballVelocity;
        public float upright;
        public PhysXContactMeasurement[] contacts;
    }
}
