using System;
using System.Collections;
using System.Linq;
using AgenticRobot.MicroDuck.Mujoco;
using Mujoco;
using NUnit.Framework;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class ReferenceMeasurementPlayModeTests
    {
        [Test]
        public void BatchDriverExistsAsASeparateReferenceOnlyComponent()
        {
            var type = AppDomain.CurrentDomain.GetAssemblies()
                .Select(a => a.GetType("AgenticRobot.Reference.ReferenceBatchDriver"))
                .FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "Missing simulation-time reference experiment runner");
            Assert.That(typeof(MonoBehaviour).IsAssignableFrom(type), Is.True);
        }

        [UnityTest]
        public IEnumerator PassiveMeasurementLeavesFrozenStandDynamicsUnchanged()
        {
            yield return EditorSceneManager.LoadSceneAsyncInPlayMode(
                "Assets/MicroDuck/Generated/Scenes/MicroDuckNativeMvp.unity",
                new LoadSceneParameters(LoadSceneMode.Single));
            var scene = UnityEngine.Object.FindObjectOfType<MjScene>();
            var controller = UnityEngine.Object.FindObjectOfType<MujocoDemoController>();
            for (int i = 0; i < 100 && (!controller.IsHealthy || controller.PolicyTicks < 2); i++)
                yield return new WaitForFixedUpdate();
            Assert.That(controller.IsHealthy, Is.True);
            foreach (var input in UnityEngine.Object.FindObjectsOfType<MujocoKeyboardPolicyInput>())
                input.enabled = false;
            controller.ResetActiveRobot();
            while (TimeOf(scene) < 0.2 - 1e-9) yield return new WaitForFixedUpdate();
            double[] before = Positions(scene);
            var type = AppDomain.CurrentDomain.GetAssemblies()
                .Select(a => a.GetType("AgenticRobot.Reference.MeasurementProbe"))
                .FirstOrDefault(t => t != null);
            Assert.That(type, Is.Not.Null, "The separately packaged passive measurement adapter is missing");
            var probe = new GameObject("Read-only reference measurement").AddComponent(type);
            controller.ResetActiveRobot();
            while (TimeOf(scene) < 0.2 - 1e-9) yield return new WaitForFixedUpdate();
            double[] after = Positions(scene);
            Assert.That(after, Is.EqualTo(before), "Reading measurements changed frozen native dynamics");
            Assert.That((int)type.GetProperty("SampleCount").GetValue(probe), Is.GreaterThan(0));
            object frame = type.GetMethod("Capture").Invoke(probe, null);
            Type frameType = frame.GetType();
            var quaternion = (double[])frameType.GetField("rootQuaternionWxyz").GetValue(frame);
            double currentUpright = 1 - 2 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2]);
            Assert.That((double)frameType.GetField("upright").GetValue(frame),
                Is.EqualTo(currentUpright).Within(1e-9),
                "Post-step upright must use the same post-step quaternion, not cached mjData.xmat");
            UnityEngine.Object.Destroy(probe.gameObject);
        }

        private static unsafe double TimeOf(MjScene scene) => scene.Data->time;
        private static unsafe double[] Positions(MjScene scene)
        {
            var result = new double[scene.Model->nq];
            for (int i = 0; i < result.Length; i++) result[i] = scene.Data->qpos[i];
            return result;
        }
    }
}
