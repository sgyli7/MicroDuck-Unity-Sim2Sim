using System;
using System.Collections;
using System.Linq;
using AgenticRobot.MicroDuck.Mujoco;
using Mujoco;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;
#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed unsafe class MujocoTerrainPlayModeTests
    {
        private const string NativeSceneAssetPath =
            "Assets/MicroDuck/Generated/Scenes/MicroDuckNativeMvp.unity";

        [UnityTest]
        [Timeout(120000)]
        public IEnumerator EveryTerrainHasNativeCollisionAndAWorkingReset()
        {
#if UNITY_EDITOR
            yield return EditorSceneManager.LoadSceneAsyncInPlayMode(
                NativeSceneAssetPath,
                new LoadSceneParameters(LoadSceneMode.Single));
#else
            yield return SceneManager.LoadSceneAsync(0, LoadSceneMode.Single);
#endif
            yield return null;

            MjScene scene = UnityEngine.Object.FindObjectOfType<MjScene>();
            MujocoDemoController controller =
                UnityEngine.Object.FindObjectOfType<MujocoDemoController>();
            MujocoTerrainNavigator navigator =
                UnityEngine.Object.FindObjectOfType<MujocoTerrainNavigator>();
            Assert.That(scene, Is.Not.Null);
            Assert.That(controller, Is.Not.Null);
            Assert.That(navigator, Is.Not.Null);
            yield return WaitForHealthy(controller);

            foreach (TerrainModuleDefinition module in MujocoTerrainCatalog.Modules)
            {
                TerrainPrimitiveDefinition first = module.Primitives[0];
                TerrainPrimitiveDefinition last = module.Primitives[module.Primitives.Count - 1];
                AssertNativeGeom(scene, module.Id, first.Id);
                AssertNativeGeom(scene, module.Id, last.Id);

                Assert.That(navigator.Select(module.Id), Is.True, module.Id);
                Vector3 actual = ReadRootPosition(scene);
                Assert.That(actual.x, Is.EqualTo(module.SpawnPosition.x).Within(1e-4f), module.Id);
                Assert.That(actual.y, Is.EqualTo(module.SpawnPosition.y).Within(1e-4f), module.Id);
                Assert.That(actual.z, Is.EqualTo(module.SpawnPosition.z).Within(1e-4f), module.Id);
                Assert.That(ReadMaximumAbsoluteVelocity(scene), Is.LessThan(1e-8), module.Id);
            }
        }

        [UnityTest]
        [Timeout(120000)]
        public IEnumerator TerrainWorldSurvivesRobotVariantRebuilds()
        {
#if UNITY_EDITOR
            yield return EditorSceneManager.LoadSceneAsyncInPlayMode(
                NativeSceneAssetPath,
                new LoadSceneParameters(LoadSceneMode.Single));
#else
            yield return SceneManager.LoadSceneAsync(0, LoadSceneMode.Single);
#endif
            yield return null;

            MjScene scene = UnityEngine.Object.FindObjectOfType<MjScene>();
            MujocoDemoController controller =
                UnityEngine.Object.FindObjectOfType<MujocoDemoController>();
            MujocoTerrainNavigator navigator =
                UnityEngine.Object.FindObjectOfType<MujocoTerrainNavigator>();
            yield return WaitForHealthy(controller);

            Assert.That(navigator.Select("upstream_roller_slope"), Is.True);
            Assert.That(controller.SwitchPolicy(7), Is.True, controller.Fault);
            yield return WaitForHealthy(controller);
            Assert.That(controller.ActivePolicySlot, Is.EqualTo(7));
            AssertNativeGeom(scene, "upstream_roller_slope", "roller_ramp_20deg");
            Assert.That(
                controller.RootPositionMeters.y,
                Is.EqualTo(navigator.ActiveModule.SpawnPosition.y + 0.0135f).Within(0.01f));

            Assert.That(controller.SwitchPolicy(2), Is.True, controller.Fault);
            yield return WaitForHealthy(controller);
            Assert.That(controller.ActivePolicySlot, Is.EqualTo(2));
            AssertNativeGeom(scene, "upstream_random_grid", "grid_16_16");
            Assert.That(
                controller.RootPositionMeters.y,
                Is.EqualTo(navigator.ActiveModule.SpawnPosition.y).Within(0.01f));
        }

        private static IEnumerator WaitForHealthy(MujocoDemoController controller)
        {
            Assert.That(controller, Is.Not.Null);
            for (int frame = 0; frame < 900; frame++)
            {
                if (controller.IsHealthy && controller.PolicyTicks > 0)
                {
                    yield break;
                }
                yield return new WaitForFixedUpdate();
            }
            Assert.Fail("Native controller did not become healthy: " + controller.Fault);
        }

        private static void AssertNativeGeom(
            MjScene scene,
            string moduleId,
            string primitiveId)
        {
            string name = $"terrain_{moduleId}_{primitiveId}";
            int id = MujocoLib.mj_name2id(
                scene.Model,
                (int)MujocoLib.mjtObj.mjOBJ_GEOM,
                name);
            Assert.That(id, Is.GreaterThanOrEqualTo(0), name);
        }

        private static Vector3 ReadRootPosition(MjScene scene)
        {
            int jointId = MujocoLib.mj_name2id(
                scene.Model,
                (int)MujocoLib.mjtObj.mjOBJ_JOINT,
                "trunk_base_freejoint");
            int address = scene.Model->jnt_qposadr[jointId];
            return MjEngineTool.UnityVector3(scene.Data->qpos + address);
        }

        private static double ReadMaximumAbsoluteVelocity(MjScene scene)
        {
            double maximum = 0.0;
            int velocityCount = checked((int)scene.Model->nv);
            for (int index = 0; index < velocityCount; index++)
            {
                maximum = Math.Max(maximum, Math.Abs(scene.Data->qvel[index]));
            }
            return maximum;
        }
    }
}
