using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class PhysXGamePlayModeTests
    {
        [UnityTest]
        public IEnumerator ReplicasUseGameFrictionAndIndependentPhysicsScenes()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;
            var template = Object.FindObjectOfType<MicroDuckDemoController>();
            template.enabled = false;
            var sceneA = SceneManager.CreateScene("sampler A", new CreateSceneParameters(LocalPhysicsMode.Physics3D));
            var sceneB = SceneManager.CreateScene("sampler B", new CreateSceneParameters(LocalPhysicsMode.Physics3D));
            var a = template.CreateReplica(sceneA);
            var b = template.CreateReplica(sceneB);
            var floor = GameObject.Find("Floor").GetComponent<Collider>();
            foreach (var root in sceneA.GetRootGameObjects())
                if (root.name == "PhysX Training Floor")
                    Assert.That(root.GetComponent<Collider>().sharedMaterial, Is.SameAs(floor.sharedMaterial));
            using (var first = new PhysXPolicySession(a))
            using (var second = new PhysXPolicySession(b))
            {
                first.Reset(2, true);
                var initialB = second.Reset(2, true);
                var before = b.ActiveRig.RootBody.transform.position;
                for (int i = 0; i < 10; i++) first.Step(new float[14]);
                Assert.That(b.ActiveRig.RootBody.transform.position, Is.EqualTo(before),
                    "Stepping one environment must not advance another physics scene");
                Assert.That(second.Step(new float[14]).physicsSteps, Is.EqualTo(4));
            }
            yield return SceneManager.UnloadSceneAsync(sceneA);
            yield return SceneManager.UnloadSceneAsync(sceneB);
        }

        [UnityTest]
        public IEnumerator ExternalActionsAdvanceRealPhysXAndReturnNextObservation()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;
            var controller = Object.FindObjectOfType<MicroDuckDemoController>();
            using (var session = new PhysXPolicySession(controller))
            {
                var initial = session.Reset(2, externalActions: true);
                var actions = new float[14];
                for (int i = 0; i < actions.Length; i++) actions[i] = 0.1f;
                var next = session.Step(actions);
                Assert.That(next.engine, Is.EqualTo("PhysX"));
                Assert.That(next.physicsSteps, Is.EqualTo(4));
                Assert.That(next.timeSeconds, Is.EqualTo(0.02f).Within(1e-7));
                Assert.That(next.observation.Length, Is.EqualTo(61));
                Assert.That(next.observation[34..48], Is.All.EqualTo(0.1f));
                Assert.That(next.jointPosition, Is.Not.EqualTo(initial.jointPosition),
                    "The action must change actual articulation state, not a replayed observation");
            }
        }

        [UnityTest]
        public IEnumerator SavedGameCanSelectTerrainAndHasNoForeignPhysics()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;
            var navigator = Object.FindObjectOfType<PhysXTerrainNavigator>();
            Assert.That(navigator, Is.Not.Null);
            Assert.That(navigator.Select(1), Is.True, "Saved scene lost its navigator/controller binding");
            Assert.That(navigator.ActiveModule.Id, Is.EqualTo("upstream_pyramid_stairs"));
            Assert.That(PhysicsIdentity.LoadedForeignPhysics(), Is.Empty);
            var controller = Object.FindObjectOfType<MicroDuckDemoController>();
            Vector3 selectedSpawn = controller.ActiveRig.RootBody.transform.position;
            controller.ResetActiveRig();
            Assert.That(Vector3.Distance(controller.ActiveRig.RootBody.transform.position, selectedSpawn),
                Is.LessThan(0.00001f), "Reset left the selected terrain");
        }

        [UnityTest]
        public IEnumerator SameRigKickSwitchesPrepareTheCorrectBallWithoutResettingRobot()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;
            var controller = Object.FindObjectOfType<MicroDuckDemoController>();
            Vector3 robotPosition = controller.ActiveRig.RootBody.transform.position;
            Assert.That(controller.HotSwapPolicy(5), Is.True);
            Assert.That(controller.SkillBall.gameObject.activeSelf, Is.True);
            Vector3 leftBall = controller.SkillBall.Position;
            Assert.That(controller.HotSwapPolicy(6), Is.True);
            Assert.That(Vector3.Distance(leftBall, controller.SkillBall.Position), Is.GreaterThan(0.05f));
            Assert.That(controller.ActiveRig.RootBody.transform.position, Is.EqualTo(robotPosition));
        }
    }
}
