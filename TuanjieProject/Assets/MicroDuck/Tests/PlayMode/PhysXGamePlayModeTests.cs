using System.Collections;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class PhysXGamePlayModeTests
    {
        [UnityTest]
        public IEnumerator DenseMeasurementRecordsEveryPhysicsTickWithoutChangingDynamics()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;
            var controller = Object.FindObjectOfType<MicroDuckDemoController>();
            using (var session = new PhysXPolicySession(controller))
            {
                var option = typeof(PhysXPolicySession).GetProperty("RecordPhysicsTrace");
                Assert.That(option, Is.Not.Null, "Missing opt-in dense physics measurement");
                var field = typeof(PhysXStepResult).GetField("physicsTrace");
                Assert.That(field, Is.Not.Null);
                session.Reset(2, true);
                var plain = new System.Collections.Generic.List<PhysXStepResult>();
                for (int i = 0; i < 20; i++) plain.Add(session.Step(new float[14]));
                option.SetValue(session, true);
                session.Reset(2, true);
                int contactEvents = 0;
                for (int i = 0; i < 20; i++)
                {
                    var measured = session.Step(new float[14]);
                    Assert.That(measured.jointPosition, Is.EqualTo(plain[i].jointPosition).Within(1e-6f));
                    Assert.That(measured.rootPosition, Is.EqualTo(plain[i].rootPosition).Within(1e-6f));
                    Assert.That(measured.policyObservation, Is.EqualTo(plain[i].policyObservation).Within(1e-6f));
                    var trace = (System.Array)field.GetValue(measured);
                    Assert.That(trace.Length, Is.EqualTo(4));
                    for (int j = 0; j < 4; j++)
                    {
                        object sample = trace.GetValue(j);
                        var type = sample.GetType();
                        Assert.That(type.GetField("physicsSteps").GetValue(sample), Is.EqualTo(i * 4 + j + 1));
                        contactEvents += ((System.Array)type.GetField("contacts").GetValue(sample)).Length;
                        Assert.That(((float[])type.GetField("jointPosition").GetValue(sample)).Length, Is.EqualTo(14));
                    }
                }
                Assert.That(contactEvents, Is.GreaterThan(0), "Actual ground contact callbacks were not recorded");
            }
        }

        [UnityTest]
        public IEnumerator ExternalActorSeesTheSameScheduledCommandAsTheGameControlLoop()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;
            var controller = Object.FindObjectOfType<MicroDuckDemoController>();
            using (var session = new PhysXPolicySession(controller))
            {
                var previous = session.Reset(3, true);
                for (int step = 0; step < 226; step++)
                {
                    var next = session.Step(new float[14]);
                    Assert.That(controller.LastObservation, Is.EqualTo(previous.observation).Within(1e-6f),
                        "External actor was given stale command/state at t=" + previous.timeSeconds);
                    previous = next;
                }
            }
        }

        [UnityTest]
        public IEnumerator ResetPublishesLinkGeometryConsistentWithReducedJointCoordinates()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;
            var controller = Object.FindObjectOfType<MicroDuckDemoController>();
            using (var session = new PhysXPolicySession(controller))
            {
                session.Reset(2, true);
                for (int i = 0; i < 10; i++) session.Step(new float[14]);
                AssertJointGeometry(controller.ActiveRig, "after real simulation");
                session.Reset(2, true);
                AssertJointGeometry(controller.ActiveRig, "after explicit reset");
            }
        }

        private static void AssertJointGeometry(MicroDuckRig rig, string stage)
        {
            foreach (var body in rig.GetComponentsInChildren<ArticulationBody>())
            {
                if (body.isRoot) continue;
                var parent = body.transform.parent.GetComponent<ArticulationBody>();
                Quaternion expectedRotation = parent.transform.rotation * body.parentAnchorRotation
                    * Quaternion.AngleAxis(body.jointPosition[0] * Mathf.Rad2Deg, Vector3.right)
                    * Quaternion.Inverse(body.anchorRotation);
                Vector3 expectedPosition = parent.transform.TransformPoint(body.parentAnchorPosition)
                    - expectedRotation * body.anchorPosition;
                Assert.That(Quaternion.Angle(expectedRotation, body.transform.rotation), Is.LessThan(0.05f),
                    stage + ": joint/geometry rotation mismatch " + body.name);
                Assert.That(Vector3.Distance(expectedPosition, body.transform.position), Is.LessThan(0.00001f),
                    stage + ": joint/geometry position mismatch " + body.name);
            }
        }

        [UnityTest]
        public IEnumerator ResetClearsPriorEpisodeDynamicsForIdenticalActions()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;
            var controller = Object.FindObjectOfType<MicroDuckDemoController>();
            using (var session = new PhysXPolicySession(controller))
            {
                session.Reset(2, true);
                var initialAnchors = controller.ActiveRig.GetComponentsInChildren<ArticulationBody>()
                    .ToDictionary(b => b.name, b => b.parentAnchorRotation);
                PhysXStepResult first = null;
                for (int i = 0; i < 10; i++)
                {
                    first = session.Step(new float[14]);
                }
                session.Reset(2, true);
                foreach (var body in controller.ActiveRig.GetComponentsInChildren<ArticulationBody>())
                    Assert.That(Quaternion.Angle(initialAnchors[body.name], body.parentAnchorRotation),
                        Is.LessThan(0.05f), "Reset changed the joint definition: " + body.name);
                PhysXStepResult second = null;
                for (int i = 0; i < 10; i++)
                {
                    second = session.Step(new float[14]);
                }
                Assert.That(second.jointPosition, Is.EqualTo(first.jointPosition).Within(1e-5f),
                    "Identical reset/actions retained dynamics from the previous episode");
                Assert.That(second.rootPosition, Is.EqualTo(first.rootPosition).Within(1e-5f));
            }
        }

        [UnityTest]
        public IEnumerator LoadedGameRetainsConfiguredSolverIterations()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;
            var controller = Object.FindObjectOfType<MicroDuckDemoController>();
            foreach (int slot in new[] { 2, 7 })
            {
                Assert.That(controller.SelectPolicy(slot), Is.True);
                foreach (var body in controller.ActiveRig.GetComponentsInChildren<ArticulationBody>())
                {
                    Assert.That(body.solverIterations, Is.EqualTo(12), body.name);
                    Assert.That(body.solverVelocityIterations, Is.EqualTo(4), body.name);
                }
            }
        }

        [UnityTest]
        public IEnumerator TrainingReplicaMatchesGameDynamicsForTheSameInitialStateAndActions()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;
            var game = Object.FindObjectOfType<MicroDuckDemoController>();
            var copyScene = SceneManager.CreateScene("game sampler parity", new CreateSceneParameters(LocalPhysicsMode.Physics3D));
            var replica = game.CreateReplica(copyScene);
            using (var gameSession = new PhysXPolicySession(game))
            using (var copySession = new PhysXPolicySession(replica))
            {
                gameSession.Reset(2, true);
                copySession.Reset(2, true);
                ComparePhysicalFrames(game.ActiveRig, replica.ActiveRig, "reset");
                float maximumJointError = 0f;
                float maximumRootError = 0f;
                for (int step = 0; step < 10; step++)
                {
                    var gameState = gameSession.Step(new float[14]);
                    var copyState = copySession.Step(new float[14]);
                    if (step == 0) ComparePhysicalFrames(game.ActiveRig, replica.ActiveRig, "step0");
                    for (int j = 0; j < 14; j++)
                        maximumJointError = Mathf.Max(maximumJointError,
                            Mathf.Abs(copyState.jointPosition[j] - gameState.jointPosition[j]));
                    for (int j = 0; j < 3; j++)
                        maximumRootError = Mathf.Max(maximumRootError,
                            Mathf.Abs(copyState.rootPosition[j] - gameState.rootPosition[j]));
                    TestContext.Progress.WriteLine($"PARITY step={step} qMax={maximumJointError:R} rootMax={maximumRootError:R}");
                }
                Assert.That(maximumJointError, Is.LessThan(0.0001f), "Training clone changed joint dynamics");
                Assert.That(maximumRootError, Is.LessThan(0.0001f), "Training clone changed root dynamics");
            }
            yield return SceneManager.UnloadSceneAsync(copyScene);
        }

        private static void ComparePhysicalFrames(MicroDuckRig game, MicroDuckRig copy, string stage)
        {
            var copies = copy.GetComponentsInChildren<ArticulationBody>().ToDictionary(b => b.name);
            foreach (var body in game.GetComponentsInChildren<ArticulationBody>())
            {
                var other = copies[body.name];
                TestContext.Progress.WriteLine($"FRAMES {stage} {body.name} " +
                    $"pos={Vector3.Distance(body.transform.position, other.transform.position):R} " +
                    $"rot={Quaternion.Angle(body.transform.rotation, other.transform.rotation):R} " +
                    $"anchorP={Vector3.Distance(body.parentAnchorPosition, other.parentAnchorPosition):R} " +
                    $"anchorR={Quaternion.Angle(body.parentAnchorRotation, other.parentAnchorRotation):R} " +
                    $"childP={Vector3.Distance(body.anchorPosition, other.anchorPosition):R} " +
                    $"childR={Quaternion.Angle(body.anchorRotation, other.anchorRotation):R} " +
                    $"auto={body.matchAnchors}/{other.matchAnchors} solver={body.solverIterations}/{other.solverIterations}");
            }
            var colliders = copy.GetComponentsInChildren<Collider>().ToDictionary(c => c.name);
            foreach (var collider in game.GetComponentsInChildren<Collider>())
            {
                var other = colliders[collider.name];
                TestContext.Progress.WriteLine($"BOUNDS {stage} {collider.name} " +
                    $"center={Vector3.Distance(collider.bounds.center, other.bounds.center):R} " +
                    $"size={Vector3.Distance(collider.bounds.size, other.bounds.size):R}");
            }
        }

        [UnityTest]
        public IEnumerator ResetObservationRestoresRootAfterActualMotionWithoutAdvancingTime()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;
            var controller = Object.FindObjectOfType<MicroDuckDemoController>();
            using (var session = new PhysXPolicySession(controller))
            {
                session.Reset(2, true);
                Vector3 home = controller.ActiveRig.ResetPosition;
                for (int step = 0; step < 30; step++) session.Step(new float[14]);
                Assert.That(Vector3.Distance(controller.ActiveRig.RootBody.transform.position, home),
                    Is.GreaterThan(0.01f), "Precondition: the body must have physically moved");
                var reset = session.Reset(2, true);
                Assert.That(reset.rootPosition, Is.EqualTo(new[] { home.x, home.y, home.z }),
                    "Reset returned a stale, pre-teleport root transform");
                Assert.That(reset.upright, Is.EqualTo(1f).Within(1e-6f));
                Assert.That(reset.observation[3..6], Is.EqualTo(new[] { 0f, 0f, -1f }).Within(1e-6f));
                Assert.That(reset.observation[20..48], Is.All.EqualTo(0f));
                Assert.That(reset.physicsSteps, Is.Zero);
            }
        }

        [UnityTest]
        public IEnumerator SamplerResetDoesNotInjectContactVelocityAndIgnoresTemplateMotion()
        {
            yield return SceneManager.LoadSceneAsync("MicroDuckMvp", LoadSceneMode.Single);
            yield return null;
            var template = Object.FindObjectOfType<MicroDuckDemoController>();
            template.enabled = false;
            Vector3 canonicalPosition = template.ActiveRig.ResetPosition;
            template.ActiveRig.RootBody.TeleportRoot(new Vector3(2f, 3f, 4f), Quaternion.Euler(20f, 30f, 40f));
            var copyScene = SceneManager.CreateScene("reset invariance", new CreateSceneParameters(LocalPhysicsMode.Physics3D));
            var replica = template.CreateReplica(copyScene);
            using (var session = new PhysXPolicySession(replica))
            {
                var reset = session.Reset(2, true);
                Assert.That(replica.ActiveRig.RootBody.transform.position,
                    Is.EqualTo(canonicalPosition), "Replica inherited a moved display pose");
                reset = session.Reset(7, true);
                foreach (float velocity in reset.observation[20..34])
                    Assert.That(Mathf.Abs(velocity), Is.LessThan(1e-6f),
                        "Reset must not run a tiny contact step that injects impulse velocities");
                Assert.That(reset.timeSeconds, Is.Zero);
            }
            yield return SceneManager.UnloadSceneAsync(copyScene);
        }

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
