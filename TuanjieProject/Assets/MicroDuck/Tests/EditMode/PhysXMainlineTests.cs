using System.Linq;
using AgenticRobot.MicroDuck.Editor;
using NUnit.Framework;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class PhysXMainlineTests
    {
        [Test]
        public void DefaultPlayerBuildContainsOnlyThePhysXGameScene()
        {
            var options = DemoSceneBuilder.CreateWindows64BuildOptions();
            Assert.That(options.scenes, Is.EqualTo(new[] { DemoSceneBuilder.SceneAssetPath }),
                "The reference Player is separate; it must never be the default game build.");
        }

        [Test]
        public void AlpineGameSurfacesHaveRealPhysXCollision()
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var world = AlpineEnvironmentBuilder.Build();
            var terrain = world.transform.Find("Terrain");
            Assert.That(terrain.childCount, Is.EqualTo(7));
            foreach (Transform module in terrain)
            {
                Assert.That(module.GetComponentsInChildren<Collider>().Length, Is.GreaterThan(0),
                    module.name + " is visual-only instead of a PhysX terrain");
            }
            Assert.That(RenderSettings.skybox, Is.Not.Null);
            Assert.That(Object.FindObjectsOfType<Light>().Any(x => x.type == LightType.Directional));
        }
    }
}
