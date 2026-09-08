using System.Linq;
using NUnit.Framework;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class TerrainCatalogTests
    {
        [Test]
        public void ContainsTheSevenApprovedModulesInStableOrder()
        {
            Assert.That(
                TerrainCatalog.Modules.Select(module => module.Id),
                Is.EqualTo(new[]
                {
                    "flat_plaza",
                    "upstream_pyramid_stairs",
                    "upstream_random_grid",
                    "upstream_pyramid_slope",
                    "upstream_roller_slope",
                    "rock_steps",
                    "stairs_bridge",
                }));
        }

        [Test]
        public void EveryModuleHasCollisionGeometryAndAUniqueSafeSpawn()
        {
            Assert.That(TerrainCatalog.Modules, Has.All.Matches<TerrainModuleDefinition>(
                module => module.Primitives.Count > 0));
            Assert.That(TerrainCatalog.Modules, Has.All.Matches<TerrainModuleDefinition>(
                module => module.Primitives.All(primitive => primitive.CollisionEnabled)));

            int uniqueSpawns = TerrainCatalog.Modules
                .Select(module => module.SpawnPosition)
                .Distinct()
                .Count();
            Assert.That(uniqueSpawns, Is.EqualTo(TerrainCatalog.Modules.Count));
        }

        [Test]
        public void UpstreamRoughGeometryRespectsMicroDuckScaleLimits()
        {
            TerrainModuleDefinition stairs =
                TerrainCatalog.Find("upstream_pyramid_stairs");
            TerrainModuleDefinition grid =
                TerrainCatalog.Find("upstream_random_grid");
            TerrainModuleDefinition slope =
                TerrainCatalog.Find("upstream_pyramid_slope");
            TerrainModuleDefinition roller =
                TerrainCatalog.Find("upstream_roller_slope");

            Assert.That(stairs.MaximumSurfaceVariationMeters, Is.EqualTo(0.015f).Within(1e-6f));
            Assert.That(grid.MaximumSurfaceVariationMeters, Is.EqualTo(0.010f).Within(1e-6f));
            Assert.That(slope.MinimumSlopeDegrees, Is.EqualTo(1.7f).Within(0.1f));
            Assert.That(slope.MaximumSlopeDegrees, Is.EqualTo(5.7f).Within(0.1f));
            Assert.That(roller.MinimumSlopeDegrees, Is.EqualTo(2f).Within(1e-6f));
            Assert.That(roller.MaximumSlopeDegrees, Is.EqualTo(20f).Within(1e-6f));
        }

        [Test]
        public void RandomGridGenerationIsDeterministic()
        {
            TerrainModuleDefinition first = TerrainCatalog.CreateModule(
                "upstream_random_grid");
            TerrainModuleDefinition second = TerrainCatalog.CreateModule(
                "upstream_random_grid");

            Assert.That(first.Primitives.Count, Is.EqualTo(290));
            Assert.That(second.Primitives.Count, Is.EqualTo(first.Primitives.Count));
            for (int index = 0; index < first.Primitives.Count; index++)
            {
                Assert.That(second.Primitives[index], Is.EqualTo(first.Primitives[index]));
            }
        }

        [Test]
        public void CompatibilityLabelsSeparateTrainingMatchesFromExtensions()
        {
            Assert.That(TerrainCatalog.Find("flat_plaza").IsTrainingMatched, Is.True);
            Assert.That(TerrainCatalog.Find("upstream_random_grid").IsTrainingMatched, Is.True);
            Assert.That(TerrainCatalog.Find("upstream_roller_slope").IsTrainingMatched, Is.True);
            Assert.That(TerrainCatalog.Find("rock_steps").IsTrainingMatched, Is.False);
            Assert.That(TerrainCatalog.Find("stairs_bridge").IsTrainingMatched, Is.False);
        }
    }
}
