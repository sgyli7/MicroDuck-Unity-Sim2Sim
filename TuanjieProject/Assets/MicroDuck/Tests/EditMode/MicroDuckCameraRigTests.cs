using NUnit.Framework;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class MicroDuckCameraRigTests
    {
        private GameObject cameraObject;
        private MicroDuckCameraRig rig;

        [SetUp]
        public void SetUp()
        {
            cameraObject = new GameObject("Camera Rig Test");
            cameraObject.AddComponent<Camera>();
            rig = cameraObject.AddComponent<MicroDuckCameraRig>();
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(cameraObject);
        }

        [Test]
        public void DefaultsToFollowSidePresetWithoutOwningRobotNavigation()
        {
            Assert.That(rig.Mode, Is.EqualTo(MicroDuckCameraMode.Follow));
            Assert.That(rig.Preset, Is.EqualTo(MicroDuckCameraPreset.Side));
            Assert.That(rig.OwnsNavigationInput, Is.False);
        }

        [Test]
        public void FreeFlyModeOwnsNavigationUntilFocusReturnsToFollow()
        {
            rig.ToggleMode();
            Assert.That(rig.Mode, Is.EqualTo(MicroDuckCameraMode.FreeFly));
            Assert.That(rig.OwnsNavigationInput, Is.True);

            rig.FocusOnRobot();
            Assert.That(rig.Mode, Is.EqualTo(MicroDuckCameraMode.Follow));
            Assert.That(rig.OwnsNavigationInput, Is.False);
        }

        [Test]
        public void CyclesAllFourDirectorPresetsInStableOrder()
        {
            Assert.That(rig.CyclePreset(), Is.EqualTo(MicroDuckCameraPreset.Rear));
            Assert.That(rig.CyclePreset(), Is.EqualTo(MicroDuckCameraPreset.Top));
            Assert.That(rig.CyclePreset(), Is.EqualTo(MicroDuckCameraPreset.Showcase));
            Assert.That(rig.CyclePreset(), Is.EqualTo(MicroDuckCameraPreset.Side));
        }

        [Test]
        public void OrbitAndZoomRemainInsideUsableLimits()
        {
            rig.ApplyOrbitInput(new Vector2(10000f, -10000f));
            rig.ApplyZoomInput(10000f);
            Assert.That(rig.PitchDegrees, Is.InRange(8f, 82f));
            Assert.That(rig.DistanceMeters, Is.InRange(0.65f, 7f));

            rig.ApplyOrbitInput(new Vector2(-10000f, 10000f));
            rig.ApplyZoomInput(-10000f);
            Assert.That(rig.PitchDegrees, Is.InRange(8f, 82f));
            Assert.That(rig.DistanceMeters, Is.InRange(0.65f, 7f));
        }
    }
}
