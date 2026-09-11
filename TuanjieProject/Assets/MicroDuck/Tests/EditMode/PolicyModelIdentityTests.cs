using NUnit.Framework;
using Unity.Barracuda;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class PolicyModelIdentityTests
    {
        [Test]
        public void IdentityHashesActualBoundGraphAndRejectsChangedOrMissingProvenance()
        {
            var model = ScriptableObject.CreateInstance<NNModel>();
            var data = ScriptableObject.CreateInstance<NNModelData>();
            model.modelData = data;
            data.Value = new byte[] { 1, 2, 3 };
            try
            {
                var binding = new PolicyModelBinding { slot = 2, model = model,
                    sourceSha256 = new string('a', 64), convertedOnnxSha256 = new string('b', 64),
                    expectedGraphSha256 = PolicyModelIdentity.Hash(data.Value) };
                var original = PolicyModelIdentity.Capture(binding);
                Assert.That(original.graphVerified, Is.True);
                Assert.That(original.sourceFile, Is.EqualTo("alpha_stand.onnx"));
                data.Value[0] = 4;
                var changed = PolicyModelIdentity.Capture(binding);
                Assert.That(changed.graphVerified, Is.False);
                Assert.That(changed.graphSha256, Is.Not.EqualTo(original.graphSha256));
                binding.expectedGraphSha256 = changed.graphSha256;
                binding.sourceSha256 = "not-a-source-hash";
                Assert.That(PolicyModelIdentity.Capture(binding).graphVerified, Is.False);
                binding.model = null;
                Assert.That(PolicyModelIdentity.Capture(binding).graphVerified, Is.False);
            }
            finally { Object.DestroyImmediate(model); Object.DestroyImmediate(data); }
        }
    }
}
