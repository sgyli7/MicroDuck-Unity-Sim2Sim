using System;
using System.Reflection;
using System.Security.Cryptography;
using AgenticRobot.MicroDuck;
using AgenticRobot.MicroDuck.Mujoco;
using UnityEngine;

namespace AgenticRobot.Reference
{
    [Serializable] public sealed class ReferenceModelRecord
    {
        public int slot;
        public string sourceFile;
        public string sourceSha256;
        public string convertedOnnxSha256;
        public string graphSha256;
        public bool graphVerified;
    }
    [Serializable] public sealed class ReferenceModelRecords { public ReferenceModelRecord[] models; }
    public static class ReferenceModelIdentity
    {
        public static string Hash(byte[] bytes)
        {
            using (var sha = SHA256.Create())
                return BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
        }
        public static PolicyModelBinding[] Bindings(MujocoDemoController controller)
        {
            return (PolicyModelBinding[])typeof(MujocoDemoController)
                .GetField("policies", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(controller);
        }
        public static ReferenceModelRecord[] Capture(MujocoDemoController controller)
        {
            var asset = Resources.Load<TextAsset>("ReferenceModelProvenance");
            if (asset == null) return Array.Empty<ReferenceModelRecord>();
            var records = JsonUtility.FromJson<ReferenceModelRecords>(asset.text).models;
            var bindings = Bindings(controller);
            foreach (var item in records)
            {
                var binding = Array.Find(bindings, value => value.slot == item.slot);
                string actual = binding?.model?.modelData?.Value == null ? string.Empty : Hash(binding.model.modelData.Value);
                item.graphVerified = actual.Length == 64 && actual == item.graphSha256;
                item.graphSha256 = actual;
            }
            return records;
        }
    }
}
