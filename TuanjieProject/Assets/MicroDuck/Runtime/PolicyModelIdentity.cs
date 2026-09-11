using System;
using System.Security.Cryptography;

namespace AgenticRobot.MicroDuck
{
    [Serializable]
    public sealed class PolicyModelIdentity
    {
        public int slot;
        public string sourceFile;
        public string sourceSha256;
        public string convertedOnnxSha256;
        public string graphSha256;
        public bool graphVerified;
        public string method = "build-attested-source-conversion-and-runtime-barracuda-graph";

        public static string Hash(byte[] bytes)
        {
            if (bytes == null || bytes.Length == 0) return string.Empty;
            using (var sha = SHA256.Create())
                return BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
        }

        private static bool IsHash(string value)
        {
            if (value == null || value.Length != 64) return false;
            foreach (char digit in value)
                if (!(digit >= '0' && digit <= '9') && !(digit >= 'a' && digit <= 'f')) return false;
            return true;
        }

        public static PolicyModelIdentity Capture(PolicyModelBinding binding)
        {
            if (binding == null) throw new ArgumentNullException(nameof(binding));
            string actual = Hash(binding.model != null && binding.model.modelData != null
                ? binding.model.modelData.Value : null);
            return new PolicyModelIdentity
            {
                slot = binding.slot,
                sourceFile = PolicyCatalog.GetBySlot(binding.slot).FileName,
                sourceSha256 = binding.sourceSha256,
                convertedOnnxSha256 = binding.convertedOnnxSha256,
                graphSha256 = actual,
                graphVerified = IsHash(actual) && actual == binding.expectedGraphSha256
                    && IsHash(binding.sourceSha256) && IsHash(binding.convertedOnnxSha256),
            };
        }
    }
}
