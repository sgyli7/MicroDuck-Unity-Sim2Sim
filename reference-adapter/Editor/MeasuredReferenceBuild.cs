using System;
using System.IO;
using System.Linq;
using AgenticRobot.MicroDuck.Mujoco;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace AgenticRobot.Reference
{
    public static class MeasuredReferenceBuild
    {
        public static void Build()
        {
            string root = Path.GetFullPath(Path.Combine(Application.dataPath, "../.."));
            string output = Path.Combine(root, "Builds/MeasuredReference/AgenticRobotGame-MuJoCoReference.exe");
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            PrepareIdentity();
            // Deliberately build existing frozen scene/assets; never invoke regeneration.
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { "Assets/MicroDuck/Generated/Scenes/MicroDuckNativeMvp.unity" },
                locationPathName = output,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None,
            });
            if (report.summary.result != BuildResult.Succeeded)
                throw new InvalidOperationException("Measured reference build failed: " + report.summary.result);
        }

        [Serializable] private sealed class AssetReport { public PolicySource[] policies; }
        [Serializable] private sealed class PolicySource
        {
            public string name;
            public string sourceSha256;
            public string barracudaSha256;
            public string originalAsset;
            public string barracudaAsset;
        }
        private static void PrepareIdentity()
        {
            EditorSceneManager.OpenScene("Assets/MicroDuck/Generated/Scenes/MicroDuckNativeMvp.unity");
            var controller = UnityEngine.Object.FindObjectOfType<MujocoDemoController>();
            var source = JsonUtility.FromJson<AssetReport>(File.ReadAllText("Assets/MicroDuck/Generated/asset-report.json"));
            var records = ReferenceModelIdentity.Bindings(controller).Select(binding =>
            {
                string path = AssetDatabase.GetAssetPath(binding.model);
                var item = source.policies.Single(policy => policy.barracudaAsset == path);
                if (ReferenceModelIdentity.Hash(File.ReadAllBytes(item.originalAsset)) != item.sourceSha256
                    || ReferenceModelIdentity.Hash(File.ReadAllBytes(path)) != item.barracudaSha256)
                    throw new InvalidOperationException("Frozen reference source differs from conversion provenance");
                AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate | ImportAssetOptions.ForceSynchronousImport);
                return new ReferenceModelRecord { slot = binding.slot, sourceFile = item.name,
                    sourceSha256 = item.sourceSha256, convertedOnnxSha256 = item.barracudaSha256,
                    graphSha256 = ReferenceModelIdentity.Hash(binding.model.modelData.Value) };
            }).ToArray();
            string directory = "Assets/ReferenceAdapter/Resources";
            Directory.CreateDirectory(directory);
            File.WriteAllText(directory + "/ReferenceModelProvenance.json",
                JsonUtility.ToJson(new ReferenceModelRecords { models = records }));
            AssetDatabase.Refresh();
            // No frozen scene, model, or control file is saved or regenerated.
        }
    }
}
