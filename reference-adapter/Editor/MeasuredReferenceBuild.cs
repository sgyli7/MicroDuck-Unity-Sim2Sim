using System;
using System.IO;
using UnityEditor;
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
    }
}
