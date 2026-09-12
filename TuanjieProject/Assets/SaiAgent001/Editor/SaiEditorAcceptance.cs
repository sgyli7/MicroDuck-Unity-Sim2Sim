using System;
using System.IO;
using Unity.Barracuda;
using UnityEditor;
using UnityEngine;

namespace SaiAgent001.Editor
{
    public static class SaiEditorAcceptance
    {
        [Serializable] public sealed class Report
        {
            public string editor_version, platform, timestamp_utc, error;
            public string inference="Unity Barracuda CSharpBurst";
            public bool actual_unity_editor=true;
            public bool keyboard_input_verified=false, rendering_verified=false, passed=false;
            public SaiPhysicsAcceptance.Result physical;
        }

        // Safe to run in an existing editor: does not replace or save its scene.
        [MenuItem("SaiAgent001/Validate Physics and Barracuda")]
        public static void Run()
        {
            string root=Path.GetFullPath(Path.Combine(Application.dataPath,"../.."));
            string destination=Path.Combine(root,"artifacts/sai-editor-test/result.json");
            var report=new Report{editor_version=Application.unityVersion,
                platform=Application.platform.ToString(),timestamp_utc=DateTime.UtcNow.ToString("o")};
            try
            {
                if(EditorApplication.isPlayingOrWillChangePlaymode)
                    throw new InvalidOperationException("Stop Play mode before running isolated physics acceptance.");
                AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                var flat=Load("flat-v1");
                var stairs=Load("stairs-dev40");
                string models=Path.Combine(Application.streamingAssetsPath,"SaiAgent001/models/full");
                using(var flatActor=new SaiBarracudaActor(flat))
                using(var stairActor=new SaiBarracudaActor(stairs))
                {
                    report.physical=SaiPhysicsAcceptance.Run(models,
                        (obs,onStairs)=>(onStairs?stairActor:flatActor).Infer(obs),
                        row=>Debug.Log("Sai acceptance: "+JsonUtility.ToJson(row)));
                    report.passed=report.physical.passed;
                }
            }
            catch(Exception exception)
            {
                report.error=exception.ToString();
                Debug.LogException(exception);
            }
            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            File.WriteAllText(destination,JsonUtility.ToJson(report,true)+"\n");
            Debug.Log("Sai editor physical acceptance "+(report.passed?"PASSED":"FAILED")+": "+destination);
            if(Application.isBatchMode)EditorApplication.Exit(report.passed?0:1);
        }

        private static NNModel Load(string name)
        {
            string path="Assets/SaiAgent001/Generated/"+name+".onnx";
            if(!File.Exists(path))throw new FileNotFoundException("Run python scripts/setup-sai-agent.py first.",path);
            AssetDatabase.ImportAsset(path,ImportAssetOptions.ForceSynchronousImport);
            var policy=AssetDatabase.LoadAssetAtPath<NNModel>(path);
            if(policy==null)throw new IOException("Barracuda did not import "+path);
            return policy;
        }
    }
}
