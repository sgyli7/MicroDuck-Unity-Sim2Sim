using System.IO;
using SaiAgent001;
using Unity.Barracuda;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace SaiAgent001.Editor
{
    public static class SaiDemoBuilder
    {
        [MenuItem("SaiAgent001/Build Demo")]
        public static void Build()
        {BuildScene("locomotion-articulated.xml","SaiDemo");}
        [MenuItem("SaiAgent001/Build Stairs/20 mm up")]
        public static void Stairs20Up(){BuildScene("stairs-20-up.xml","SaiStairs20Up");}
        [MenuItem("SaiAgent001/Build Stairs/20 mm down")]
        public static void Stairs20Down(){BuildScene("stairs-20-down.xml","SaiStairs20Down");}
        [MenuItem("SaiAgent001/Build Stairs/40 mm up")]
        public static void Stairs40Up(){BuildScene("stairs-40-up.xml","SaiStairs40Up");}
        [MenuItem("SaiAgent001/Build Stairs/40 mm down")]
        public static void Stairs40Down(){BuildScene("stairs-40-down.xml","SaiStairs40Down");}
        [MenuItem("SaiAgent001/Build Windows Player")]
        public static void BuildWindows64()
        {
            if(!BuildScene("locomotion-articulated.xml","SaiDemo"))return;
            string root=Path.GetFullPath(Path.Combine(Application.dataPath,"../.."));
            string destination=Path.Combine(root,"Builds/SaiWindows64/Sai_Agent_001.exe");
            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            var result=BuildPipeline.BuildPlayer(new BuildPlayerOptions{
                scenes=new[]{"Assets/SaiAgent001/Generated/SaiDemo.unity"},
                locationPathName=destination,target=BuildTarget.StandaloneWindows64,
                options=BuildOptions.None});
            if(result.summary.result!=BuildResult.Succeeded)
                throw new IOException("Sai Windows build failed: "+result.summary.result+"; errors="+result.summary.totalErrors);
            string evidence=Path.Combine(root,"artifacts/sai-editor-test");
            Directory.CreateDirectory(evidence);
            File.WriteAllText(Path.Combine(evidence,"windows-build.completed"),System.DateTime.UtcNow.ToString("o"));
            Debug.Log("Sai Windows Player built: "+destination);
        }
        private static bool BuildScene(string modelName,string sceneName)
        {
            string model=Path.Combine(Application.streamingAssetsPath,"SaiAgent001/models/full/"+modelName);
            string actor="Assets/SaiAgent001/Generated/flat-v1.onnx";
            string stairActor="Assets/SaiAgent001/Generated/stairs-dev40.onnx";
            if(!File.Exists(model)||!File.Exists(actor)||!File.Exists(stairActor))
                throw new FileNotFoundException("Run python scripts/setup-sai-agent.py from the repository root first.");
            AssetDatabase.Refresh();
            var policy=AssetDatabase.LoadAssetAtPath<NNModel>(actor);
            var stairPolicy=AssetDatabase.LoadAssetAtPath<NNModel>(stairActor);
            if(policy==null || stairPolicy==null)throw new IOException("Barracuda has not imported the Sai ONNX assets");
            if(!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())return false;
            EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects,NewSceneMode.Single);
            var controller=new GameObject("Sai_Agent_001").AddComponent<SaiNativeDemo>();
            controller.Policy=policy;
            controller.StairPolicy=stairPolicy;
            controller.ModelRelativePath="SaiAgent001/models/full/"+modelName;
            Directory.CreateDirectory("Assets/SaiAgent001/Generated");
            if(!EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(),"Assets/SaiAgent001/Generated/"+sceneName+".unity"))
                throw new IOException("Could not save Sai scene "+sceneName);
            Selection.activeGameObject=controller.gameObject;
            return true;
        }
    }
}
