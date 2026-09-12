using System.IO;
using SaiAgent001;
using Unity.Barracuda;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace SaiAgent001.Editor
{
    public static class SaiDemoBuilder
    {
        [MenuItem("SaiAgent001/Build Demo")]
        public static void Build()
        {
            string model=Path.Combine(Application.streamingAssetsPath,"SaiAgent001/models/full/locomotion-articulated.xml");
            string actor="Assets/SaiAgent001/Generated/flat-v1.onnx";
            if(!File.Exists(model)||!File.Exists(actor))
                throw new FileNotFoundException("Run python scripts/setup-sai-agent.py from the repository root first.");
            AssetDatabase.Refresh();
            var policy=AssetDatabase.LoadAssetAtPath<NNModel>(actor);
            if(policy==null)throw new IOException("Barracuda has not imported the Sai ONNX asset");
            if(!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())return;
            EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects,NewSceneMode.Single);
            var controller=new GameObject("Sai_Agent_001").AddComponent<SaiNativeDemo>();
            controller.Policy=policy;
            Directory.CreateDirectory("Assets/SaiAgent001/Generated");
            EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(),"Assets/SaiAgent001/Generated/SaiDemo.unity");
            Selection.activeGameObject=controller.gameObject;
        }
    }
}
