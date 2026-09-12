using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Mujoco;
using SaiAgent001;

unsafe class Program
{
    sealed class Actor : IDisposable
    {
        readonly InferenceSession session;
        public Actor(string path)
        {
            using var options=new SessionOptions{IntraOpNumThreads=2,InterOpNumThreads=1};
            session=new InferenceSession(path,options);
        }
        public float[] Infer(float[] obs)
        {
            using var output=session.Run(new[]{NamedOnnxValue.CreateFromTensor("obs",new DenseTensor<float>(obs,new[]{1,82}))});
            return output.First().AsEnumerable<float>().ToArray();
        }
        public void Dispose(){session.Dispose();}
    }
    static int Main(string[] args)
    {
        string root=Path.GetFullPath(args.Length>0?args[0]:".");
        string outPath=args.Length>1?args[1]:Path.Combine(root,"artifacts/sai-native-test/result.json");
        using var runtime=JsonDocument.Parse(File.ReadAllText(Path.Combine(root,"artifacts/sai-native-test/runtime.json")));
        string lib=runtime.RootElement.GetProperty("native_library").GetString();
        NativeLibrary.SetDllImportResolver(typeof(MujocoLib).Assembly,(name,assembly,path)=>name=="mujoco"?NativeLibrary.Load(lib):IntPtr.Zero);
        string models=Path.Combine(root,"TuanjieProject/Assets/StreamingAssets/SaiAgent001/models/full");
        string policies=Path.Combine(root,"TuanjieProject/Assets/SaiAgent001/Generated");
        using var flat=new Actor(Path.Combine(policies,"flat-v1.onnx"));
        using var stair=new Actor(Path.Combine(policies,"stairs-dev40.onnx"));
        float[] Infer(float[] obs,bool stairs)=>(stairs?stair:flat).Infer(obs);
        var physical=SaiAgent001.Editor.SaiPhysicsAcceptance.Run(models,Infer,row=>Console.WriteLine(JsonSerializer.Serialize(row,new JsonSerializerOptions{IncludeFields=true})));
        var result=new{suite=physical.suite,native_version=physical.native_version,inference="ONNX Runtime 1.24.4; Unity uses Barracuda and remains separately unverified",actual_unity_editor=false,physics_class="SaiNativeWorld (same source as Unity component)",cases=physical.cases,passed=physical.passed};
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath)));
        File.WriteAllText(outPath,JsonSerializer.Serialize(result,new JsonSerializerOptions{WriteIndented=true,IncludeFields=true})+"\n");
        return physical.passed?0:1;
    }
}
