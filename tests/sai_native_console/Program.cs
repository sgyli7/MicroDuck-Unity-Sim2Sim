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
    record Frame(double t,double[] xyz,double yaw,double up,float[] obs,int contacts,double minWheelX);
    static double Mean(IEnumerable<double> values)=>values.Average();
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
        var cases=new List<object>();bool passed=true;
        var inputs=new[]{("stop",0.0,0.0,false),("W",.16,0.0,false),("S",-.16,0.0,false),
            ("A",0.0,.45,false),("D",0.0,-.45,false),("WA",.14,.3,false),("shift",0.0,0.0,true),("W_shift",.14,0.0,true)};
        foreach(var (name,vx,wz,shift) in inputs)
        {
            using var world=new SaiNativeWorld(Path.Combine(models,"locomotion-articulated.xml"));
            var frames=new List<Frame>();double unwrapped=0,lastYaw=0;
            for(int k=0;k<600;k++)
            {
                double t=world.Data->time;
                world.Step(t>=1?vx:0,t>=1?wz:0,shift && t>=3 && t<8?1:0,Infer);
                double yaw=Yaw(world.Q);unwrapped+=Math.Atan2(Math.Sin(yaw-lastYaw),Math.Cos(yaw-lastYaw));lastYaw=yaw;
                frames.Add(new Frame(world.Data->time,world.Q.Take(3).ToArray(),unwrapped,Upright(world.Q),(float[])world.Observation.Clone(),world.WheelContacts(),world.WheelPositions().Min(p=>p[0])));
            }
            var last=frames.Last();var steady=frames.Where(f=>f.t>=2).ToArray();
            double vel=Mean(steady.Select(f=>(double)Math.Abs(f.obs[3]-vx))),yawError=Mean(steady.Select(f=>(double)Math.Abs(f.obs[8]-wz)));
            double lateral=Mean(steady.Select(f=>(double)Math.Abs(f.obs[4]))),drop=Mean(frames.Where(f=>f.t>=1 && f.t<3).Select(f=>f.xyz[2]))-Mean(frames.Where(f=>f.t>=5 && f.t<8).Select(f=>f.xyz[2]));
            double recovery=Math.Abs(Mean(frames.Where(f=>f.t>=1 && f.t<3).Select(f=>f.xyz[2]))-Mean(frames.Where(f=>f.t>=9).Select(f=>f.xyz[2])));
            double drift=Math.Sqrt(last.xyz[0]*last.xyz[0]+last.xyz[1]*last.xyz[1]);
            var checks=new Dictionary<string,bool>{{"velocity",vel<.06},{"yaw",yawError<.22},{"lateral",lateral<.04},{"upright",frames.Min(f=>f.up)>.9}};
            if(name=="W" || name=="S"){checks["travel"]=Math.Sign(vx)*last.xyz[0]>.7;checks["heading"]=Math.Abs(last.yaw)<.35;}
            if(name=="A" || name=="D"){checks["turn"]=Math.Sign(wz)*last.yaw>1.5;checks["center"]=drift<.4;}
            if(name=="stop" || name=="shift")checks["parked"]=drift<.12;
            if(shift){checks["crouch"]=drop>.023 && drop<.055;checks["recovery"]=recovery<.012;}
            world.Reset();checks["reset"]=Math.Abs(world.Q[0])<1e-12 && Math.Abs(world.Q[1])<1e-12 && world.Data->time==0 && world.Action.All(x=>x==0) && world.Crouch==0;
            bool ok=checks.Values.All(x=>x);passed&=ok;
            var row=new{name,passed=ok,checks,velocity_mae=vel,yaw_mae=yawError,lateral_mae=lateral,final_xyz=last.xyz,yaw_change=last.yaw,crouch_drop=drop,recovery};cases.Add(row);Console.WriteLine(JsonSerializer.Serialize(row));
        }
        foreach(int mm in new[]{20,40})foreach(bool down in new[]{false,true})
        {
            string name=$"stairs-{mm}-{(down?"down":"up")}";
            using var world=new SaiNativeWorld(Path.Combine(models,name+".xml"));
            double cleared=-1,minimumUp=1,maxY=0;var support=new Queue<int>();
            for(int k=0;k<1500;k++)
            {
                world.Step(k>=25 && cleared<0?.12:0,0,0,Infer);
                minimumUp=Math.Min(minimumUp,Upright(world.Q));maxY=Math.Max(maxY,Math.Abs(world.Q[1]));
                support.Enqueue(world.WheelContacts());if(support.Count>50)support.Dequeue();
                if(cleared<0 && world.WheelPositions().Min(p=>p[0])>1.07)cleared=world.Data->time;
                if(Upright(world.Q)<.6 || (cleared>=0 && world.Data->time-cleared>=3))break;
            }
            double finalHeight=.2192+(down?0:4*mm*.001);
            var checks=new Dictionary<string,bool>{{"all_wheels_cleared",world.WheelPositions().Min(p=>p[0])>1.04},
                {"stopped_3s",cleared>=0 && world.Data->time-cleared>=3},{"upright",Upright(world.Q)>.9},
                {"lane",maxY<.3},{"height",Math.Abs(world.Q[2]-finalHeight)<.02},
                {"supported",support.Count==50 && support.Count(n=>n==4)>35},{"no_fall",minimumUp>.6}};
            bool ok=checks.Values.All(x=>x);passed&=ok;
            var row=new{name,passed=ok,checks,final_xyz=world.Q.Take(3).ToArray(),min_upright=minimumUp,max_lateral=maxY,seconds=world.Data->time};cases.Add(row);Console.WriteLine(JsonSerializer.Serialize(row));
        }
        var result=new{suite="shared-CSharp-native-physics-v1",native_version=MujocoLib.mj_version(),inference="ONNX Runtime 1.24.4; Unity uses Barracuda and remains separately unverified",actual_unity_editor=false,physics_class="SaiNativeWorld (same source as Unity component)",cases,passed};
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath)));
        File.WriteAllText(outPath,JsonSerializer.Serialize(result,new JsonSerializerOptions{WriteIndented=true})+"\n");
        return passed?0:1;
    }
    static double Yaw(double[] q)=>Math.Atan2(2*(q[3]*q[6]+q[4]*q[5]),1-2*(q[5]*q[5]+q[6]*q[6]));
    static double Upright(double[] q)=>1-2*(q[4]*q[4]+q[5]*q[5]);
}
