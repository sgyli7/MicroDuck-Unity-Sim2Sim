using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Mujoco;

namespace SaiAgent001.Editor
{
    // Shared unchanged physical criteria, compiled both by the editor and .NET.
    // Commands here are programmatic; keyboard input/rendering need Play acceptance.
    public static unsafe class SaiPhysicsAcceptance
    {
        [Serializable] public sealed class Check { public string name; public bool passed; }
        [Serializable] public sealed class CaseResult
        {
            public string name;
            public bool passed;
            public Check[] checks;
            public double[] final_xyz;
            public double velocity_mae,yaw_mae,lateral_mae,yaw_change,crouch_drop,recovery;
            public double min_upright,max_lateral,seconds;
        }
        [Serializable] public sealed class Result
        {
            public string suite="shared-CSharp-native-physics-v1";
            public int native_version;
            public bool passed;
            public CaseResult[] cases;
        }
        sealed class Frame
        {
            public double t,yaw,up,minWheelX;
            public double[] xyz;
            public float[] obs;
            public int contacts;
            public Frame(double t,double[] xyz,double yaw,double up,float[] obs,int contacts,double minWheelX)
            {this.t=t;this.xyz=xyz;this.yaw=yaw;this.up=up;this.obs=obs;this.contacts=contacts;this.minWheelX=minWheelX;}
        }
        static Check[] Checks(Dictionary<string,bool> checks)=>checks.Select(k=>new Check{name=k.Key,passed=k.Value}).ToArray();
        static double Mean(IEnumerable<double> values)=>values.Average();
        public static Result Run(string models,Func<float[],bool,float[]> Infer,Action<CaseResult> report=null)
        {
            var cases=new List<CaseResult>();bool passed=true;
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
                var row=new CaseResult{name=name,passed=ok,checks=Checks(checks),velocity_mae=vel,yaw_mae=yawError,lateral_mae=lateral,final_xyz=last.xyz,yaw_change=last.yaw,crouch_drop=drop,recovery=recovery};cases.Add(row);report?.Invoke(row);
            }
            foreach(int mm in new[]{20,40})foreach(bool down in new[]{false,true})
            {
                var row=RunStair(models,mm,down,new SaiStairControl(),30,Infer);
                cases.Add(row);passed&=row.passed;report?.Invoke(row);
            }
            return new Result{native_version=MujocoLib.mj_version(),cases=cases.ToArray(),passed=passed};
        }
        public static Result RunExperimental(string models,Func<float[],string,float[]> infer,Action<CaseResult> report=null)
        {
            var cases=new List<CaseResult>();
            foreach(bool down in new[]{false,true})
            {
                string actor=down?"descent60":"ascent60";
                var row=RunStair(models,60,down,down?SaiStairControl.Descent60():SaiStairControl.Ascent60(),45,
                    (obs,stairs)=>infer(obs,stairs?actor:"flat-v1"));
                cases.Add(row);report?.Invoke(row);
            }
            return new Result{suite="shared-CSharp-experimental-stairs60-v1",native_version=MujocoLib.mj_version(),cases=cases.ToArray(),passed=cases.All(c=>c.passed)};
        }
        static CaseResult RunStair(string models,int mm,bool down,SaiStairControl controls,int seconds,Func<float[],bool,float[]> infer)
        {
            string name=$"stairs-{mm}-{(down?"down":"up")}";
            using var world=new SaiNativeWorld(Path.Combine(models,name+".xml"),controls);
            double cleared=-1,minimumUp=1,maxY=0;var support=new Queue<int>();
            for(int k=0;k<seconds*50;k++)
            {
                world.Step(k>=25 && cleared<0?controls.Speed:0,0,0,infer);
                minimumUp=Math.Min(minimumUp,Upright(world.Q));maxY=Math.Max(maxY,Math.Abs(world.Q[1]));
                support.Enqueue(world.WheelContacts());if(support.Count>50)support.Dequeue();
                if(cleared<0 && world.WheelPositions().Min(p=>p[0])>1.07)cleared=world.Data->time;
                if(Upright(world.Q)<.6 || (cleared>=0 && world.Data->time-cleared>=3))break;
            }
            double finalHeight=.2192+(down?0:4*mm*.001)-.035*world.Crouch;
            var checks=new Dictionary<string,bool>{{"all_wheels_cleared",world.WheelPositions().Min(p=>p[0])>1.04},
                {"stopped_3s",cleared>=0 && world.Data->time-cleared>=3},{"upright",Upright(world.Q)>.9},
                {"lane",maxY<.3},{"height",Math.Abs(world.Q[2]-finalHeight)<.02},
                {"supported",support.Count==50 && support.Count(n=>n==4)>35},{"no_fall",minimumUp>.6}};
            return new CaseResult{name=name,passed=checks.Values.All(x=>x),checks=Checks(checks),final_xyz=world.Q.Take(3).ToArray(),min_upright=minimumUp,max_lateral=maxY,seconds=world.Data->time};
        }
        static double Yaw(double[] q)=>Math.Atan2(2*(q[3]*q[6]+q[4]*q[5]),1-2*(q[5]*q[5]+q[6]*q[6]));
        static double Upright(double[] q)=>1-2*(q[4]*q[4]+q[5]*q[5]);
    }
}
