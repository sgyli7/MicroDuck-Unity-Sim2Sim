using System;
using System.Text;
using Mujoco;

namespace SaiAgent001
{
    // This exact class is used by the Unity component and native .NET acceptance.
    // It advances MuJoCo, never Unity rigid bodies or prescribed chassis poses.
    public sealed unsafe class SaiNativeWorld : IDisposable
    {
        public MujocoLib.mjModel_* Model {get;private set;}
        public MujocoLib.mjData_* Data {get;private set;}
        public float[] Observation {get;private set;}=new float[82];
        public float[] Action {get;private set;}=new float[16];
        public double[] Q {get;private set;}=new double[23];
        public double[] V {get;private set;}=new double[22];
        public double Crouch {get;private set;}
        public bool OnStairs {get;private set;}
        public bool HeadingControl=true;
        private readonly int[] qa=new int[16],va=new int[16],aa=new int[16];
        private readonly int[] hq=new int[7],hv=new int[7],ha=new int[7],wheels=new int[4];
        private readonly SaiHeadingHold heading=new SaiHeadingHold();
        private int substeps;

        public SaiNativeWorld(string modelPath)
        {
            // Check ABI before dereferencing native structs from this generated binding.
            int version=MujocoLib.mj_version();
            if(version!=MujocoLib.mjVERSION_HEADER)
                throw new InvalidOperationException("MuJoCo library/binding ABI mismatch: "+version+" / "+MujocoLib.mjVERSION_HEADER);
            var error=new StringBuilder(2048);
            Model=MujocoLib.mj_loadXML(modelPath,null,error,2048);
            if(Model==null)throw new InvalidOperationException("Sai model load failed: "+error);
            try
            {
                Data=MujocoLib.mj_makeData(Model);
                if(Data==null)throw new InvalidOperationException("Sai native data allocation failed");
                if(Model->nu!=23 || Model->neq!=2)throw new InvalidOperationException("Sai articulation contract mismatch");
                for(int i=0;i<16;i++)Map(SaiContract.Legs[i/4]+"_"+SaiContract.Axes[i%4],out qa[i],out va[i],out aa[i]);
                for(int i=0;i<7;i++)Map(i<6?"so101_"+SaiContract.Arm[i]:"cargo_drive",out hq[i],out hv[i],out ha[i]);
                for(int i=0;i<4;i++)wheels[i]=Id(MujocoLib.mjtObj.mjOBJ_BODY,SaiContract.Legs[i]+"_wheel");
                substeps=(int)Math.Round(.02/Model->opt.timestep);
                if(substeps<1 || Math.Abs(substeps*Model->opt.timestep-.02)>1e-8)
                    throw new InvalidOperationException("Native timestep must divide 20 ms exactly");
                Reset();
            }
            catch{Dispose();throw;}
        }

        private int Id(MujocoLib.mjtObj kind,string name)
        {
            int id=MujocoLib.mj_name2id(Model,(int)kind,name);
            if(id<0)throw new InvalidOperationException("Missing Sai component: "+name);
            return id;
        }
        private void Map(string name,out int q,out int v,out int actuator)
        {
            int j=Id(MujocoLib.mjtObj.mjOBJ_JOINT,name);q=Model->jnt_qposadr[j];v=Model->jnt_dofadr[j];actuator=-1;
            for(int i=0;i<checked((int)Model->nu);i++)if(Model->actuator_trnid[2*i]==j){actuator=i;break;}
            if(actuator<0)throw new InvalidOperationException("Missing actuator for "+name);
        }
        public void Reset()
        {
            MujocoLib.mj_resetData(Model,Data);MujocoLib.mj_forward(Model,Data);
            Action=new float[16];Observation=new float[82];Crouch=0;OnStairs=false;heading.Reset();ReadState();
        }
        private void ReadState()
        {
            for(int i=0;i<7;i++)Q[i]=Data->qpos[i];
            for(int i=0;i<6;i++)V[i]=Data->qvel[i];
            for(int i=0;i<16;i++){Q[7+i]=Data->qpos[qa[i]];V[6+i]=Data->qvel[va[i]];}
        }

        public double[] HeightScan()
        {
            // Group 5 is reserved for actual terrain geometry by the setup script.
            // Rays cannot hit the robot's arm, cargo or display meshes.
            var heights=new double[24];double[] xs={-.36,-.18,0,.18,.36,.54,.72,.9},ys={-.24,0,.24};
            double w=Q[3],x=Q[4],y=Q[5],z=Q[6];
            double yaw=Math.Atan2(2*(w*z+x*y),1-2*(y*y+z*z)),c=Math.Cos(yaw),s=Math.Sin(yaw);
            byte* groups=stackalloc byte[6];for(int i=0;i<6;i++)groups[i]=(byte)(i==5?1:0);
            double* origin=stackalloc double[3];double* ray=stackalloc double[3];ray[0]=0;ray[1]=0;ray[2]=-1;
            int index=0,hit=-1;
            foreach(double sx in xs)foreach(double sy in ys)
            {
                origin[0]=Q[0]+c*sx-s*sy;origin[1]=Q[1]+s*sx+c*sy;origin[2]=Q[2]+1;
                double distance=MujocoLib.mj_ray(Model,Data,origin,ray,groups,1,-1,&hit,null);
                if(distance<0)throw new InvalidOperationException("Terrain ray missed; no invented height substituted");
                heights[index++]=origin[2]-distance;
            }
            return heights;
        }

        public void Step(double vx,double wz,double requestedCrouch,Func<float[],bool,float[]> infer)
        {
            if(double.IsNaN(vx+wz+requestedCrouch)||double.IsInfinity(vx+wz+requestedCrouch))throw new ArithmeticException("Non-finite command");
            ReadState();var heights=HeightScan();OnStairs=SaiContract.UseStairs(vx,heights);
            if(OnStairs)vx=Math.Min(vx,.12);
            Crouch+=SaiContract.Clamp(SaiContract.Clamp(requestedCrouch,0,1)-Crouch,-.04,.04);
            Observation=SaiContract.Observe(Q,V,vx,wz,Crouch,Action,Data->time,heights,OnStairs?3.2:2.4);
            Action=infer(Observation,OnStairs);
            var target=OnStairs?SaiContract.StairTargets(Action,vx,wz,Crouch,Data->time,heights):SaiContract.Targets(Action,vx,wz,Crouch);
            double yaw=Math.Atan2(2*(Q[3]*Q[6]+Q[4]*Q[5]),1-2*(Q[5]*Q[5]+Q[6]*Q[6]));
            if(HeadingControl)heading.Apply(target,vx,wz,yaw,V[5]);
            for(int s=0;s<substeps;s++)
            {
                for(int i=0;i<16;i++)Data->ctrl[aa[i]]=i%4==3
                    ?SaiContract.Clamp(.4*(target[i]-Data->qvel[va[i]]),-1.3,1.3)
                    :SaiContract.Clamp(80*(target[i]-Data->qpos[qa[i]])-2*Data->qvel[va[i]],-8,8);
                for(int i=0;i<7;i++)
                {
                    double cap=i==5?1.4:2.94;
                    Data->ctrl[ha[i]]=i==6?SaiContract.Clamp(-.25*Data->qpos[hq[i]]-.015*Data->qvel[hv[i]],-.12,.12)
                        :SaiContract.Clamp(-998.22*Data->qpos[hq[i]]-2.731*Data->qvel[hv[i]]+Data->qfrc_bias[hv[i]],-cap,cap);
                }
                MujocoLib.mj_step(Model,Data);
            }
            ReadState();
            foreach(double v in Q)if(double.IsNaN(v)||double.IsInfinity(v))throw new ArithmeticException("Non-finite physics state");
        }

        public double[][] WheelPositions()
        {
            var p=new double[4][];
            for(int i=0;i<4;i++)p[i]=new[]{Data->xpos[3*wheels[i]],Data->xpos[3*wheels[i]+1],Data->xpos[3*wheels[i]+2]};
            return p;
        }
        public int WheelContacts()
        {
            var touching=new bool[4];
            for(int i=0;i<Data->ncon;i++)for(int side=0;side<2;side++)
            {
                int body=Model->geom_bodyid[Data->contact[i].geom[side]];
                for(int j=0;j<4;j++)if(body==wheels[j])touching[j]=true;
            }
            int count=0;foreach(bool value in touching)if(value)count++;return count;
        }
        public void Dispose()
        {
            if(Data!=null){MujocoLib.mj_deleteData(Data);Data=null;}
            if(Model!=null){MujocoLib.mj_deleteModel(Model);Model=null;}
        }
    }
}
