using System;
using System.Collections.Generic;
using System.IO;
using Mujoco;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Rendering;

namespace SaiAgent001
{
    // Owns the original native MJCF directly: no Unity reserialization of
    // full inertias, cargo sliders, contact meshes or belt equalities.
    public sealed unsafe class SaiNativeDemo : MonoBehaviour
    {
        public NNModel Policy;
        public string ModelRelativePath="SaiAgent001/models/full/locomotion-articulated.xml";
        private MujocoLib.mjModel_* model;
        private MujocoLib.mjData_* data;
        private IWorker worker;
        private string inputName,outputName;
        private readonly int[] qa=new int[16],va=new int[16],aa=new int[16];
        private readonly List<int[]> held=new List<int[]>();
        private readonly List<KeyValuePair<int,Transform>> visuals=new List<KeyValuePair<int,Transform>>();
        private readonly List<UnityEngine.Object> resources=new List<UnityEngine.Object>();
        private double[] target=new double[16];
        private float[] previous=new float[16];
        private double vx,wz,crouch,requestedCrouch;
        private float previousFixedDelta;
        private string fault="";
        private Camera view;

        private int Id(MujocoLib.mjtObj kind,string name)
        {
            int id=MujocoLib.mj_name2id(model,(int)kind,name);
            if(id<0)throw new InvalidOperationException("Missing Sai component: "+name);
            return id;
        }
        private int ActuatorFor(int joint)
        {
            for(int i=0;i<model->nu;i++)if(model->actuator_trnid[2*i]==joint)return i;
            throw new InvalidOperationException("Joint lacks actuator");
        }
        private void Start()
        {
            previousFixedDelta=Time.fixedDeltaTime;
            try
            {
                if(Policy==null)throw new InvalidOperationException("Sai ONNX is not assigned; run SaiAgent001/Build Demo");
                model=MjEngineTool.LoadModelFromFile(Path.Combine(Application.streamingAssetsPath,ModelRelativePath));
                if(model==null)throw new InvalidOperationException("Sai MJCF could not load");
                data=MujocoLib.mj_makeData(model);
                if(data==null)throw new InvalidOperationException("Sai native data allocation failed");
                if(model->nu!=23 || model->neq!=2)throw new InvalidOperationException("Sai model contract mismatch");
                for(int i=0;i<16;i++)
                {
                    int j=Id(MujocoLib.mjtObj.mjOBJ_JOINT,SaiContract.Legs[i/4]+"_"+SaiContract.Axes[i%4]);
                    qa[i]=model->jnt_qposadr[j];va[i]=model->jnt_dofadr[j];aa[i]=ActuatorFor(j);
                }
                for(int i=0;i<7;i++)
                {
                    int j=Id(MujocoLib.mjtObj.mjOBJ_JOINT,i<6?"so101_"+SaiContract.Arm[i]:"cargo_drive");
                    held.Add(new[]{model->jnt_qposadr[j],model->jnt_dofadr[j],ActuatorFor(j),i});
                }
                var net=ModelLoader.Load(Policy);
                if(net.inputs.Count!=1 || net.outputs.Count!=1)throw new InvalidOperationException("Sai ONNX needs one input/output");
                inputName=net.inputs[0].name;outputName=net.outputs[0];
                worker=WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpBurst,net);
                Time.fixedDeltaTime=.02f;
                MujocoLib.mj_forward(model,data);
                BuildVisuals();SyncVisuals();
                view=Camera.main;
                if(view==null){var c=new GameObject("Sai follow camera");view=c.AddComponent<Camera>();}
            }
            catch(Exception e){fault=e.Message;Debug.LogException(e);enabled=false;}
        }
        private void Update()
        {
            vx=.16*((Input.GetKey(KeyCode.W)?1:0)-(Input.GetKey(KeyCode.S)?1:0));
            wz=.45*((Input.GetKey(KeyCode.A)?1:0)-(Input.GetKey(KeyCode.D)?1:0));
            requestedCrouch=Input.GetKey(KeyCode.LeftShift)||Input.GetKey(KeyCode.RightShift)?1:0;
            if(Input.GetKeyDown(KeyCode.R) && data!=null)
            {
                MujocoLib.mj_resetData(model,data);MujocoLib.mj_forward(model,data);
                previous=new float[16];crouch=0;
            }
        }
        private void FixedUpdate()
        {
            if(data==null || worker==null || fault!="")return;
            try
            {
                crouch+=SaiContract.Clamp(requestedCrouch-crouch,-.04,.04);
                var q=new double[23];var v=new double[22];
                for(int i=0;i<7;i++)q[i]=data->qpos[i];
                for(int i=0;i<6;i++)v[i]=data->qvel[i];
                for(int i=0;i<16;i++){q[7+i]=data->qpos[qa[i]];v[6+i]=data->qvel[va[i]];}
                // This profile is the validated flat policy. Stair profile adds
                // native terrain queries and its own tested policy contract.
                var obs=SaiContract.Observe(q,v,vx,wz,crouch,previous,data->time,new double[24]);
                var action=new float[16];
                using(var input=new Tensor(1,82,obs,inputName))
                {
                    worker.Execute(input);var result=worker.PeekOutput(outputName);
                    if(result.length!=16)throw new InvalidOperationException("Sai ONNX output is not 16D");
                    for(int i=0;i<16;i++)action[i]=result[i];
                }
                target=SaiContract.Targets(action,vx,wz,crouch);previous=action;
                int substeps=(int)Math.Round(.02/model->opt.timestep);
                if(substeps<1 || Math.Abs(substeps*model->opt.timestep-.02)>1e-8)
                    throw new InvalidOperationException("Native timestep must divide 20 ms exactly");
                for(int s=0;s<substeps;s++)
                {
                    for(int i=0;i<16;i++)data->ctrl[aa[i]]=i%4==3
                        ?SaiContract.Clamp(.4*(target[i]-data->qvel[va[i]]),-1.3,1.3)
                        :SaiContract.Clamp(80*(target[i]-data->qpos[qa[i]])-2*data->qvel[va[i]],-8,8);
                    foreach(var h in held)
                    {
                        int i=h[3];double cap=i==5?1.4:2.94;
                        data->ctrl[h[2]]=i==6?SaiContract.Clamp(-.25*data->qpos[h[0]]-.015*data->qvel[h[1]],-.12,.12)
                            :SaiContract.Clamp(-998.22*data->qpos[h[0]]-2.731*data->qvel[h[1]]+data->qfrc_bias[h[1]],-cap,cap);
                    }
                    MujocoLib.mj_step(model,data);
                }
                SyncVisuals();
            }
            catch(Exception e){fault=e.Message;Debug.LogException(e);}
        }
        private void LateUpdate()
        {
            if(data==null || view==null)return;
            var p=MjEngineTool.UnityVector3(data->qpos);
            view.transform.position=p+new Vector3(.8f,.55f,.8f);
            view.transform.LookAt(p+Vector3.up*.05f);view.fieldOfView=45;
        }
        private void SyncVisuals()
        {
            foreach(var pair in visuals)
            {
                double* mat=data->geom_xmat+9*pair.Key;
                double* quat=stackalloc double[4];MujocoLib.mju_mat2Quat(quat,mat);
                pair.Value.SetPositionAndRotation(MjEngineTool.UnityVector3(data->geom_xpos+3*pair.Key),MjEngineTool.UnityQuaternion(quat));
            }
        }
        private void BuildVisuals()
        {
            for(int i=0;i<model->ngeom;i++)
            {
                float* rgba=model->geom_rgba+4*i;
                if(rgba[3]<.01)continue;
                int meshId=model->geom_dataid[i];GameObject go;
                if(model->geom_type[i]==(int)MujocoLib.mjtGeom.mjGEOM_MESH)
                {
                    var mesh=new Mesh{indexFormat=IndexFormat.UInt32};resources.Add(mesh);
                    int n=model->mesh_vertnum[meshId],offset=model->mesh_vertadr[meshId];
                    var vertices=new Vector3[n];
                    for(int j=0;j<n;j++)
                    {
                        float* p=model->mesh_vert+3*(offset+j);vertices[j]=new Vector3(p[0],p[2],p[1]);
                    }
                    int nf=model->mesh_facenum[meshId],fo=model->mesh_faceadr[meshId];var triangles=new int[3*nf];
                    for(int j=0;j<nf;j++)
                    {
                        int* f=model->mesh_face+3*(fo+j);triangles[3*j]=f[0];triangles[3*j+1]=f[2];triangles[3*j+2]=f[1];
                    }
                    mesh.vertices=vertices;mesh.triangles=triangles;mesh.RecalculateNormals();mesh.RecalculateBounds();
                    go=new GameObject("Sai visual "+i);go.AddComponent<MeshFilter>().sharedMesh=mesh;go.AddComponent<MeshRenderer>();
                }
                else
                {
                    int type=model->geom_type[i];
                    var primitive=type==(int)MujocoLib.mjtGeom.mjGEOM_SPHERE?PrimitiveType.Sphere:
                        type==(int)MujocoLib.mjtGeom.mjGEOM_CYLINDER?PrimitiveType.Cylinder:PrimitiveType.Cube;
                    go=GameObject.CreatePrimitive(primitive);Destroy(go.GetComponent<Collider>());
                    double* sz=model->geom_size+3*i;
                    go.transform.localScale=type==(int)MujocoLib.mjtGeom.mjGEOM_PLANE?new Vector3(20,.002f,20):
                        primitive==PrimitiveType.Sphere?Vector3.one*(float)(2*sz[0]):
                        primitive==PrimitiveType.Cylinder?new Vector3((float)(2*sz[0]),(float)sz[1],(float)(2*sz[0])):
                        new Vector3((float)(2*sz[0]),(float)(2*sz[2]),(float)(2*sz[1]));
                }
                go.transform.SetParent(transform,false);
                var material=new Material(Shader.Find("Standard"));resources.Add(material);
                material.color=new Color(rgba[0],rgba[1],rgba[2],rgba[3]);material.SetFloat("_Glossiness",.25f);
                go.GetComponent<MeshRenderer>().sharedMaterial=material;visuals.Add(new KeyValuePair<int,Transform>(i,go.transform));
            }
        }
        private void OnGUI()
        {
            GUI.Box(new Rect(12,12,540,76),"Sai_Agent_001 · native MuJoCo + ONNX\nW/S drive · A/D turn · hold Shift crouch · R reset\n"+(fault==""?"Flat policy · Unity runtime acceptance pending":fault));
        }
        private void OnDestroy()
        {
            worker?.Dispose();
            if(data!=null){MujocoLib.mj_deleteData(data);data=null;}
            if(model!=null){MujocoLib.mj_deleteModel(model);model=null;}
            foreach(var resource in resources)Destroy(resource);
            if(previousFixedDelta>0)Time.fixedDeltaTime=previousFixedDelta;
        }
    }
}
