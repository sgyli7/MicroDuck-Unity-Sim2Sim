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
        public NNModel StairPolicy;
        public SaiStairControl StairControl=new SaiStairControl();
        public string ModelRelativePath="SaiAgent001/models/full/locomotion-articulated.xml";
        private SaiNativeWorld world;
        private MujocoLib.mjModel_* model => world==null?null:world.Model;
        private MujocoLib.mjData_* data => world==null?null:world.Data;
        private SaiBarracudaActor actor,stairActor;
        private readonly List<KeyValuePair<int,Transform>> visuals=new List<KeyValuePair<int,Transform>>();
        private readonly List<UnityEngine.Object> resources=new List<UnityEngine.Object>();
        private double vx,wz,requestedCrouch;
        private float previousFixedDelta;
        private string fault="";
        private Camera view;

        private void Start()
        {
            previousFixedDelta=Time.fixedDeltaTime;
            try
            {
                if(Policy==null || StairPolicy==null)throw new InvalidOperationException("Sai ONNX assets are not assigned; run SaiAgent001/Build Demo");
                world=new SaiNativeWorld(Path.Combine(Application.streamingAssetsPath,ModelRelativePath),StairControl);
                actor=new SaiBarracudaActor(Policy);stairActor=new SaiBarracudaActor(StairPolicy);
                Time.fixedDeltaTime=.02f;BuildVisuals();SyncVisuals();
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
            if(Input.GetKeyDown(KeyCode.R) && world!=null){world.Reset();SyncVisuals();}
        }
        private void FixedUpdate()
        {
            if(world==null || actor==null || stairActor==null || fault!="")return;
            try
            {
                world.Step(vx,wz,requestedCrouch,(observation,stairs)=>(stairs?stairActor:actor).Infer(observation));
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
            for(int i=0;i<checked((int)model->ngeom);i++)
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
            GUI.Box(new Rect(12,12,540,76),"Sai_Agent_001 · native MuJoCo + ONNX\nW/S drive · A/D turn · hold Shift crouch · R reset\n"+(fault==""?(world!=null && world.OnStairs?"Stair policy":"Flat policy")+" · Unity editor acceptance pending":fault));
        }
        private void OnDestroy()
        {
            actor?.Dispose();stairActor?.Dispose();world?.Dispose();world=null;
            foreach(var resource in resources)Destroy(resource);
            if(previousFixedDelta>0)Time.fixedDeltaTime=previousFixedDelta;
        }
    }
}
