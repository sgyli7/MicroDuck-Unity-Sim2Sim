using System;
using System.Collections;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using AgenticRobot.Experiments;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace AgenticRobot.MicroDuck
{
    // Explicit opt-in, loopback only. All simulation remains on Unity's main thread.
    public sealed class PhysXSamplerServer : MonoBehaviour
    {
        [Serializable] private sealed class ActionRow { public float[] values; }
        [Serializable] private sealed class Request
        {
            public string op;
            public int slot;
            public bool external;
            public int[] indices;
            public ActionRow[] actions;
            public string experimentJson;
            public int caseIndex;
        }
        [Serializable] private sealed class Response
        {
            public bool ok = true;
            public string engine = "PhysX";
            public string role = "target_sampler";
            public string unityVersion = Application.unityVersion;
            public string buildGuid = Application.buildGUID;
            public PolicyModelIdentity[] models;
            public float physicsDt = 0.005f;
            public float controlDt = 0.02f;
            public float resetInitializationDt = 0f;
            public int numEnvs;
            public string error;
            public PhysXStepResult[] results;
        }
        private const int MaximumBytes = 8 * 1024 * 1024;
        private PhysXPolicySession[] sessions;
        private TcpListener listener;
        private bool externalMode;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Bootstrap()
        {
            if (Array.IndexOf(Environment.GetCommandLineArgs(), "-physxPort") < 0) return;
            new GameObject("PhysX Sampler Server").AddComponent<PhysXSamplerServer>();
        }

        private static int Argument(string name, int fallback)
        {
            var args = Environment.GetCommandLineArgs();
            int index = Array.IndexOf(args, name);
            if (index < 0) return fallback;
            if (index + 1 >= args.Length || !int.TryParse(args[index + 1], out int value))
                throw new ArgumentException("Expected an integer after " + name);
            return value;
        }

        private IEnumerator Start()
        {
            yield return null; // Let scene controller initialize before disabling automatic control.
            int port = Argument("-physxPort", 62101);
            int count = Argument("-physxEnvs", 1);
            if (port < 1024 || port > 65535 || count < 1 || count > 64)
                throw new ArgumentOutOfRangeException("Sampler needs port 1024..65535 and 1..64 environments");
            if (PhysicsIdentity.LoadedForeignPhysics().Length != 0)
                throw new InvalidOperationException("Foreign physics detected in target sampler");
            var template = FindObjectOfType<MicroDuckDemoController>();
            if (template == null) throw new InvalidOperationException("No PhysX game controller found");
            template.enabled = false;
            foreach (var input in FindObjectsOfType<KeyboardPolicyInput>()) input.enabled = false;
            Physics.simulationMode = SimulationMode.Script;
            sessions = new PhysXPolicySession[count];
            for (int i = 0; i < count; i++)
            {
                var scene = SceneManager.CreateScene("PhysX sample " + i,
                    new CreateSceneParameters(LocalPhysicsMode.Physics3D));
                sessions[i] = new PhysXPolicySession(template.CreateReplica(scene));
                sessions[i].Reset(2, true);
            }
            externalMode = true;
            listener = new TcpListener(IPAddress.Loopback, port);
            listener.Start(1);
            Debug.Log($"PHYSX_SAMPLER_READY port={port} envs={count} physics=PhysX");
            while (true)
            {
                if (!listener.Pending()) { yield return null; continue; }
                using (var client = listener.AcceptTcpClient())
                {
                    client.NoDelay = true;
                    client.ReceiveTimeout = client.SendTimeout = 120000;
                    using (var stream = client.GetStream())
                    {
                        while (client.Connected)
                        {
                            try
                            {
                                string json = Receive(stream);
                                Response response;
                                try { response = Execute(JsonUtility.FromJson<Request>(json)); }
                                catch (Exception error)
                                { response = new Response { ok = false, error = error.Message, numEnvs = count }; }
                                Send(stream, JsonUtility.ToJson(response));
                            }
                            catch (IOException) { break; }
                            catch (SocketException) { break; }
                        }
                    }
                }
                yield return null;
            }
        }

        private Response Execute(Request request)
        {
            if (request == null) throw new ArgumentException("Empty request");
            var response = new Response { numEnvs = sessions.Length };
            if (request.op == "hello")
            {
                response.models = sessions[0].Controller.CaptureModelIdentities();
                return response;
            }
            if (request.op == "experiment")
            {
                if (string.IsNullOrWhiteSpace(request.experimentJson)) throw new ArgumentException("Missing experiment JSON");
                var batch = JsonUtility.FromJson<ExperimentBatch>(request.experimentJson);
                if (batch == null) throw new ArgumentException("Missing experiment batch");
                batch.Validate();
                if (request.caseIndex < 0 || request.caseIndex >= batch.cases.Length)
                    throw new ArgumentException("Invalid experiment case index");
                // Validate all requested sources before mutating any environment.
                var identities = sessions[0].Controller.CaptureModelIdentities();
                foreach (var expected in batch.models)
                {
                    var actual = Array.Find(identities, item => item.slot == expected.slot);
                    if (actual == null || !actual.graphVerified || actual.sourceSha256 != expected.sha256)
                        throw new ArgumentException("Experiment model source differs from the actual Player binding");
                }
                string hash = PolicyModelIdentity.Hash(Encoding.UTF8.GetBytes(request.experimentJson));
                response.results = new PhysXStepResult[sessions.Length];
                for (int i = 0; i < sessions.Length; i++)
                    response.results[i] = sessions[i].ResetExperiment(batch.cases[request.caseIndex], hash);
                externalMode = false;
                return response;
            }
            if (request.op == "reset")
            {
                PolicyCatalog.GetBySlot(request.slot);
                int[] selected = request.indices;
                if (selected == null || selected.Length == 0)
                {
                    selected = new int[sessions.Length];
                    for (int i = 0; i < selected.Length; i++) selected[i] = i;
                }
                if (selected.Length != sessions.Length && request.external != externalMode)
                    throw new ArgumentException("Cannot change actor mode in a partial reset");
                var seen = new System.Collections.Generic.HashSet<int>();
                foreach (int i in selected)
                    if (i < 0 || i >= sessions.Length || !seen.Add(i))
                        throw new ArgumentException("Invalid or duplicate environment index");
                response.results = new PhysXStepResult[selected.Length];
                for (int i = 0; i < selected.Length; i++)
                    response.results[i] = sessions[selected[i]].Reset(request.slot, request.external);
                externalMode = request.external;
                return response;
            }
            if (request.op != "step") throw new ArgumentException("Unsupported sampler operation");
            if (externalMode)
            {
                if (request.actions == null || request.actions.Length != sessions.Length)
                    throw new ArgumentException("One action row is required per environment");
                foreach (var row in request.actions)
                {
                    PolicyContract.ValidateAction(row?.values);
                    foreach (float value in row.values)
                        if (float.IsNaN(value) || float.IsInfinity(value) || Mathf.Abs(value) > 5f)
                            throw new ArgumentException("Action outside the finite [-5,5] execution envelope");
                }
            }
            else if (request.actions != null && request.actions.Length > 0)
                throw new ArgumentException("Internal ONNX mode takes no actions");
            response.results = new PhysXStepResult[sessions.Length];
            for (int i = 0; i < sessions.Length; i++)
                response.results[i] = sessions[i].Step(externalMode ? request.actions[i].values : null);
            return response;
        }

        private static byte[] Exact(NetworkStream stream, int count)
        {
            byte[] buffer = new byte[count];
            int offset = 0;
            while (offset < count)
            {
                int read = stream.Read(buffer, offset, count - offset);
                if (read == 0) throw new IOException("Sampler client disconnected");
                offset += read;
            }
            return buffer;
        }
        private static string Receive(NetworkStream stream)
        {
            int length = BitConverter.ToInt32(Exact(stream, 4), 0);
            if (length <= 0 || length > MaximumBytes) throw new IOException("Invalid message size");
            return Encoding.UTF8.GetString(Exact(stream, length));
        }
        private static void Send(NetworkStream stream, string json)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(json);
            if (bytes.Length > MaximumBytes) throw new IOException("Response exceeds message size limit");
            byte[] header = BitConverter.GetBytes(bytes.Length);
            stream.Write(header, 0, header.Length);
            stream.Write(bytes, 0, bytes.Length);
        }
        private void OnDestroy()
        {
            listener?.Stop();
            if (sessions != null)
                for (int i = sessions.Length - 1; i >= 0; i--) sessions[i]?.Dispose();
        }
    }
}
