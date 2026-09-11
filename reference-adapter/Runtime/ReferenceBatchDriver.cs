using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using AgenticRobot.MicroDuck.Mujoco;
using Mujoco;
using UnityEngine;

namespace AgenticRobot.Reference
{
    [Serializable] public sealed class ReferenceEpisode
    {
        public int slot;
        public string policy;
        public float requestedSeconds;
        public bool completed;
        public bool behaviorAccepted = false;
        public string error;
        public ReferenceFrame[] frames;
    }
    [Serializable] public sealed class ReferenceBatchReport
    {
        public int schemaVersion = 1;
        public string engine = "MuJoCo";
        public string role = "current-frozen-reference-measured-copy";
        public string sourceRevision = "3a97ebec04455d9e04f725f0b6e5db7042fe94f8";
        public string coordinateBasis = "right-handed X-forward Y-left Z-up; quaternion WXYZ";
        public string sampling = "passive postUpdateEvent, every native 0.005s step";
        public string unityVersion = Application.unityVersion;
        public bool behaviorAccepted = false;
        public ReferenceEpisode[] episodes;
    }

    // Commands use public frozen-controller APIs at native simulation times.
    // This adapter never substitutes dynamics or replaces the existing control callback.
    public sealed class ReferenceBatchDriver : MonoBehaviour
    {
        private static readonly float[] Durations = { 6, 4, 8, 4, 3, 3, 6, 3, 3 };
        private MjScene scene;
        private MujocoDemoController controller;
        private MeasurementProbe probe;
        private bool recording;
        private bool sitTriggered;
        private bool standTriggered;
        private bool running;
        public bool Finished { get; private set; }
        public string Error { get; private set; }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Bootstrap()
        {
            string[] args = Environment.GetCommandLineArgs();
            int index = Array.IndexOf(args, "-referenceOutput");
            if (index < 0) return;
            if (index + 1 >= args.Length) throw new ArgumentException("-referenceOutput requires a path");
            var driver = new GameObject("MuJoCo Current Reference Measurements").AddComponent<ReferenceBatchDriver>();
            driver.Begin(args[index + 1]);
        }

        public void Begin(string outputPath)
        {
            if (running) throw new InvalidOperationException("Reference batch already running");
            if (File.Exists(outputPath)) throw new IOException("Refusing to overwrite reference evidence");
            running = true;
            StartCoroutine(Run(outputPath));
        }

        private IEnumerator Run(string outputPath)
        {
            yield return null;
            scene = FindObjectOfType<MjScene>();
            controller = FindObjectOfType<MujocoDemoController>();
            if (scene == null || controller == null)
                throw new InvalidOperationException("The frozen reference scene is required");
            foreach (var keyboard in FindObjectsOfType<MujocoKeyboardPolicyInput>()) keyboard.enabled = false;
            probe = gameObject.AddComponent<MeasurementProbe>();
            scene.preUpdateEvent += ApplyTimeline;
            var episodes = new List<ReferenceEpisode>();
            for (int slot = 1; slot <= 9; slot++)
            {
                recording = false;
                var episode = new ReferenceEpisode { slot = slot, requestedSeconds = Durations[slot - 1] };
                if (!controller.SelectPolicy(slot))
                {
                    episode.error = controller.Fault;
                    episodes.Add(episode);
                    continue;
                }
                float deadline = Time.realtimeSinceStartup + 20f;
                while ((!controller.IsHealthy || controller.PolicyTicks < 1) && Time.realtimeSinceStartup < deadline)
                    yield return new WaitForFixedUpdate();
                if (!controller.IsHealthy || controller.PolicyTicks < 1)
                {
                    episode.error = "Native controller initialization timed out: " + controller.Fault;
                    episodes.Add(episode);
                    continue;
                }
                controller.ResetActiveRobotAt(new Vector3(0f, 0.125f, 0f), 0f, Vector3.zero);
                if (slot == 1) controller.SetTwist(0.2f, 0f, 0f);
                if (slot == 7) controller.SetTwist(0.3f, 0f, 0f);
                if (slot == 4 || slot == 8) controller.TriggerSkill(0f);
                sitTriggered = standTriggered = false;
                probe.Frames.Clear();
                probe.Frames.Add(probe.Capture());
                recording = true;
                episode.policy = controller.ActivePolicyName;
                deadline = Time.realtimeSinceStartup + 45f;
                while (SimulationTime() < episode.requestedSeconds - 1e-9 && controller.IsHealthy
                    && Time.realtimeSinceStartup < deadline)
                    yield return new WaitForFixedUpdate();
                recording = false;
                episode.frames = probe.Frames.ToArray();
                episode.completed = controller.IsHealthy && SimulationTime() >= episode.requestedSeconds - 1e-9;
                episode.error = episode.completed ? null : controller.IsHealthy ? "Wall-clock timeout" : controller.Fault;
                episodes.Add(episode);
                Write(outputPath, episodes);
                Debug.Log($"REFERENCE_CASE slot={slot} engine=MuJoCo completed={episode.completed} accepted=false");
            }
            Write(outputPath, episodes);
            scene.preUpdateEvent -= ApplyTimeline;
            Finished = true;
            if (!Application.isEditor) Application.Quit(episodes.TrueForAll(e => e.completed) ? 0 : 1);
        }

        private unsafe double SimulationTime() => scene.Data->time;

        private void ApplyTimeline(object sender, MjStepArgs args)
        {
            if (!recording || controller.ActivePolicySlot != 3) return;
            double time = SimulationTime();
            if (!sitTriggered && time + 1e-9 >= 1.0)
            { controller.TriggerSkill((float)time); sitTriggered = true; }
            if (!standTriggered && time + 1e-9 >= 4.5)
            { controller.TriggerSkill((float)time); standTriggered = true; }
        }

        private static void Write(string path, List<ReferenceEpisode> episodes)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
            File.WriteAllText(path, JsonUtility.ToJson(new ReferenceBatchReport { episodes = episodes.ToArray() }));
        }

        private void OnDestroy()
        {
            if (scene != null) scene.preUpdateEvent -= ApplyTimeline;
        }
    }
}
