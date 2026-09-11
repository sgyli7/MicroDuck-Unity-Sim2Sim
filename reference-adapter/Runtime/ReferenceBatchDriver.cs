using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using AgenticRobot.Experiments;
using AgenticRobot.MicroDuck.Mujoco;
using Mujoco;
using UnityEngine;

namespace AgenticRobot.Reference
{
    [Serializable] public sealed class ReferenceEpisode
    {
        public int slot;
        public string policy;
        public string inference = "Barracuda";
        public string experimentId;
        public string experimentSha256;
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
        public string buildGuid = Application.buildGUID;
        public ReferenceModelRecord[] models;
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
        private ExperimentBatch experimentBatch;
        private ExperimentCase experiment;
        private string experimentHash;
        private int nextEvent;
        private ReferenceModelRecord[] identities;
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
            int specIndex = Array.IndexOf(args, "-referenceExperiment");
            if (specIndex >= 0)
            {
                if (specIndex + 1 >= args.Length) throw new ArgumentException("-referenceExperiment requires a JSON path");
                string json = File.ReadAllText(args[specIndex + 1]);
                driver.experimentBatch = JsonUtility.FromJson<ExperimentBatch>(json);
                driver.experimentBatch.Validate();
                driver.experimentHash = ReferenceModelIdentity.Hash(Encoding.UTF8.GetBytes(json));
            }
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
            identities = ReferenceModelIdentity.Capture(controller);
            if (experimentBatch != null)
                foreach (var expected in experimentBatch.models)
                {
                    var actual = Array.Find(identities, item => item.slot == expected.slot);
                    if (actual == null || !actual.graphVerified || actual.sourceSha256 != expected.sha256)
                        throw new InvalidOperationException("Reference model differs from shared experiment source");
                }
            foreach (var keyboard in FindObjectsOfType<MujocoKeyboardPolicyInput>()) keyboard.enabled = false;
            probe = gameObject.AddComponent<MeasurementProbe>();
            scene.preUpdateEvent += ApplyTimeline;
            var episodes = new List<ReferenceEpisode>();
            int count = experimentBatch == null ? 9 : experimentBatch.cases.Length;
            for (int caseIndex = 0; caseIndex < count; caseIndex++)
            {
                recording = false;
                experiment = experimentBatch?.cases[caseIndex];
                int slot = experiment == null ? caseIndex + 1 : experiment.slot;
                int initialSlot = experiment == null ? slot : experiment.initialSlot;
                var episode = new ReferenceEpisode { slot = slot,
                    experimentId = experiment?.id, experimentSha256 = experimentHash,
                    requestedSeconds = experiment == null ? Durations[slot - 1] : experiment.physicsSteps * 0.005f };
                if (!controller.SelectPolicy(initialSlot))
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
                Vector3 startPosition = experiment == null ? new Vector3(0f, 0.125f, 0f)
                    : new Vector3(experiment.initialRootPosition[2], experiment.initialRootPosition[1],
                        -experiment.initialRootPosition[0]);
                // Frozen plug-in SetMjVector3 maps API [x,y,z] to native [x,z,y].
                // Shared Unity basis instead maps to native [z,-x,y]. Adapt this
                // API boundary only; never modify the frozen controller itself.
                // Frozen API accepts the legged reference height and adds 0.0135 m
                // for roller rigs. The common file instead names the actual root height.
                if (experiment != null && ExperimentCase.Roller(initialSlot)) startPosition.y -= 0.0135f;
                if (!controller.ResetActiveRobotAt(startPosition, experiment?.initialYawDegrees ?? 0f, Vector3.zero))
                    throw new InvalidOperationException("Reference reset failed: " + controller.Fault);
                if (experiment == null)
                {
                    if (slot == 1) controller.SetTwist(0.2f, 0f, 0f);
                    if (slot == 7) controller.SetTwist(0.3f, 0f, 0f);
                    if (slot == 4 || slot == 8) controller.TriggerSkill(0f);
                }
                nextEvent = 0;
                ApplyExperimentEvents();
                sitTriggered = standTriggered = false;
                probe.Frames.Clear();
                probe.Frames.Add(probe.Capture());
                recording = true;
                episode.policy = controller.ActivePolicyName;
                deadline = Time.realtimeSinceStartup + Mathf.Max(45f, episode.requestedSeconds * 3f);
                int requestedTicks = experiment == null ? (int)Math.Round(Durations[slot - 1] / 0.005) : experiment.physicsSteps;
                while (SimulationTicks() < requestedTicks && controller.IsHealthy
                    && Time.realtimeSinceStartup < deadline)
                    yield return new WaitForFixedUpdate();
                recording = false;
                episode.frames = probe.Frames.ToArray();
                episode.completed = controller.IsHealthy && SimulationTicks() >= requestedTicks;
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
        private int SimulationTicks() => (int)Math.Round(SimulationTime() / 0.005);

        private void ApplyTimeline(object sender, MjStepArgs args)
        {
            if (!recording) return;
            if (experiment != null) { ApplyExperimentEvents(); return; }
            if (controller.ActivePolicySlot != 3) return;
            double time = SimulationTime();
            if (!sitTriggered && time + 1e-9 >= 1.0)
            { controller.TriggerSkill((float)time); sitTriggered = true; }
            if (!standTriggered && time + 1e-9 >= 4.5)
            { controller.TriggerSkill((float)time); standTriggered = true; }
        }

        private void ApplyExperimentEvents()
        {
            if (experiment == null) return;
            int tick = SimulationTicks();
            while (nextEvent < experiment.events.Length && experiment.events[nextEvent].physicsStep <= tick)
            {
                var item = experiment.events[nextEvent++];
                if (item.kind == "switch" && !controller.HotSwapPolicy(item.slot))
                    throw new InvalidOperationException(controller.Fault);
                if (item.kind == "twist") controller.SetTwist(item.values[0], item.values[1], item.values[2]);
                if (item.kind == "trigger") controller.TriggerSkill((float)SimulationTime());
            }
        }

        private void Write(string path, List<ReferenceEpisode> episodes)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path)));
            File.WriteAllText(path, JsonUtility.ToJson(new ReferenceBatchReport {
                models = identities, episodes = episodes.ToArray() }));
        }

        private void OnDestroy()
        {
            if (scene != null) scene.preUpdateEvent -= ApplyTimeline;
        }
    }
}
