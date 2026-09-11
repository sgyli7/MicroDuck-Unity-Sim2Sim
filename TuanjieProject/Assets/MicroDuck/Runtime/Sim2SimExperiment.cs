using System;
using System.Collections.Generic;

namespace AgenticRobot.Experiments
{
    // This exact source is also compiled into the measured reference adapter.
    // It describes inputs only, never states computed by either physics engine.
    [Serializable] public sealed class ExperimentModel { public int slot; public string sha256; }
    [Serializable] public sealed class ExperimentEvent
    {
        public int physicsStep;
        public string kind;
        public int slot;
        public float[] values;
    }
    [Serializable] public sealed class ExperimentCase
    {
        public string id;
        public int seed;
        public int slot;
        public int initialSlot;
        public string terrain;
        public int physicsSteps;
        public float[] initialRootPosition;
        public float initialYawDegrees;
        public ExperimentEvent[] events;

        public static bool Roller(int slot) => slot == 7 || slot == 8;
        private static bool Finite(float x) => !float.IsNaN(x) && !float.IsInfinity(x);
        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(id) || id.Length > 120) throw new ArgumentException("Invalid case id");
            if (slot < 1 || slot > 9 || initialSlot < 1 || initialSlot > 9 || Roller(slot) != Roller(initialSlot))
                throw new ArgumentException("Invalid policy slot");
            if (terrain != "flat") throw new ArgumentException("This experiment adapter does not yet implement the requested terrain");
            if (physicsSteps < 4 || physicsSteps > 12000 || physicsSteps % 4 != 0)
                throw new ArgumentException("Experiment horizon must use 4..12000 physics ticks divisible by four");
            if (initialRootPosition == null || initialRootPosition.Length != 3
                || !Finite(initialYawDegrees) || Math.Abs(initialYawDegrees) > 180)
                throw new ArgumentException("Invalid initial pose");
            foreach (float value in initialRootPosition)
                if (!Finite(value)) throw new ArgumentException("Non-finite initial position");
            if (Math.Abs(initialRootPosition[0]) > 2 || Math.Abs(initialRootPosition[2]) > 2
                || initialRootPosition[1] < 0.08f || initialRootPosition[1] > 0.3f)
                throw new ArgumentException("Initial position outside the supported flat-floor region");
            int last = -1;
            int active = initialSlot;
            if (events == null || events.Length > 100) throw new ArgumentException("Invalid event list");
            foreach (var item in events)
            {
                if (item == null || item.physicsStep < 0 || item.physicsStep >= physicsSteps
                    || item.physicsStep % 4 != 0 || item.physicsStep < last)
                    throw new ArgumentException("Events must be ordered control-boundary ticks before the horizon");
                last = item.physicsStep;
                if (item.kind == "switch")
                {
                    if (item.slot < 1 || item.slot > 9 || Roller(item.slot) != Roller(active))
                        throw new ArgumentException("Live skill switch must retain the same robot variant");
                    active = item.slot;
                }
                else if (item.kind == "twist")
                {
                    if (item.values == null || item.values.Length != 3)
                        throw new ArgumentException("Twist requires three values");
                    foreach (float value in item.values)
                        if (!Finite(value) || Math.Abs(value) > 1)
                            throw new ArgumentException("Twist outside experiment command envelope");
                }
                else if (item.kind != "trigger") throw new ArgumentException("Unknown experiment event");
            }
        }
    }
    [Serializable] public sealed class ExperimentBatch
    {
        public int schemaVersion;
        public string coordinateBasis;
        public ExperimentModel[] models;
        public ExperimentCase[] cases;
        public void Validate()
        {
            if (schemaVersion != 1 || coordinateBasis != "Unity-Xright-Yup-Zforward")
                throw new ArgumentException("Unsupported shared experiment schema/basis");
            if (models == null || models.Length != 9) throw new ArgumentException("Nine source model hashes required");
            var slots = new HashSet<int>();
            foreach (var model in models)
            {
                if (model == null || model.slot < 1 || model.slot > 9 || !slots.Add(model.slot)
                    || model.sha256 == null || model.sha256.Length != 64)
                    throw new ArgumentException("Invalid model hash/slot");
                foreach (char digit in model.sha256)
                    if (!(digit >= '0' && digit <= '9') && !(digit >= 'a' && digit <= 'f'))
                        throw new ArgumentException("Invalid source hash");
            }
            if (cases == null || cases.Length == 0 || cases.Length > 900)
                throw new ArgumentException("Expected 1..900 experiment cases");
            var ids = new HashSet<string>();
            foreach (var item in cases)
            {
                if (item == null) throw new ArgumentException("Missing experiment case");
                item.Validate();
                if (!ids.Add(item.id)) throw new ArgumentException("Duplicate case id");
            }
        }
    }
}
