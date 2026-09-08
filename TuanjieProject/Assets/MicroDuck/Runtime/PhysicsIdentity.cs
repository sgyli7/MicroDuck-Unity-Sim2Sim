using System;
using System.Diagnostics;
using System.Linq;
using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    public sealed class PhysicsIdentity : MonoBehaviour
    {
        public const string Engine = "PhysX";

        public static string[] LoadedForeignPhysics()
        {
            var managed = AppDomain.CurrentDomain.GetAssemblies()
                .Select(assembly => assembly.GetName().Name)
                .Where(name => name.IndexOf("mujoco", StringComparison.OrdinalIgnoreCase) >= 0);
            using (Process process = Process.GetCurrentProcess())
            {
                var native = process.Modules.Cast<ProcessModule>().Select(module => module.ModuleName)
                    .Where(name => name.IndexOf("mujoco", StringComparison.OrdinalIgnoreCase) >= 0);
                return managed.Concat(native).Distinct().ToArray();
            }
        }

        private void Start()
        {
            string[] foreign = LoadedForeignPhysics();
            if (foreign.Length != 0)
                throw new InvalidOperationException("PhysX game loaded foreign physics: "
                    + string.Join(", ", foreign));
            UnityEngine.Debug.Log("SIM2SIM_ENGINE=PhysX;ROLE=game;EXTERNAL_PHYSICS=none");
        }
    }
}
