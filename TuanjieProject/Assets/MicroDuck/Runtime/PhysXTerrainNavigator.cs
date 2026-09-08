using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    public sealed class PhysXTerrainNavigator : MonoBehaviour
    {
        [SerializeField] private MicroDuckDemoController controller;
        [SerializeField] private int index;
        public TerrainModuleDefinition ActiveModule => TerrainCatalog.Modules[index];
        public void Configure(MicroDuckDemoController value) { controller = value; }

        public bool Select(int value)
        {
            if (controller == null || controller.ActiveRig == null) return false;
            index = ((value % TerrainCatalog.Modules.Count) + TerrainCatalog.Modules.Count)
                % TerrainCatalog.Modules.Count;
            Vector3 spawn = ActiveModule.SpawnPosition;
            if (controller.ActiveRig.Variant == RobotVariant.Roller) spawn.y += 0.0135f;
            // The inherited reference terrain uses world X as forward; the imported
            // PhysX robot uses world Z as forward. This is an explicit terrain reset.
            controller.ActiveRig.SetResetPose(spawn,
                Quaternion.Euler(0f, ActiveModule.SpawnYawDegrees + 90f, 0f));
            controller.ResetActiveRig();
            controller.ActiveRig.RootBody.velocity = ActiveModule.InitialVelocityMetersPerSecond;
            return true;
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.T))
                Select(index + (Input.GetKey(KeyCode.LeftShift) ? -1 : 1));
        }
    }
}
