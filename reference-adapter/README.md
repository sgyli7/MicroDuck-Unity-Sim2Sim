# Additive measurements for the frozen native reference

This directory is **not** part of the PhysX Tuanjie project. It is an overlay for a
separate copy of the preserved MuJoCo reference, whose physics/controller sources
remain unchanged. Never copy it into the game or modify the frozen original.

Current local derivative: `../AgenticRobotGame-MuJoCoMeasured-20260909`.
Copy Runtime/Editor under its `TuanjieProject/Assets/ReferenceAdapter/`; copy the test
source into that derivative's existing `Assets/MicroDuck/Tests/PlayMode/` assembly.
The test suite verifies observer noninterference (full native qpos, stand at 0.2 s)
and same-step quaternion/upright consistency. It does not prove all nine skills.

Build with `-executeMethod AgenticRobot.Reference.MeasuredReferenceBuild.Build`.
This builds the existing frozen scene without regenerating its robot or environment.
The resulting separate Player accepts `-referenceOutput <new absolute JSON path>`.
It runs all nine fixed diagnostic cases and exits. Existing output is refused.
Frames are passively captured at every 0.005 s native step. Contacts describe the
just-solved interval; pose/upright describe the post-integration state. No extra
`mj_step` or `mj_forward` is issued to obtain measurements.

`completed` means only that a healthy controller reached the requested horizon.
`behaviorAccepted` stays false. In particular standalone roller crouch does not yet
implement the required moving approach. These traces are neither official-upstream
baseline evidence nor PhysX passes. The current batch driver still uses fixed case
definitions; shared experiment files and full phase-aware acceptance are outstanding.
