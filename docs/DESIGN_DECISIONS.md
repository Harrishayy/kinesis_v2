# Design decisions

Short notes on the choices that shaped this repo, in the order they were made. Results
and numbers live in `RESULTS.md`.

## 1. Hand tracking: MediaPipe, monocular, no calibration

The phone clips are hand-held, uncalibrated, and the sim scene shares nothing with my
kitchen table except the task structure. Recovering metric 3-D hand pose (HaMeR, WiLoR,
or a full SfM pass) would be slower to set up and would still leave a frame-alignment
problem. MediaPipe's hand landmarker runs on a laptop CPU at video rate and gives three
signals that are enough for a top-down pick-and-place:

- wrist position in the image,
- palm width in the image, a proxy for distance to the camera and hence height above the table,
- thumb-to-index distance over palm width, a scale-free grip signal.

Short dropouts are interpolated; long ones are left as gaps and the clip is rejected.

## 2. Retargeting: anchor on task structure, not geometry

A pick-and-place clip has two moments that carry meaning: the grasp and the release. In
the sim those correspond to the bowl and the plate, whose positions are known at reset.
We fit a 2-D similarity transform (scale, rotation, translation) that sends the human
grasp point to the bowl and the human release point to the plate, and apply it to the
whole track. Height is the palm-scale profile normalised between its value at grasp
(lowest) and its maximum (highest). The result is a reference trajectory whose every
sample came from the video, expressed in the robot's workspace.

Because the fit is per reset, one clip re-anchors cleanly onto every LIBERO initial
state. That is the data multiplier: clips x init states candidate episodes, filtered by
the simulator's own success check.

## 3. Driving the arm: a small closed-loop tracker

LIBERO exposes an OSC_POSE controller (7-D delta actions). We command translation and
grip with a saturated proportional controller plus a feed-forward term from the
reference velocity, hold orientation at the home pose, and lead the grip command by two
steps to absorb gripper latency. This is the task-space cousin of the IK feed-forward +
residual used in Kinesis; it is enough to follow the reference to roughly a centimetre.

## 4. Data format: match the SmolVLA-LIBERO distribution exactly

Episodes are written as a LeRobot v3 dataset with the same features as
`HuggingFaceVLA/libero`: two 256x256 cameras (agentview and wrist, rotated 180 degrees to
the HF convention), an 8-D state (eef position, axis-angle, gripper), 7-D actions and the
task's language string. The reason is that the policy stage starts from
`HuggingFaceVLA/smolvla_libero`, so the fine-tuning data should look like its pre-training
data in everything except where the actions came from.

Control and recording rate are 20 Hz, the LIBERO default, rather than the 10 fps label on
the HF dataset: the pretrained checkpoint is evaluated at 20 Hz in lerobot's own eval
loop, so this is the rate the policy actually acts at.
