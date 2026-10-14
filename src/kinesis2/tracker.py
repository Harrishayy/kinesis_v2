"""Closed-loop end-effector tracker for LIBERO's OSC_POSE action space.

LIBERO actions are 7-D in [-1, 1]: (dx, dy, dz, droll, dpitch, dyaw, gripper), where the
first three are scaled by the controller to a max of `pos_gain_m` metres per step and the
gripper is -1 open / +1 close. We only command translation and grip; orientation is held
at the home pose (gripper pointing down), which is all a top-down pick-and-place needs.

The tracker is a saturated proportional controller with a feed-forward term from the
reference velocity. That is the same structure as the IK feed-forward + correction used
in Kinesis, just in task space. It is deliberately simple: the point of this repo is the
data path, not the controller.
"""
from __future__ import annotations

import numpy as np

from kinesis2.retarget import Reference


class EETracker:
    def __init__(self, kp: float = 8.0, pos_gain_m: float = 0.05, ff: float = 1.0,
                 grip_lead_steps: int = 2):
        self.kp = kp
        self.pos_gain_m = pos_gain_m   # robosuite OSC_POSE output_max for position
        self.ff = ff
        self.grip_lead = grip_lead_steps

    def action(self, eef_pos: np.ndarray, ref: Reference, k: int) -> np.ndarray:
        """Action at reference index k given the measured end-effector position."""
        k = min(k, len(ref.t) - 1)
        target = ref.xyz[k]
        err = target - np.asarray(eef_pos)
        v_ff = (ref.xyz[min(k + 1, len(ref.t) - 1)] - target) if k + 1 < len(ref.t) else np.zeros(3)
        step_m = self.kp * err / ref.hz + self.ff * v_ff      # metres we want to move this step
        u = np.clip(step_m / self.pos_gain_m, -1.0, 1.0)
        g = ref.grip[min(k + self.grip_lead, len(ref.t) - 1)]
        return np.array([u[0], u[1], u[2], 0.0, 0.0, 0.0, 1.0 if g > 0.5 else -1.0], dtype=np.float32)


def tracking_rms(ref: Reference, achieved: np.ndarray) -> float:
    n = min(len(ref.xyz), len(achieved))
    return float(np.sqrt(np.mean(np.sum((ref.xyz[:n] - achieved[:n]) ** 2, axis=1))))
