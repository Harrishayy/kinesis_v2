"""Thin wrapper over LIBERO's OffScreenRenderEnv exposing raw object poses.

We bypass lerobot's gym wrapper on purpose: it hides object positions, and the
retargeting needs the bowl and plate positions at reset to anchor the reference.
Observation formatting mirrors lerobot's libero_processor so the recorded data
matches HuggingFaceVLA/libero (the SmolVLA-LIBERO training distribution).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

LANGUAGE_BY_TASK: dict[str, str] = {}


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    """robosuite xyzw quaternion -> axis-angle (3,). Same maths as lerobot's processor."""
    q = np.asarray(q, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - q[3] ** 2, 0.0))
    if den < 1e-8:
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


@dataclass
class SimStep:
    image: np.ndarray      # agentview (H, W, 3) uint8, HF-libero orientation
    image2: np.ndarray     # wrist cam
    state: np.ndarray      # (8,) eef pos, axis-angle, gripper qpos
    eef_pos: np.ndarray    # (3,)
    raw: dict


class LiberoScene:
    def __init__(self, suite: str, task_id: int, image_size: int = 256, hz: int = 20,
                 flip_images: bool = True):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.suite = benchmark.get_benchmark_dict()[suite]()
        task = self.suite.get_task(task_id)
        self.language = task.language
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=image_size,
                                      camera_widths=image_size, control_freq=hz)
        self.init_states = self.suite.get_task_init_states(task_id)
        self.flip = flip_images
        self.hz = hz

    def reset(self, init_state_id: int, settle_steps: int = 10) -> SimStep:
        self.env.reset()
        raw = self.env.set_init_state(self.init_states[init_state_id % len(self.init_states)])
        for _ in range(settle_steps):
            raw, *_ = self.env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
        return self._fmt(raw)

    def step(self, action: np.ndarray) -> tuple[SimStep, bool]:
        raw, _, done, _ = self.env.step(np.asarray(action, dtype=np.float32))
        return self._fmt(raw), bool(self.env.check_success())

    def _fmt(self, raw: dict) -> SimStep:
        im, im2 = raw["agentview_image"], raw["robot0_eye_in_hand_image"]
        if self.flip:
            im, im2 = im[::-1, ::-1].copy(), im2[::-1, ::-1].copy()
        pos, quat, g = raw["robot0_eef_pos"], raw["robot0_eef_quat"], raw["robot0_gripper_qpos"]
        state = np.concatenate([pos, quat2axisangle(quat), g]).astype(np.float32)
        return SimStep(im, im2, state, np.asarray(pos, dtype=np.float64), raw)

    def object_xyz(self, key: str, raw: dict) -> np.ndarray:
        return np.asarray(raw[key], dtype=np.float64)

    def close(self) -> None:
        self.env.close()
