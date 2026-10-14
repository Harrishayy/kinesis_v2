"""Perturbations for the robustness evaluation.

Applied around the policy, not inside the simulator, so the same wrappers work for any
policy and any sim. Each is deterministic given its seed.

- ActionDelay(k): the arm executes the action the policy emitted k steps ago.
- StateNoise(sigma): Gaussian noise on the proprioceptive state the policy sees.
- ImageShift: brightness/contrast jitter and small translations of camera images.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class ActionDelay:
    def __init__(self, steps: int, noop: np.ndarray):
        self.steps = int(steps)
        self.buf: deque[np.ndarray] = deque([np.asarray(noop, np.float32)] * self.steps, maxlen=max(self.steps, 1))
        self.noop = np.asarray(noop, np.float32)

    def __call__(self, action: np.ndarray) -> np.ndarray:
        if self.steps == 0:
            return action
        out = self.buf[0]
        self.buf.append(np.asarray(action, np.float32))
        return out

    def reset(self) -> None:
        self.buf = deque([self.noop] * self.steps, maxlen=max(self.steps, 1))


class StateNoise:
    def __init__(self, sigma_pos: float, sigma_rot: float = 0.0, sigma_grip: float = 0.0, seed: int = 0):
        self.s = np.array([sigma_pos] * 3 + [sigma_rot] * 3 + [sigma_grip] * 2, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return (np.asarray(state, np.float32) + self.rng.normal(0, 1, 8).astype(np.float32) * self.s)


class ImageShift:
    def __init__(self, brightness: float = 0.0, max_shift_px: int = 0, seed: int = 0):
        self.b, self.px = float(brightness), int(max_shift_px)
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        out = img.astype(np.float32)
        if self.b > 0:
            out = out * (1.0 + self.rng.uniform(-self.b, self.b))
        if self.px > 0:
            dx, dy = self.rng.integers(-self.px, self.px + 1, 2)
            out = np.roll(out, (int(dy), int(dx)), axis=(0, 1))
        return np.clip(out, 0, 255).astype(np.uint8)
