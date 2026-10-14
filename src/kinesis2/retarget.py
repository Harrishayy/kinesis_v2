"""Hand track -> Panda end-effector reference trajectory, anchored to sim objects.

The human clip and the simulated scene never share a frame, so we do not try to
calibrate the phone. Instead we use the one thing both share: the task structure.
A pick-and-place clip starts with the hand over the source object and ends with it
over the target. We fit a 2-D similarity transform (scale, rotation, translation)
that maps the human track's grasp point and release point onto the sim object's
and target's table-plane positions, and apply it to the whole in-plane track.
Height comes from a normalised palm-scale profile, grip from the pinch signal.

Result: a time-indexed reference (xyz, gripper) at the sim control rate that a
simple closed-loop tracker can follow. Every number in it traces back to the
phone video.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from kinesis2.hands import HandTrack


@dataclass
class Reference:
    t: np.ndarray        # (N,) seconds, uniform at `hz`
    xyz: np.ndarray      # (N, 3) world metres
    grip: np.ndarray     # (N,) 1 = closed, 0 = open
    grasp_idx: int
    release_idx: int
    hz: float


def _clean(track: HandTrack) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ok = np.isfinite(track.uv).all(1) & np.isfinite(track.scale) & np.isfinite(track.pinch)
    if ok.sum() < 10:
        raise ValueError(f"{track.source}: too few valid frames ({ok.sum()})")
    return track.t[ok], track.uv[ok], track.scale[ok], track.pinch[ok]


def grip_from_pinch(pinch: np.ndarray, closed_thresh: float = 0.45, open_thresh: float = 0.75,
                    min_len: int = 5) -> np.ndarray:
    """Hysteresis threshold on pinch ratio -> boolean closed signal. Drops blips < min_len."""
    g = np.zeros(len(pinch), dtype=bool)
    closed = False
    for i, p in enumerate(pinch):
        if not closed and p < closed_thresh:
            closed = True
        elif closed and p > open_thresh:
            closed = False
        g[i] = closed
    # remove short runs
    i = 0
    while i < len(g):
        j = i
        while j < len(g) and g[j] == g[i]:
            j += 1
        if j - i < min_len and 0 < i and j < len(g):
            g[i:j] = g[i - 1]
        i = j
    return g


def grasp_release(grip: np.ndarray) -> tuple[int, int]:
    """Indices of the first close and the following open. Longest closed segment wins."""
    best, cur_start = None, None
    for i, c in enumerate(np.append(grip, False)):
        if c and cur_start is None:
            cur_start = i
        elif not c and cur_start is not None:
            if best is None or (i - cur_start) > (best[1] - best[0]):
                best = (cur_start, i)
            cur_start = None
    if best is None:
        raise ValueError("no grasp found in clip (pinch never closed)")
    return best


def similarity_2d(src_a: np.ndarray, src_b: np.ndarray, dst_a: np.ndarray, dst_b: np.ndarray):
    """Similarity (scale s, rotation R, translation t) with src_a->dst_a, src_b->dst_b.
    Returns a function mapping (N,2) -> (N,2)."""
    ds, dd = src_b - src_a, dst_b - dst_a
    ns, nd = np.linalg.norm(ds), np.linalg.norm(dd)
    if ns < 1e-6:
        raise ValueError("grasp and release points coincide in the human track")
    s = nd / ns
    ang = np.arctan2(dd[1], dd[0]) - np.arctan2(ds[1], ds[0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    t = dst_a - s * R @ src_a
    return lambda p: (s * (R @ np.asarray(p).T)).T + t


def build_reference(track: HandTrack, obj_xy: np.ndarray, target_xy: np.ndarray,
                    z_table: float, z_lift: float, hz: float = 20.0,
                    smooth_s: float = 0.25, image_y_down_is_world_minus_y: bool = True) -> Reference:
    """Map a hand track onto the sim scene. obj_xy/target_xy are world table-plane coords.

    Height model: z = z_table + (z_lift - z_table) * h, where h in [0,1] is the palm scale
    normalised between its value at grasp (0, hand lowest) and its max (1, hand highest).
    Before grasp and after release, we hold at z_lift to approach from above.
    """
    t, uv, scale, pinch = _clean(track)
    uv = uv.copy()
    if image_y_down_is_world_minus_y:
        uv[:, 1] = -uv[:, 1]   # phone looks down at the table: image-down ~ world -y
    # resample to uniform rate first so smoothing windows are in seconds
    tu = np.arange(t[0], t[-1], 1.0 / hz)
    uv = np.stack([np.interp(tu, t, uv[:, c]) for c in range(2)], 1)
    scale = np.interp(tu, t, scale)
    pinch = np.interp(tu, t, pinch)
    win = max(5, int(smooth_s * hz) | 1)
    if len(tu) > win:
        uv = savgol_filter(uv, win, 3, axis=0)
        scale = savgol_filter(scale, win, 3)
    grip = grip_from_pinch(pinch)
    gi, ri = grasp_release(grip)

    f = similarity_2d(uv[gi], uv[ri - 1], obj_xy, target_xy)
    xy = f(uv)

    s_g, s_max = scale[gi], scale.max()
    h = np.clip((scale - s_g) / max(s_max - s_g, 1e-6), 0, 1)
    # larger palm = closer to camera = higher above table (camera looks down)
    z = z_table + (z_lift - z_table) * h
    z[:gi] = z_lift        # approach from above
    z[ri:] = z_lift        # retreat upward after release
    xyz = np.column_stack([xy, z])
    return Reference(tu - tu[0], xyz, grip.astype(float), gi, ri, hz)


def rebase_reference(ref: Reference, obj_xy: np.ndarray, target_xy: np.ndarray) -> Reference:
    """Re-anchor an existing reference to new object/target positions (augmentation)."""
    f = similarity_2d(ref.xyz[ref.grasp_idx, :2], ref.xyz[ref.release_idx - 1, :2], obj_xy, target_xy)
    xyz = ref.xyz.copy(); xyz[:, :2] = f(ref.xyz[:, :2])
    return Reference(ref.t, xyz, ref.grip, ref.grasp_idx, ref.release_idx, ref.hz)
