import numpy as np
from kinesis2.hands import HandTrack
from kinesis2.retarget import build_reference, grip_from_pinch, grasp_release, similarity_2d, rebase_reference


def test_grip_hysteresis_and_blip_removal():
    p = np.array([1, 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1, 1, 1, 1, 0.3, 1, 1, 1, 1, 1])
    g = grip_from_pinch(p, min_len=3)
    assert g[2:8].all() and not g[:2].any()
    assert not g[12]                       # single-frame blip removed
    assert grasp_release(g) == (2, 8)


def test_similarity_maps_anchors_exactly():
    f = similarity_2d(np.array([0.2, 0.8]), np.array([0.6, 0.8]), np.array([0.0, 0.1]), np.array([0.1, 0.3]))
    assert np.allclose(f(np.array([[0.2, 0.8]]))[0], [0.0, 0.1])
    assert np.allclose(f(np.array([[0.6, 0.8]]))[0], [0.1, 0.3])


def _synthetic_track(n=120, hz=30.0):
    t = np.arange(n) / hz
    # hand moves right across the image, descends (palm grows), pinches in the middle
    uv = np.column_stack([np.linspace(0.2, 0.8, n), np.full(n, 0.5)])
    scale = 0.10 + 0.04 * np.sin(np.linspace(0, np.pi, n))      # low at ends, high mid-clip
    pinch = np.ones(n); pinch[40:80] = 0.2
    return HandTrack(t, uv, scale, pinch, np.ones(n, bool), 1080, 1920, "synthetic")


def test_build_reference_anchors_and_heights():
    ref = build_reference(_synthetic_track(), obj_xy=np.array([0.0, 0.0]), target_xy=np.array([0.2, 0.1]),
                          z_table=0.9, z_lift=1.1, hz=20.0)
    assert np.allclose(ref.xyz[ref.grasp_idx, :2], [0.0, 0.0], atol=1e-6)
    assert np.allclose(ref.xyz[ref.release_idx - 1, :2], [0.2, 0.1], atol=1e-6)
    assert np.allclose(ref.xyz[:ref.grasp_idx, 2], 1.1)
    assert ref.grip[ref.grasp_idx] == 1 and ref.grip[ref.release_idx] == 0
    assert np.isclose(ref.t[1] - ref.t[0], 0.05)
    moved = rebase_reference(ref, np.array([0.1, 0.1]), np.array([0.3, 0.2]))
    assert np.allclose(moved.xyz[ref.grasp_idx, :2], [0.1, 0.1], atol=1e-6)
