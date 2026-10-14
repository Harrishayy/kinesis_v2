import numpy as np
from kinesis2.hands import HandTrack, features_from_landmarks, interpolate_gaps, INDEX_MCP, PINKY_MCP, THUMB_TIP, INDEX_TIP, WRIST


def _lm(pinch_dist: float, palm: float = 0.1):
    lm = np.zeros((21, 3))
    lm[WRIST] = [0.5, 0.5, 0]
    lm[INDEX_MCP] = [0.5, 0.4, 0]
    lm[PINKY_MCP] = [0.5 + palm, 0.4, 0]
    lm[THUMB_TIP] = [0.6, 0.3, 0]
    lm[INDEX_TIP] = [0.6 + pinch_dist, 0.3, 0]
    return lm


def test_features_pinch_normalised_by_palm():
    uv, scale, pinch = features_from_landmarks(_lm(0.05))
    assert np.allclose(uv, [0.5, 0.5])
    assert np.isclose(scale, 0.1)
    assert np.isclose(pinch, 0.5)
    _, _, closed = features_from_landmarks(_lm(0.0))
    assert closed == 0.0


def test_interpolate_short_gap_only():
    t = np.arange(10) / 10.0
    ok = np.ones(10, bool); ok[3] = False; ok[6:9] = False  # 0.1 s gap and 0.3 s gap
    uv = np.tile(t[:, None], (1, 2)); uv[~ok] = np.nan
    tr = HandTrack(t, uv, t.copy(), t.copy(), ok, 100, 100, "x")
    tr.scale[~ok] = np.nan; tr.pinch[~ok] = np.nan
    out = interpolate_gaps(tr, max_gap_s=0.25)
    assert np.isclose(out.uv[3, 0], 0.3)          # short gap filled
    assert np.isnan(out.uv[7, 0])                 # long gap left as NaN
