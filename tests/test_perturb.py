import numpy as np
from kinesis2.perturb import ActionDelay, StateNoise, ImageShift


def test_action_delay_shifts_by_k():
    noop = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
    d = ActionDelay(2, noop)
    outs = [d(np.full(7, i, np.float32)) for i in range(1, 5)]
    assert np.allclose(outs[0], noop) and np.allclose(outs[1], noop)
    assert outs[2][0] == 1 and outs[3][0] == 2
    assert np.array_equal(ActionDelay(0, noop)(np.ones(7)), np.ones(7))


def test_state_noise_only_on_requested_dims():
    n = StateNoise(0.01, 0.0, 0.0, seed=1)
    s = n(np.zeros(8))
    assert np.abs(s[:3]).sum() > 0 and np.abs(s[3:]).sum() == 0


def test_image_shift_keeps_dtype_and_range():
    img = np.full((8, 8, 3), 200, np.uint8)
    out = ImageShift(0.5, 2, seed=0)(img)
    assert out.dtype == np.uint8 and out.shape == img.shape and out.max() <= 255
