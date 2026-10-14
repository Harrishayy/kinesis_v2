import numpy as np
from kinesis2.retarget import Reference
from kinesis2.tracker import EETracker, tracking_rms


def _ref():
    t = np.arange(40) / 20.0
    xyz = np.column_stack([np.linspace(0, 0.2, 40), np.zeros(40), np.full(40, 1.0)])
    grip = np.zeros(40); grip[15:30] = 1
    return Reference(t, xyz, grip, 15, 30, 20.0)


def test_action_direction_and_saturation():
    tr = EETracker(kp=8.0, pos_gain_m=0.05)
    ref = _ref()
    a = tr.action(np.array([-1.0, 0.0, 1.0]), ref, 0)      # far behind in x -> saturate +x
    assert a[0] == 1.0 and abs(a[1]) < 1e-6 and abs(a[2]) < 1e-6
    assert a[6] == -1.0                                     # open before grasp
    a = tr.action(ref.xyz[20], ref, 20)
    assert a[6] == 1.0                                      # closed mid-grasp
    assert 0 < a[0] < 0.2                                   # only feed-forward remains on-track


def test_pure_feedforward_converges_in_sim_of_integrator():
    tr = EETracker(kp=6.0, pos_gain_m=0.05, ff=1.0)
    ref = _ref()
    p = ref.xyz[0] + np.array([0.02, -0.01, 0.0])
    got = []
    for k in range(len(ref.t)):
        a = tr.action(p, ref, k)
        p = p + a[:3] * tr.pos_gain_m          # perfect integrator plant
        got.append(p.copy())
    assert tracking_rms(ref, np.array(got)) < 0.01
