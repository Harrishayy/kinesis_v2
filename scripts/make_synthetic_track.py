"""Write a synthetic hand track that mimics a pick-and-place clip, for sim smoke tests only.

    uv run python scripts/make_synthetic_track.py --out data/hands_synth

Not used for any reported result: the challenge requires personally collected data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kinesis2.hands import HandTrack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/hands_synth"))
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()
    hz, n = 30.0, int(a.seconds * 30)
    t = np.arange(n) / hz
    # approach (0-1.5s), descend+grasp (1.5-2.5s), carry (2.5-4.5s), release (4.5-5s), retreat
    u = np.interp(t, [0, 1.5, 2.5, 4.5, 5.0, a.seconds], [0.3, 0.35, 0.35, 0.65, 0.65, 0.6])
    v = np.interp(t, [0, 1.5, 2.5, 4.5, 5.0, a.seconds], [0.7, 0.55, 0.55, 0.45, 0.45, 0.6])
    # palm scale: larger when the hand is high (close to a downward-looking phone)
    scale = np.interp(t, [0, 1.5, 2.5, 3.5, 4.5, 5.0, a.seconds], [0.14, 0.14, 0.10, 0.14, 0.12, 0.12, 0.14])
    pinch = np.interp(t, [0, 2.3, 2.5, 4.6, 4.8, a.seconds], [1.0, 1.0, 0.2, 0.2, 1.0, 1.0])
    rng = np.random.default_rng(0)
    uv = np.column_stack([u, v]) + rng.normal(0, 0.003, (n, 2))
    a.out.mkdir(parents=True, exist_ok=True)
    HandTrack(t, uv, scale + rng.normal(0, 0.002, n), pinch, np.ones(n, bool), 1080, 1920,
              "synthetic").save(a.out / "synthetic.npz")
    print("wrote", a.out / "synthetic.npz")


if __name__ == "__main__":
    main()
