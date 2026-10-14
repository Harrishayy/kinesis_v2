"""Phone videos -> per-video hand tracks (.npz) + a summary json.

    uv run python scripts/extract_hands.py --in data/raw --out data/hands
"""
from __future__ import annotations

import argparse
from pathlib import Path

from kinesis2.hands import ensure_model, extract_track, interpolate_gaps, summary, write_summary

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/hands"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--max-fps", type=float, default=30.0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    model = ensure_model(a.models)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    vids = sorted(p for p in a.inp.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not vids:
        raise SystemExit(f"no videos in {a.inp}")
    for v in vids:
        dst = a.out / (v.stem + ".npz")
        if dst.exists() and not a.overwrite:
            from kinesis2.hands import HandTrack
            tr = HandTrack.load(dst)
        else:
            tr = interpolate_gaps(extract_track(v, model, a.max_fps))
            tr.save(dst)
        s = summary(tr)
        rows.append(s)
        print(f"{v.name:40s} frames={s['frames']:5d} detected={s['detected_frac']:.2f} "
              f"pinch=[{s['pinch_min']}, {s['pinch_max']}]")
    write_summary(rows, a.out / "summary.json")


if __name__ == "__main__":
    main()
