"""Hand tracks -> retargeted Panda references -> closed-loop replay in LIBERO -> LeRobot dataset.

    MUJOCO_GL=egl uv run python scripts/retarget_replay.py --config configs/replay_spatial.yaml \
        --hands data/hands --out data/lerobot/ego_bowl

Each human clip is re-anchored to the bowl/plate position of every LIBERO init state, so
N clips x M init states candidate episodes; only successful ones are written (unless
keep_failures). A JSON log with per-episode success and tracking RMS goes next to the dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

from kinesis2.hands import HandTrack
from kinesis2.retarget import Reference, build_reference
from kinesis2.sim import LiberoScene
from kinesis2.tracker import EETracker, tracking_rms


def dataset_features(image_size: int) -> dict:
    img = {"dtype": "video", "shape": (image_size, image_size, 3), "names": ["height", "width", "channel"]}
    return {
        "observation.images.image": img,
        "observation.images.image2": dict(img),
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }


def run_episode(scene: LiberoScene, track: HandTrack, cfg: dict, init_id: int):
    obs = scene.reset(init_id, cfg["settle_steps"])
    obj = scene.object_xyz(cfg["object_key"], obs.raw) + np.asarray(cfg["grasp_offset_xyz"])
    tgt = scene.object_xyz(cfg["target_key"], obs.raw) + np.asarray(cfg["release_offset_xyz"])
    ref = build_reference(track, obj[:2], tgt[:2], z_table=obj[2] + cfg["z_grasp"],
                          z_lift=obj[2] + cfg["z_lift"], hz=cfg["hz"])
    tr = EETracker(**cfg["tracker"])
    frames, achieved, success = [], [], False
    n = min(len(ref.t) + cfg["hold_after_release"], cfg["max_steps"])
    for k in range(n):
        a = tr.action(obs.eef_pos, ref, k)
        frames.append((obs, a))
        achieved.append(obs.eef_pos.copy())
        obs, success = scene.step(a)
        if success:
            frames.append((obs, np.zeros(7, np.float32) + np.array([0, 0, 0, 0, 0, 0, -1], np.float32)))
            break
    return ref, frames, success, tracking_rms(ref, np.array(achieved))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--hands", type=Path, default=Path("data/hands"))
    ap.add_argument("--out", type=Path, default=Path("data/lerobot/ego_bowl"))
    ap.add_argument("--previews", type=Path, default=Path("results/replay_previews"))
    ap.add_argument("--limit-tracks", type=int, default=0)
    ap.add_argument("--n-init", type=int, default=0, help="override n_init_states")
    ap.add_argument("--no-dataset", action="store_true", help="only replay + log, skip dataset write")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    if a.n_init:
        cfg["n_init_states"] = a.n_init

    tracks = sorted(a.hands.glob("*.npz"))
    if a.limit_tracks:
        tracks = tracks[: a.limit_tracks]
    if not tracks:
        raise SystemExit(f"no hand tracks in {a.hands}")

    scene = LiberoScene(cfg["suite"], cfg["task_id"], cfg["image_size"], cfg["hz"], cfg["flip_images"])
    print("task:", scene.language)

    ds = None
    if not a.no_dataset:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        if a.out.exists():
            shutil.rmtree(a.out)
        ds = LeRobotDataset.create(cfg["out_repo_id"], fps=cfg["hz"], root=a.out,
                                   features=dataset_features(cfg["image_size"]), robot_type="panda",
                                   use_videos=True, image_writer_threads=4)
    a.previews.mkdir(parents=True, exist_ok=True)
    log = []
    for tp in tracks:
        track = HandTrack.load(tp)
        for init_id in range(cfg["n_init_states"]):
            try:
                ref, frames, ok, rms = run_episode(scene, track, cfg, init_id)
            except ValueError as e:
                print(f"{tp.stem} init={init_id}: skipped ({e})")
                log.append({"track": tp.stem, "init": init_id, "success": False, "error": str(e)})
                break
            row = {"track": tp.stem, "init": init_id, "success": ok, "steps": len(frames),
                   "track_rms_m": rms, "grasp_idx": ref.grasp_idx, "release_idx": ref.release_idx}
            log.append(row)
            print(f"{tp.stem:28s} init={init_id:2d} success={ok!s:5s} steps={len(frames):3d} rms={rms*100:.1f}cm")
            if ds is not None and (ok or cfg["keep_failures"]):
                for obs, act in frames:
                    ds.add_frame({"observation.images.image": obs.image,
                                  "observation.images.image2": obs.image2,
                                  "observation.state": obs.state,
                                  "action": np.asarray(act, np.float32),
                                  "task": scene.language})
                ds.save_episode()
            if init_id == 0:
                imageio.mimsave(a.previews / f"{tp.stem}_init0_{'ok' if ok else 'fail'}.mp4",
                                [o.image for o, _ in frames[::2]], fps=cfg["hz"] // 2)
    if ds is not None:
        ds.finalize()
    (a.previews / "replay_log.json").write_text(json.dumps(log, indent=2))
    n_ok = sum(r.get("success", False) for r in log)
    print(f"\n{n_ok}/{len(log)} candidate episodes succeeded; "
          f"{'dataset at ' + str(a.out) if ds is not None else 'no dataset written'}")
    scene.close()


if __name__ == "__main__":
    main()
