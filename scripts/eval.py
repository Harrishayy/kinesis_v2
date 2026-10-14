"""Evaluate a LeRobot policy (SmolVLA by default) in LIBERO with optional perturbations.

    MUJOCO_GL=egl uv run python scripts/eval.py --policy HuggingFaceVLA/smolvla_libero \
        --suite libero_spatial --task-id 2 --n-episodes 20 --tag baseline
    ... --delay 2 --state-noise 0.01 --brightness 0.3 --tag delay2_noise1cm

Writes results/eval/<tag>.json (success rate, steps, per-step policy latency) and a few mp4s.
Rolls out with our own LiberoScene so the same loop serves the retarget replay, the
robustness sweep and the latency measurements; observation tensors are built the same
way lerobot's eval does (uint8 HWC -> float CHW in [0,1], batch dim 1, HF-libero flip).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from kinesis2.perturb import ActionDelay, ImageShift, StateNoise
from kinesis2.sim import LiberoScene

NOOP = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)


def load_policy(path: str, suite: str, task_id: int, image_size: int, device: str, dtype: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    cfg.device = device
    env_cfg = LiberoEnv(task=suite, task_ids=[task_id], observation_height=image_size,
                        observation_width=image_size)
    policy = make_policy(cfg=cfg, env_cfg=env_cfg)
    policy.eval()
    if dtype == "bf16":
        policy = policy.to(torch.bfloat16)
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def to_obs(step, language: str, device: str) -> dict:
    def img(x):
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return {
        "observation.images.image": img(step.image),
        "observation.images.image2": img(step.image2),
        "observation.state": torch.from_numpy(step.state).float().unsqueeze(0),
        "task": [language],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=2)
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--init-offset", type=int, default=0, help="first LIBERO init state to use")
    ap.add_argument("--max-steps", type=int, default=280)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--state-noise", type=float, default=0.0, help="sigma on eef position (m)")
    ap.add_argument("--brightness", type=float, default=0.0)
    ap.add_argument("--shift-px", type=int, default=0)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out", type=Path, default=Path("results/eval"))
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    policy, pre, post = load_policy(a.policy, a.suite, a.task_id, a.image_size, a.device, a.dtype)
    scene = LiberoScene(a.suite, a.task_id, a.image_size, 20, flip_images=True)
    print("task:", scene.language, "| policy:", a.policy, "| tag:", a.tag)
    a.out.mkdir(parents=True, exist_ok=True)

    delay = ActionDelay(a.delay, NOOP)
    snoise = StateNoise(a.state_noise, seed=a.seed)
    ishift = ImageShift(a.brightness, a.shift_px, seed=a.seed)
    rows, lat = [], []
    for ep in range(a.n_episodes):
        obs = scene.reset(a.init_offset + ep)
        policy.reset(); delay.reset()
        frames, ok = [], False
        for k in range(a.max_steps):
            if a.state_noise > 0:
                obs.state = snoise(obs.state)
            if a.brightness > 0 or a.shift_px > 0:
                obs.image, obs.image2 = ishift(obs.image), ishift(obs.image2)
            o = pre(to_obs(obs, scene.language, a.device))
            t0 = time.perf_counter()
            with torch.inference_mode():
                act = policy.select_action(o)
            if a.device == "cuda":
                torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
            act = post(act).to("cpu").float().numpy().reshape(-1)[:7]
            act = delay(act)
            if ep < a.videos:
                frames.append(obs.image)
            obs, ok = scene.step(act)
            if ok:
                break
        rows.append({"episode": ep, "success": ok, "steps": k + 1})
        print(f"ep {ep:2d} success={ok!s:5s} steps={k+1:3d} "
              f"policy_ms(median so far)={1000*np.median(lat):.1f}")
        if ep < a.videos and frames:
            imageio.mimsave(a.out / f"{a.tag}_ep{ep}_{'ok' if ok else 'fail'}.mp4", frames[::2], fps=10)
    lat_ms = 1000 * np.array(lat)
    summary = {
        "tag": a.tag, "policy": a.policy, "suite": a.suite, "task_id": a.task_id,
        "n_episodes": a.n_episodes, "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "perturb": {"delay": a.delay, "state_noise": a.state_noise, "brightness": a.brightness, "shift_px": a.shift_px},
        "dtype": a.dtype, "device": a.device,
        "policy_latency_ms": {"median": float(np.median(lat_ms)), "p90": float(np.percentile(lat_ms, 90)),
                              "mean": float(lat_ms.mean())},
        "episodes": rows,
    }
    (a.out / f"{a.tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{a.tag}: success {summary['success_rate']:.2f} over {a.n_episodes} eps, "
          f"policy median {summary['policy_latency_ms']['median']:.1f} ms")
    scene.close()


if __name__ == "__main__":
    main()
