#!/usr/bin/env bash
# One-time per machine: write LIBERO's config non-interactively and check EGL rendering.
#   bash scripts/spark_bootstrap.sh
set -euo pipefail
source .venv/bin/activate
python - <<'PY'
import importlib.util, os, pathlib, yaml
spec = importlib.util.find_spec("libero")
root = pathlib.Path(spec.origin).parent / "libero"
cfg_dir = pathlib.Path.home() / ".libero"; cfg_dir.mkdir(exist_ok=True)
cfg = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(cfg_dir / "datasets"),
    "assets": str(root / "assets"),
}
(cfg_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
print("libero config:", cfg)
for k, v in cfg.items():
    print(f"  {k:15s} exists={os.path.exists(v)}")
PY
MUJOCO_GL=egl python - <<'PY'
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos='0 0 1'/><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d)
img = r.render(); print("EGL render ok, mean pixel", float(img.mean()))
PY
