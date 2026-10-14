# kinesis2 — phone hand video -> LIBERO Panda
PY ?= uv run python

setup:            ## create .venv and install (CPU/Mac: vision stack only)
	uv venv --python 3.11 .venv
	uv pip install -e ".[dev]"

setup-sim:        ## full stack incl. lerobot + LIBERO (GPU box)
	uv venv --python 3.11 .venv
	uv pip install -e ".[dev,sim]"

test:
	uv run pytest -q

hands:            ## phone videos in data/raw -> hand tracks in data/hands
	$(PY) scripts/extract_hands.py --in data/raw --out data/hands

retarget:         ## hand tracks -> Panda EE actions, replayed in LIBERO -> lerobot dataset
	$(PY) scripts/retarget_replay.py --hands data/hands --out data/lerobot/ego_bowl

train:
	$(PY) scripts/train.py --config configs/smolvla_ego.yaml

eval:
	$(PY) scripts/eval.py --config configs/smolvla_ego.yaml

.PHONY: setup setup-sim test hands retarget train eval
