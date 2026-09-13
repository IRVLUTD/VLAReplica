# MolmoAct2 VLAReplica setup

MolmoAct2 is included as the `molmoact2` submodule. Its `vlareplica-finetune`
branch contains the VLAReplica dataset mixture, training launchers, validation
workflow, checkpoint conversion, and the nested LeRobot, YAM, and EVA_DROID
submodules.

## Clone VLAReplica and its dependencies

```bash
git clone --recurse-submodules https://github.com/IRVLUTD/VLAReplica.git
cd VLAReplica
git submodule update --init --recursive
```

The recursive update initializes MolmoAct2 and its nested submodules. Do not
commit model weights, datasets, Conda environments, caches, logs, or checkpoints.

## Install MolmoAct2

Choose a storage location with enough space for environments, model downloads,
datasets, and checkpoints:

```bash
export VLA_STORAGE=/path/to/vla-storage
mkdir -p "$VLA_STORAGE"/{conda-envs,conda-pkgs,tmp,pip-cache,huggingface,lerobot-data,molmo-data,molmoact2-checkpoints}

conda create --prefix "$VLA_STORAGE/conda-envs/molmoact2" python=3.12 -y
conda activate "$VLA_STORAGE/conda-envs/molmoact2"

cd molmoact2/experiments
TMPDIR="$VLA_STORAGE/tmp" PIP_CACHE_DIR="$VLA_STORAGE/pip-cache" \
  python -m pip install -e '.[all]'
```

Configure shared cache and dataset locations:

```bash
conda env config vars set \
  HF_HOME="$VLA_STORAGE/huggingface" \
  HF_HUB_CACHE="$VLA_STORAGE/huggingface/hub" \
  LEROBOT_DATA_ROOT="$VLA_STORAGE/lerobot-data" \
  MOLMO_DATA_DIR="$VLA_STORAGE/molmo-data" \
  TMPDIR="$VLA_STORAGE/tmp" \
  PIP_CACHE_DIR="$VLA_STORAGE/pip-cache" \
  LEROBOT_VIDEO_BACKEND=pyav
```

Reactivate the environment after setting those variables. See
`molmoact2/experiments/README.md` for the VLAReplica pilot, full fine-tuning,
checkpoint conversion, and open-loop evaluation commands.

## Hardware benchmark

Use the MolmoAct2 section in the top-level `README.md` to run the VLAReplica
hardware benchmark. Start with `--molmoact2-dry-run`, then enable hardware
actions only after checking the local camera, calibration, and action output.
