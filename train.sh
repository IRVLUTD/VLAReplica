#!/bin/bash
###############################################################################
# Unified training script for VLAReplica_SFT_data — single-GPU or multi-GPU.
#
# Usage:
#   ./train.sh <policy> [num_gpus]
#     policy:   act | smolvla | pi0 | pi0_fast | pi05 | dit | flow_matching_dit | xvla
#     num_gpus: 1 = single GPU (default) | >1 = multi-GPU via accelerate
#
# Examples:
#   ./train.sh pi0                                 # single GPU
#   ./train.sh pi0 4                               # 4 GPUs (pi0 uses FSDP, needs fsdp_pi0.yaml)
#   CUDA_VISIBLE_DEVICES=0,1 ./train.sh smolvla 2
#
# Optional environment variables:
#   DATASET       dataset repo id on the Hugging Face Hub  (default: HenryZhang/VLAReplica_SFT_data)
#   OUTPUT_BASE   checkpoint/output directory              (default: ./VLAReplica_outputs)
#   WANDB         enable wandb logging                     (default: false; run `wandb login` first)
#   PUSH          push checkpoints to YOUR HF Hub          (default: false)
#   HF_USER       your HF username/org for pushed repo ids (no default; required if PUSH=true)
#   FSDP_CONFIG   accelerate FSDP yaml for multi-GPU pi0   (default: ./fsdp_pi0.yaml)
#
# Logging / pushing example:
#   WANDB=true PUSH=true HF_USER=<your_hf_username> ./train.sh act 2
#
# NOTE: single-GPU and multi-GPU keep the exact hyperparameters from the two
# original validated scripts, so the same policy may use different settings
# (chunk_size, batch_size, steps, ...) depending on the mode.
###############################################################################

set -e

POLICY=${1:-act}
NUM_GPUS=${2:-1}

case ${NUM_GPUS} in
  ''|*[!0-9]*) echo "num_gpus must be a positive integer, got: '${NUM_GPUS}'"; exit 1 ;;
esac

# ------------------------------ configuration -------------------------------
DATASET=${DATASET:-"HenryZhang/VLAReplica_SFT_data"}
OUTPUT_BASE=${OUTPUT_BASE:-"./VLAReplica_outputs"}
HF_USER=${HF_USER:-""}
PUSH=${PUSH:-false}
WANDB=${WANDB:-false}
FSDP_CONFIG=${FSDP_CONFIG:-"fsdp_pi0.yaml"}

if [ "${PUSH}" = "true" ] && [ -z "${HF_USER}" ]; then
  echo "ERROR: PUSH=true requires HF_USER=<your_hf_username_or_org> (used for --policy.repo_id)."
  exit 1
fi

# Emits push_to_hub / repo_id args for a given repo name.
hub_args() {
  if [ "${PUSH}" = "true" ]; then
    echo "--policy.push_to_hub=true --policy.repo_id=${HF_USER}/$1"
  else
    echo "--policy.push_to_hub=false"
  fi
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OUTPUT_BASE}"

LEROBOT_TRAIN=$(command -v lerobot-train || true)
if [ -z "${LEROBOT_TRAIN}" ]; then
  echo "ERROR: lerobot-train not found in PATH — activate your lerobot environment first."
  exit 1
fi

# -------------------------------- launcher ----------------------------------
if [ "${NUM_GPUS}" -gt 1 ]; then
  MODE="multi"
  LAUNCH="accelerate launch --num_processes=${NUM_GPUS} --multi_gpu ${LEROBOT_TRAIN}"
else
  MODE="single"
  LAUNCH="${LEROBOT_TRAIN}"
fi

echo "Policy: ${POLICY} | GPUs: ${NUM_GPUS} (${MODE}) | Output: ${OUTPUT_BASE}"
echo "Dataset: ${DATASET} | push_to_hub: ${PUSH} | wandb: ${WANDB}"

COMMON="--dataset.repo_id=${DATASET} --wandb.enable=${WANDB}"

case ${POLICY} in

###############################################################################
# 1. ACT (ResNet50 backbone)
###############################################################################
act)
  if [ "${MODE}" = "single" ]; then
    ${LAUNCH} ${COMMON} \
      --policy.type=act \
      --policy.chunk_size=16 \
      --policy.n_action_steps=16 \
      --policy.vision_backbone=resnet50 \
      --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V2 \
      --policy.optimizer_lr_backbone=1e-5 \
      --policy.n_encoder_layers=6 \
      --policy.use_amp=True \
      --policy.optimizer_lr=5e-5 \
      --policy.device=cuda \
      --output_dir=${OUTPUT_BASE}/0407_5task_resnet50_encoder_6_v4 \
      --batch_size=8 \
      --num_workers=10 \
      --save_freq=5000 \
      --eval_freq=5000 \
      $(hub_args VLAReplica_act)
  else
    ${LAUNCH} ${COMMON} \
      --policy.type=act \
      --policy.chunk_size=32 \
      --policy.n_action_steps=32 \
      --policy.vision_backbone=resnet50 \
      --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V2 \
      --policy.optimizer_lr_backbone=5e-6 \
      --policy.n_encoder_layers=4 \
      --policy.use_amp=True \
      --policy.optimizer_lr=5e-5 \
      --output_dir=${OUTPUT_BASE}/0426_5task_resnet50_encoder_4_v4_multigpu_chunk32 \
      --batch_size=64 \
      --steps=40000 \
      --num_workers=10 \
      --save_freq=1000 \
      --eval_freq=1000 \
      $(hub_args VLAReplica_act_v4_4layers_multigpu_chunk32)
  fi
  ;;

###############################################################################
# 2. SmolVLA (fine-tune from pretrained smolvla_base)
###############################################################################
smolvla)
  if [ "${MODE}" = "single" ]; then
    ${LAUNCH} ${COMMON} \
      --policy.path=lerobot/smolvla_base \
      --policy.chunk_size=16 \
      --policy.n_action_steps=16 \
      --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
      --policy.empty_cameras=1 \
      --policy.device=cuda \
      --output_dir=${OUTPUT_BASE}/0412_5task_smolvla_v4 \
      --job_name=smolvla_5task \
      --batch_size=64 \
      --steps=50000 \
      --num_workers=10 \
      --save_freq=1000 \
      --eval_freq=1000 \
      $(hub_args VLAReplica_smolvla_0412_v4data)
  else
    ${LAUNCH} ${COMMON} \
      --policy.path=lerobot/smolvla_base \
      --policy.chunk_size=32 \
      --policy.n_action_steps=32 \
      --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
      --policy.empty_cameras=1 \
      --output_dir=${OUTPUT_BASE}/0425_smolvla_sft_chunk32 \
      --job_name=smolvla_sft \
      --batch_size=32 \
      --steps=40000 \
      --num_workers=8 \
      --save_freq=1000 \
      --eval_freq=1000 \
      $(hub_args VLAReplica_smolvla_sft_chunk32)
  fi
  ;;

###############################################################################
# 3. Pi0 (single: DDP-free plain training | multi: FSDP via accelerate config)
###############################################################################
pi0)
  if [ "${MODE}" = "single" ]; then
    ${LAUNCH} ${COMMON} \
      --policy.type=pi0 \
      --policy.pretrained_path=lerobot/pi0_base \
      --policy.chunk_size=32 \
      --policy.n_action_steps=32 \
      --policy.compile_model=false \
      --policy.gradient_checkpointing=true \
      --policy.dtype=bfloat16 \
      --policy.freeze_vision_encoder=true \
      --policy.train_expert_only=false \
      --policy.device=cuda \
      --output_dir=${OUTPUT_BASE}/0426_5task_pi0_v4_chunk32 \
      --job_name=pi0_5task_v4_chunk32 \
      --batch_size=64 \
      --steps=40000 \
      --num_workers=16 \
      --save_freq=1000 \
      --eval_freq=1000 \
      $(hub_args VLAReplica_pi0_v4_chunk32)
  else
    if [ ! -f "${FSDP_CONFIG}" ]; then
      echo "ERROR: FSDP config '${FSDP_CONFIG}' not found (required for multi-GPU pi0)."
      echo "Copy your fsdp_pi0.yaml next to this script, or set FSDP_CONFIG=/path/to/yaml."
      exit 1
    fi
    # --num_processes on the CLI overrides the value inside the yaml.
    accelerate launch --config_file "${FSDP_CONFIG}" --num_processes=${NUM_GPUS} "${LEROBOT_TRAIN}" ${COMMON} \
      --policy.type=pi0 \
      --policy.pretrained_path=lerobot/pi0_base \
      --policy.chunk_size=16 \
      --policy.n_action_steps=16 \
      --policy.compile_model=false \
      --policy.gradient_checkpointing=true \
      --policy.dtype=bfloat16 \
      --policy.freeze_vision_encoder=true \
      --policy.train_expert_only=false \
      --output_dir=${OUTPUT_BASE}/0422_pi0_sft_fsdp \
      --job_name=pi0_sft_fsdp \
      --batch_size=4 \
      --steps=80000 \
      --num_workers=1 \
      --save_freq=500 \
      --eval_freq=500 \
      $(hub_args VLAReplica_pi0_sft_fsdp)
  fi
  ;;

###############################################################################
# 4. Pi0-FAST
#    NOTE: original scripts differ — single-GPU fine-tunes from lerobot/pi0fast-base,
#    multi-GPU from lerobot/pi0_base. Kept verbatim; double-check this is intended.
###############################################################################
pi0_fast)
  if [ "${MODE}" = "single" ]; then
    ${LAUNCH} ${COMMON} \
      --policy.type=pi0_fast \
      --policy.pretrained_path=lerobot/pi0fast-base \
      --policy.chunk_size=32 \
      --policy.n_action_steps=32 \
      --policy.compile_model=false \
      --policy.gradient_checkpointing=false \
      --policy.dtype=bfloat16 \
      --policy.max_action_tokens=256 \
      --policy.device=cuda \
      --output_dir=${OUTPUT_BASE}/0426_5task_pi0_fast_v4_chunk32_SFT \
      --job_name=pi0_fast_5task_v4_chunk32_SFT \
      --batch_size=8 \
      --steps=40000 \
      --num_workers=8 \
      --save_freq=1000 \
      --eval_freq=1000 \
      $(hub_args VLAReplica_pi0_fast_v4_chunk32_SFT)
  else
    ${LAUNCH} ${COMMON} \
      --policy.type=pi0_fast \
      --policy.pretrained_path=lerobot/pi0_base \
      --policy.chunk_size=16 \
      --policy.n_action_steps=16 \
      --policy.compile_model=true \
      --policy.gradient_checkpointing=true \
      --policy.dtype=bfloat16 \
      --output_dir=${OUTPUT_BASE}/0409_5task_pi0_fast_v4_multigpu \
      --job_name=pi0_fast_5task \
      --batch_size=32 \
      --steps=3000 \
      --num_workers=10 \
      --save_freq=1000 \
      --eval_freq=1000 \
      $(hub_args VLAReplica_pi0_fast_v4)
  fi
  ;;

###############################################################################
# 5. Pi0.5 (fine-tune Pi0.5 with quantile normalization)
###############################################################################
pi05)
  if [ "${MODE}" = "single" ]; then
    ${LAUNCH} ${COMMON} \
      --policy.type=pi05 \
      --policy.pretrained_path=lerobot/pi05_base \
      --policy.chunk_size=32 \
      --policy.n_action_steps=32 \
      --policy.compile_model=false \
      --policy.gradient_checkpointing=false \
      --policy.dtype=bfloat16 \
      --policy.freeze_vision_encoder=true \
      --policy.train_expert_only=false \
      --policy.device=cuda \
      --output_dir=${OUTPUT_BASE}/0426_5task_pi05_v4_chunk32_SFT \
      --job_name=pi05_5task_v4_chunk32_SFT \
      --batch_size=4 \
      --steps=40000 \
      --num_workers=8 \
      --save_freq=1000 \
      --eval_freq=1000 \
      $(hub_args VLAReplica_pi05_v4_chunk32_SFT)
  else
    ${LAUNCH} ${COMMON} \
      --policy.type=pi05 \
      --policy.pretrained_path=lerobot/pi05_base \
      --policy.chunk_size=16 \
      --policy.n_action_steps=16 \
      --policy.compile_model=false \
      --policy.gradient_checkpointing=true \
      --policy.dtype=bfloat16 \
      --policy.freeze_vision_encoder=true \
      --policy.train_expert_only=false \
      --output_dir=${OUTPUT_BASE}/0412_5task_pi05_v4_expert_multigpu \
      --job_name=pi05_5task \
      --batch_size=16 \
      --steps=20000 \
      --num_workers=10 \
      --save_freq=1000 \
      --eval_freq=1000 \
      $(hub_args VLAReplica_pi05_v4_expert)
  fi
  ;;

###############################################################################
# 6. Diffusion Transformer (multi-task DiT)
###############################################################################
dit)
  if [ "${MODE}" = "single" ]; then
    ${LAUNCH} ${COMMON} \
      --policy.type=multi_task_dit \
      --policy.device=cuda \
      --output_dir=${OUTPUT_BASE}/0409_5task_dit_v4 \
      --job_name=dit_5task_v4 \
      --batch_size=32 \
      --steps=5000 \
      --save_freq=500 \
      --log_freq=100 \
      --num_workers=10 \
      $(hub_args VLAReplica_dit)
  else
    ${LAUNCH} ${COMMON} \
      --policy.type=multi_task_dit \
      --policy.image_resize_shape='[256,256]' \
      --policy.image_crop_shape='[224,224]' \
      --policy.image_crop_is_random=true \
      --policy.num_layers=6 \
      --policy.hidden_dim=512 \
      --policy.num_heads=8 \
      --output_dir=${OUTPUT_BASE}/0426_5task_dit_v4_multigpu_sft_chunk32 \
      --job_name=dit_5task \
      --batch_size=64 \
      --steps=80000 \
      --save_freq=1000 \
      --log_freq=20 \
      --num_workers=8 \
      $(hub_args VLAReplica_dit_v4_multigpu_sft_chunk32)
  fi
  ;;

flow_matching_dit)
  if [ "${MODE}" = "single" ]; then
    ${LAUNCH} ${COMMON} \
      --policy.type=multi_task_dit \
      --policy.objective=flow_matching \
      --policy.timestep_sampling_strategy=beta \
      --policy.num_integration_steps=100 \
      --policy.integration_method=euler \
      --policy.sigma_min=0.0 \
      --policy.device=cuda \
      --output_dir=${OUTPUT_BASE}/0422_flow_matching_dit \
      --job_name=flow_matching_dit_5task_v4 \
      --batch_size=64 \
      --steps=5000 \
      --save_freq=500 \
      --log_freq=100 \
      --num_workers=10 \
      $(hub_args VLAReplica_flow_matching_dit)
  else
    ${LAUNCH} ${COMMON} \
      --policy.type=multi_task_dit \
      --policy.image_resize_shape='[256,256]' \
      --policy.image_crop_shape='[224,224]' \
      --policy.image_crop_is_random=true \
      --policy.num_layers=6 \
      --policy.hidden_dim=512 \
      --policy.num_heads=8 \
      --policy.objective=flow_matching \
      --policy.timestep_sampling_strategy=beta \
      --policy.num_integration_steps=100 \
      --policy.integration_method=euler \
      --policy.timestep_sampling_alpha=1.5 \
      --policy.timestep_sampling_beta=1.0 \
      --policy.timestep_sampling_s=0.999 \
      --policy.sigma_min=0.0 \
      --output_dir=${OUTPUT_BASE}/0422_flow_matching_dit_multigpu \
      --job_name=flow_matching_dit_5task \
      --batch_size=64 \
      --steps=80000 \
      --save_freq=1000 \
      --log_freq=20 \
      --num_workers=8 \
      $(hub_args VLAReplica_flow_matching_dit)
  fi
  ;;

###############################################################################
# 7. X-VLA (soft-prompted flow-matching VLA, fine-tune from xvla-base)
###############################################################################
xvla)
  if [ "${MODE}" = "single" ]; then
    ${LAUNCH} ${COMMON} \
      --policy.path=lerobot/xvla-base \
      --policy.dtype=bfloat16 \
      --policy.action_mode=auto \
      --policy.chunk_size=32 \
      --policy.n_action_steps=32 \
      --policy.freeze_vision_encoder=false \
      --policy.freeze_language_encoder=false \
      --policy.train_policy_transformer=true \
      --policy.train_soft_prompts=true \
      --policy.device=cuda \
      --output_dir=${OUTPUT_BASE}/0409_5task_xvla_v4 \
      --job_name=xvla_5task_v4 \
      --batch_size=16 \
      --steps=20000 \
      --num_workers=10 \
      --save_freq=1000 \
      --eval_freq=1000 \
      $(hub_args VLAReplica_xvla)
  else
    ${LAUNCH} ${COMMON} \
      --policy.path=lerobot/xvla-base \
      --policy.dtype=bfloat16 \
      --policy.action_mode=auto \
      --policy.chunk_size=16 \
      --policy.n_action_steps=16 \
      --policy.freeze_vision_encoder=false \
      --policy.freeze_language_encoder=false \
      --policy.train_policy_transformer=true \
      --policy.train_soft_prompts=true \
      --rename_map='{"observation.images.top": "observation.images.image", "observation.images.wrist": "observation.images.image2"}' \
      --output_dir=${OUTPUT_BASE}/0409_5task_xvla_v4_multigpu \
      --job_name=xvla_5task \
      --batch_size=32 \
      --steps=80000 \
      --num_workers=10 \
      --save_freq=1000 \
      --eval_freq=1000 \
      $(hub_args VLAReplica_xvla)
  fi
  ;;

*)
  echo "Unknown policy: ${POLICY}"
  echo "Usage: ./train.sh <act|smolvla|pi0|pi0_fast|pi05|dit|flow_matching_dit|xvla> [num_gpus]"
  exit 1
  ;;

###############################################################################
# 8. Gr00t (NVIDIA Isaac GR00T foundation model)
###############################################################################
groot)
  if [ "${MODE}" = "single" ]; then
    ${LAUNCH} ${COMMON} \
      --policy.type=groot \
      --policy.tune_diffusion_model=false \
      --output_dir=${OUTPUT_BASE}/0428_5task_groot_v4 \
      --job_name=groot_5task_v4 \
      --batch_size=16 \
      --steps=40000 \
      --num_workers=10 \
      --save_freq=5000 \
      --env_eval_freq=2500 \
      $(hub_args VLAReplica_groot)
  else
    ${LAUNCH} ${COMMON} \
      --policy.type=groot \
      --policy.tune_diffusion_model=false \
      --output_dir=${OUTPUT_BASE}/0428_5task_groot_v4_multigpu \
      --job_name=groot_5task \
      --batch_size=32 \
      --steps=40000 \
      --num_workers=10 \
      --save_freq=5000 \
      --env_eval_freq=2500 \
      $(hub_args VLAReplica_groot_v4_multigpu)
  fi
  ;;

*)
  echo "Unknown policy: ${POLICY}"
  echo "Usage: ./train.sh <act|smolvla|pi0|pi0_fast|pi05|dit|flow_matching_dit|xvla> [num_gpus]"
  exit 1
  ;;

esac
