#!/usr/bin/env python3
"""Hardware-free inference smoke test for a fine-tuned MolmoAct2 checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        help="Hugging Face repository ID or local converted-checkpoint directory.",
    )
    parser.add_argument("--task", default="put the bread on the red plate.")
    parser.add_argument("--norm-tag", default="vlareplica_so101_v3")
    parser.add_argument("--state", default="0,0,0,0,0,0", help="Six comma-separated joint values.")
    parser.add_argument("--top-image", type=Path)
    parser.add_argument("--wrist-image", type=Path)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def parse_state(value: str):
    import numpy as np

    state = np.asarray([float(item.strip()) for item in value.split(",")], dtype=np.float32)
    if state.shape != (6,):
        raise ValueError(f"--state must contain exactly six values; received shape {state.shape}.")
    if not np.isfinite(state).all():
        raise ValueError("--state must contain only finite values.")
    return state


def load_image(path: Path | None, view: str):
    import numpy as np
    from PIL import Image

    if path is not None:
        return np.asarray(Image.open(path).convert("RGB"))

    height, width = 480, 640
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if view == "top":
        image[..., 0] = x
        image[..., 1] = y
        image[..., 2] = 96
    else:
        image[..., 0] = 64
        image[..., 1] = x
        image[..., 2] = y
    return image


def main() -> int:
    args = parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    state = parse_state(args.state)
    images = [load_image(args.top_image, "top"), load_image(args.wrist_image, "wrist")]

    print(f"Loading: {args.checkpoint}")
    print(f"Device: {device}; dtype: {args.dtype}; norm tag: {args.norm_tag}")
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        dtype=dtype,
    ).to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and dtype == torch.bfloat16,
    ):
        output = model.predict_action(
            processor=processor,
            images=images,
            task=args.task,
            state=state,
            norm_tag=args.norm_tag,
            inference_action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=args.num_steps,
            normalize_language=True,
            enable_cuda_graph=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    actions = output.actions
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().float().cpu().numpy()
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise RuntimeError(f"FAIL: expected action shape (chunk, 6), received {actions.shape}.")
    if not np.isfinite(actions).all():
        raise RuntimeError("FAIL: predicted actions contain NaN or infinite values.")

    result = {
        "status": "PASS",
        "checkpoint": args.checkpoint,
        "task": args.task,
        "used_synthetic_top_image": args.top_image is None,
        "used_synthetic_wrist_image": args.wrist_image is None,
        "action_shape": list(actions.shape),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "first_action": actions[0].tolist(),
        "inference_seconds": elapsed,
    }
    if device.type == "cuda":
        result["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
    print(json.dumps(result, indent=2))
    print("PASS: model loaded and produced a finite 6-DoF action chunk; no hardware was accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
