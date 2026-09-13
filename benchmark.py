#!/usr/bin/env python3
"""Unified SO101 evaluator for ACT/SmolVLA/Pi0/Pi0.5/Pi0-fast/DiT with teleop-first handoff,
top-camera zoom, OpenCV overlay visualization, and per-episode video recording.

Usage:
    python so101_unified_policy_evaluate_crop_camera.py \\
        --policy-type dit \\
        --policy-path outputs/train/.../pretrained_model \\
        --task "pick up the object" \\
        --iterations 5

Supports: act, smolvla, pi0, pi0_fast, pi05, dit, xvla, molmoact2
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty
import types
from typing import Any, Optional

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
try:
    from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
except Exception:
    try:
        from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
    except Exception:
        from lerobot.common.datasets.utils import build_dataset_frame
        from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
except Exception:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

try:
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
except Exception:
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
try:
    from lerobot.utils.device_utils import get_safe_torch_device
except Exception:
    from lerobot.utils.utils import get_safe_torch_device


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _patch_dit_multi_resolution() -> None:
    """Monkey-patch the DiT policy to handle multi-resolution cameras.

    Training uses image_resize_shape=[256,256] + image_crop_shape=[224,224] to unify
    cameras with different native resolutions (e.g. 720x1280 top vs 480x640 wrist).
    The stock _prepare_batch tries to torch.stack before resizing, which fails when
    cameras differ.  This patch:
      1) Resizes each camera image individually before stacking in _prepare_batch.
      2) Skips the raw-shape equality check in validate_features when resize is set.
    """
    try:
        from lerobot.policies.multi_task_dit import (
            modeling_multi_task_dit as dit_mod,
            configuration_multi_task_dit as dit_cfg,
        )
    except ImportError:
        return

    if getattr(dit_mod, "_multi_res_patch_installed", False):
        return

    from torch import Tensor
    from lerobot.utils.constants import OBS_IMAGES as _OBS_IMAGES

    def _prepare_batch_patched(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.image_features:
            batch = dict(batch)
            images = [batch[key] for key in self.config.image_features]
            if self.config.image_resize_shape is not None:
                target_h, target_w = self.config.image_resize_shape
                resized = []
                for img in images:
                    if img.shape[-2] != target_h or img.shape[-1] != target_w:
                        leading = img.shape[:-3]
                        img = img.reshape(-1, *img.shape[-3:])
                        img = torch.nn.functional.interpolate(
                            img, size=(target_h, target_w),
                            mode="bilinear", align_corners=False, antialias=True,
                        )
                        img = img.reshape(*leading, *img.shape[-3:])
                    resized.append(img)
                images = resized
            batch[_OBS_IMAGES] = torch.stack(images, dim=-4)
        return batch

    dit_mod.MultiTaskDiTPolicy._prepare_batch = _prepare_batch_patched

    import logging as _logging

    def _validate_features_patched(self) -> None:
        if self.image_crop_shape is not None:
            for key, image_ft in self.image_features.items():
                effective_h, effective_w = (
                    self.image_resize_shape
                    if self.image_resize_shape is not None
                    else (image_ft.shape[1], image_ft.shape[2])
                )
                if self.image_crop_shape[0] > effective_h or self.image_crop_shape[1] > effective_w:
                    _logging.warning(
                        "image_crop_shape %s doesn't fit within effective image shape (%s, %s) for '%s'; disabling cropping.",
                        self.image_crop_shape, effective_h, effective_w, key,
                    )
                    self.image_crop_shape = None
                    break

        if len(self.image_features) > 0 and self.image_resize_shape is None:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_ft.shape:
                    raise ValueError(
                        f"Image '{key}' shape {image_ft.shape} != '{first_key}' shape {first_ft.shape}"
                    )

    dit_cfg.MultiTaskDiTConfig.validate_features = _validate_features_patched

    dit_mod._multi_res_patch_installed = True
    print("[PATCH] Applied DiT multi-resolution camera patch (_prepare_batch + validate_features)")


POLICY_REGISTRY = {
    "act": {
        "default_path": "outputs/train/0406Task7.3/checkpoints/025000/pretrained_model",
        "default_robot_type": "so101_follower",
    },
    "smolvla": {
        "module": "lerobot.policies.smolvla.modeling_smolvla",
        "class": "SmolVLAPolicy",
        "default_path": "lerobot/smolvla_base",
        "default_robot_type": "so101_follower",
    },
    "pi0": {
        "module": "lerobot.policies.pi0.modeling_pi0",
        "class": "PI0Policy",
        "default_path": "lerobot/pi0",
        "default_robot_type": "so101_follower",
    },
    "pi05": {
        "module": "lerobot.policies.pi05.modeling_pi05",
        "class": "PI05Policy",
        "default_path": "lerobot/pi05",
        "default_robot_type": "so101_follower",
    },
    "pi0_fast": {
        "module": "lerobot.policies.pi0_fast.modeling_pi0_fast",
        "class": "PI0FastPolicy",
        "module_candidates": [
            "lerobot.policies.pi0_fast.modeling_pi0_fast",
            "lerobot.policies.pi0fast.modeling_pi0fast",
            "lerobot.policies.pi0_fast.modeling_pi0fast",
            "lerobot.policies.pi0fast.modeling_pi0_fast",
        ],
        "class_candidates": [
            "PI0FastPolicy",
            "PI0FastModel",
            "Pi0FastPolicy",
            "Pi0FastModel",
        ],
        "default_path": "lerobot/pi0_fast",
        "default_robot_type": "so101_follower",
    },
    "dit": {
        "module": "lerobot.policies.multi_task_dit.modeling_multi_task_dit",
        "class": "MultiTaskDiTPolicy",
        "default_path": "HenryZhang/VLAReplica_dit_v4_expert",
        "default_robot_type": "so101_follower",
    },
    "xvla": {
        "module": "lerobot.policies.x_vla.modeling_x_vla",
        "class": "XVLAPolicy",
        "module_candidates": [
            "lerobot.policies.x_vla.modeling_x_vla",
            "lerobot.policies.x_vla.modeling_xvla",
            "lerobot.policies.xvla.modeling_x_vla",
            "lerobot.policies.xvla.modeling_xvla",
        ],
        "class_candidates": [
            "XVLAPolicy",
            "XVlaPolicy",
            "XVLAPolicy",
            "XVLAModel",
        ],
        "default_path": "lerobot/x-vla",
        "default_robot_type": "so101_follower",
    },
    "molmoact2": {
        "default_path": "allenai/MolmoAct2-SO100_101",
        "default_robot_type": "so101_follower",
    },
}


class MolmoAct2PolicyAdapter:
    """Direct Transformers adapter for AllenAI's SO-100/SO-101 checkpoint.

    MolmoAct2 consumes two RGB images, an absolute six-joint robot state, and a
    language instruction. It returns a chunk of absolute actions in robot scale,
    so it intentionally bypasses LeRobot's normalized policy processors.
    """

    def __init__(
        self,
        model,
        processor,
        device: torch.device,
        dtype: torch.dtype,
        norm_tag: str,
        num_steps: int,
        actions_per_chunk: int,
        enable_cuda_graph: bool,
        max_joint_step_deg: float,
        dry_run: bool,
        joint_signs: np.ndarray,
        joint_offsets: np.ndarray,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.norm_tag = norm_tag
        self.num_steps = num_steps
        self.actions_per_chunk = actions_per_chunk
        self.enable_cuda_graph = enable_cuda_graph
        self.max_joint_step_deg = max_joint_step_deg
        self.dry_run = dry_run
        self.joint_signs = np.asarray(joint_signs, dtype=np.float32)
        self.joint_offsets = np.asarray(joint_offsets, dtype=np.float32)
        self._action_queue: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self.last_debug: dict[str, Any] = {}

    def reset(self) -> None:
        self._action_queue.clear()
        self.last_debug = {}

    def _predict_chunk(self, images: list[np.ndarray], task: str, state: np.ndarray) -> None:
        if state.shape != self.joint_signs.shape or state.shape != self.joint_offsets.shape:
            raise ValueError(
                f"MolmoAct2 joint transform has shape {self.joint_signs.shape}, but state has shape {state.shape}."
            )
        # The released checkpoint uses LeRobot v2.1 (old SO-100 degree) joint
        # coordinates, while current SO-101 hardware reports v3 coordinates.
        model_state = self.joint_signs * state + self.joint_offsets
        autocast_enabled = self.device.type == "cuda" and self.dtype == torch.bfloat16
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=task,
                state=model_state,
                norm_tag=self.norm_tag,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=self.num_steps,
                normalize_language=True,
                enable_cuda_graph=self.enable_cuda_graph,
            )

        raw_actions = output.actions
        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.detach().float().cpu().numpy()
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != state.shape[0]:
            raise ValueError(
                f"MolmoAct2 returned action shape {actions.shape}; expected (chunk, {state.shape[0]})."
            )
        if not np.isfinite(actions).all():
            raise ValueError("MolmoAct2 returned NaN or infinite action values.")

        count = min(self.actions_per_chunk, actions.shape[0])
        self._action_queue.extend((action, model_state.copy()) for action in actions[:count])
        self.last_debug = {
            "new_chunk": True,
            "predicted_chunk": actions.copy(),
            "executed_chunk_size": count,
        }

    def predict_action(
        self,
        images: list[np.ndarray],
        task: str,
        state: np.ndarray,
    ) -> np.ndarray:
        requested_new_chunk = not self._action_queue
        if requested_new_chunk:
            self._predict_chunk(images=images, task=task, state=state)

        model_action, model_state = self._action_queue.popleft()
        model_action = model_action.copy()
        # Inverse of model_state = signs * arm_state + offsets. Signs are +/-1.
        arm_target_unclamped = self.joint_signs * (model_action - self.joint_offsets)
        action = arm_target_unclamped.copy()
        if self.max_joint_step_deg > 0:
            max_step = float(self.max_joint_step_deg)
            action = np.clip(action, state - max_step, state + max_step)
        chunk_debug = self.last_debug if requested_new_chunk else {}
        self.last_debug = {
            **chunk_debug,
            "new_chunk": requested_new_chunk,
            "arm_state": state.copy(),
            "model_state": model_state,
            "model_action": model_action,
            "arm_target_unclamped": arm_target_unclamped,
            "arm_action": action.copy(),
            "arm_delta": action - state,
            "clipped": ~np.isclose(action, arm_target_unclamped),
            "queued_actions_remaining": len(self._action_queue),
        }
        return action.astype(np.float32, copy=False)


def load_molmoact2_policy(
    policy_path: str,
    device: torch.device,
    revision: str | None,
    dtype_name: str,
    norm_tag: str,
    num_steps: int,
    actions_per_chunk: int,
    enable_cuda_graph: bool,
    max_joint_step_deg: float,
    dry_run: bool,
    joint_signs: np.ndarray,
    joint_offsets: np.ndarray,
) -> MolmoAct2PolicyAdapter:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if device.type != "cuda":
        raise RuntimeError("MolmoAct2 real-time inference requires a CUDA device.")

    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    print(f"[INIT] Loading MolmoAct2 from {policy_path} ({dtype_name})")
    processor = AutoProcessor.from_pretrained(
        policy_path,
        revision=revision,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        policy_path,
        revision=revision,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device).eval()
    return MolmoAct2PolicyAdapter(
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
        norm_tag=norm_tag,
        num_steps=num_steps,
        actions_per_chunk=actions_per_chunk,
        enable_cuda_graph=enable_cuda_graph,
        max_joint_step_deg=max_joint_step_deg,
        dry_run=dry_run,
        joint_signs=joint_signs,
        joint_offsets=joint_offsets,
    )


def parse_molmoact2_joint_transform(value: str, flag_name: str) -> np.ndarray:
    """Parse a six-value SO-100/SO-101 joint-frame transform option."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise ValueError(f"{flag_name} must contain exactly 6 comma-separated values; got {len(parts)}.")
    try:
        result = np.asarray([float(part) for part in parts], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"{flag_name} contains a non-numeric value: {value!r}") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{flag_name} values must all be finite.")
    return result


def can_open_cv_preview_window() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def show_setup_preview(window_name: str, image_rgb: np.ndarray) -> bool:
    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]
        
        # Create resizable window (don't use WINDOW_AUTOSIZE which is non-resizable)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Make the window larger by default (2x up to a max size)
        scale = 2
        max_w, max_h = 1600, 1000
        new_w = min(max_w, w * scale)
        new_h = min(max_h, h * scale)
        cv2.resizeWindow(window_name, int(new_w), int(new_h))
        
        # Position window at leftmost monitor (0, 0)
        cv2.moveWindow(window_name, 0, 0)
        
        # Try to set window to always-on-top (may not work on all platforms)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass  # Property setting may not be supported on this platform
        
        # Display image (does not steal focus by default with WINDOW_NORMAL)
        cv2.imshow(window_name, image_bgr)
        cv2.waitKey(1)
        return True
    except Exception as exc:
        print(f"[WARN] Setup preview window disabled: {exc}")
        return False


def close_setup_preview(window_name: str) -> None:
    try:
        cv2.destroyWindow(window_name)
    except Exception:
        pass


class TerminalKeyReader:
    """Non-blocking key reader for terminal arrows/enter/q."""

    def __init__(self):
        self._enabled = sys.stdin.isatty()
        self._fd = None
        self._old = None
        self._buffer = b""

    def __enter__(self):
        if self._enabled:
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._enabled and self._fd is not None and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def get_key(self) -> str | None:
        if not self._enabled or self._fd is None:
            return None

        ready, _, _ = select.select([self._fd], [], [], 0)
        if not ready:
            return None

        chunk = os.read(self._fd, 32)
        if not chunk:
            return None

        self._buffer += chunk

        if b"\x1b[C" in self._buffer:
            self._buffer = self._buffer.replace(b"\x1b[C", b"", 1)
            return "RIGHT"

        if b"\n" in self._buffer or b"\r" in self._buffer:
            self._buffer = self._buffer.replace(b"\n", b"", 1).replace(b"\r", b"", 1)
            return "ENTER"

        for c in (b"q", b"Q"):
            if c in self._buffer:
                self._buffer = self._buffer.replace(c, b"", 1)
                return "QUIT"

        if len(self._buffer) > 128:
            self._buffer = self._buffer[-32:]
        return None


def zoom_frame_center(frame: np.ndarray, zoom_factor: float = 1.5) -> np.ndarray:
    """Center-crop zoom then resize back to original size."""
    if frame is None or zoom_factor <= 1.0:
        return frame

    h, w = frame.shape[:2]
    crop_w = max(1, int(w / zoom_factor))
    crop_h = max(1, int(h / zoom_factor))
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


class ZoomRobot:
    """Wraps a robot and zooms top-camera observations before the policy sees them."""

    def __init__(self, robot: SO101Follower, zoom_factor: float = 1.5):
        self.robot = robot
        self.zoom_factor = zoom_factor

    def __getattr__(self, name: str) -> Any:
        return getattr(self.robot, name)

    def connect(self):
        return self.robot.connect()

    def disconnect(self):
        return self.robot.disconnect()

    def send_action(self, action):
        return self.robot.send_action(action)

    def get_observation(self):
        obs = self.robot.get_observation()
        return self._zoom_top_camera(obs)

    def _zoom_top_camera(self, obj: Any):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                key = str(k).lower()
                if isinstance(v, np.ndarray) and v.ndim >= 2 and ("top" in key or "camera1" in key):
                    out[k] = zoom_frame_center(v, self.zoom_factor)
                elif isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[0] == 480 and v.shape[1] == 640:
                    out[k] = zoom_frame_center(v, self.zoom_factor)
                else:
                    out[k] = self._zoom_top_camera(v)
            return out

        if isinstance(obj, list):
            return [self._zoom_top_camera(v) for v in obj]

        if isinstance(obj, tuple):
            return tuple(self._zoom_top_camera(v) for v in obj)

        return obj


class MultiRobot:
    """Fan-out actions to multiple robots while using one robot for observations."""

    def __init__(self, robots: list[Any], primary_idx: int = 0):
        if not robots:
            raise ValueError("MultiRobot requires at least one robot")
        if primary_idx < 0 or primary_idx >= len(robots):
            raise ValueError("primary_idx out of range")
        self.robots = robots
        self.primary_idx = primary_idx

    @property
    def primary(self):
        return self.robots[self.primary_idx]

    def connect(self):
        for r in self.robots:
            r.connect()

    def disconnect(self):
        for r in self.robots:
            r.disconnect()

    def send_action(self, action):
        for r in self.robots:
            r.send_action(action)

    def get_observation(self):
        return self.primary.get_observation()


class FakeRobot:
    """Minimal fake robot wrapper that avoids hardware access for dry-run/testing.

    It mirrors the small subset of the SO101Follower API used by the evaluator:
    `connect()`, `disconnect()`, `send_action()`, `get_observation()`, and
    `action_features` / `observation_features` attributes.
    """

    def __init__(self, real_robot: Any, top_shape: tuple[int, int], wrist_shape: tuple[int, int], action_names: list[str]):
        self._real = real_robot
        # Preserve feature metadata so downstream code can build dataset frames
        self.action_features = getattr(real_robot, "action_features", {})
        self.observation_features = getattr(real_robot, "observation_features", {})
        self.id = getattr(real_robot, "id", "fake_robot")
        self._top_h, self._top_w = top_shape
        self._wrist_h, self._wrist_w = wrist_shape
        self._action_names = list(action_names) if action_names is not None else []

    def connect(self):
        print(f"[FAKE] Skipping hardware connect for {self.id}")

    def disconnect(self):
        print(f"[FAKE] Skipping hardware disconnect for {self.id}")

    def send_action(self, action):
        # No-op: intentionally do not forward commands to hardware
        return None

    def get_observation(self):
        # Return synthetic top/wrist RGB images and a zero robot state vector.
        top = np.full((self._top_h, self._top_w, 3), 128, dtype=np.uint8)
        wrist = np.full((self._wrist_h, self._wrist_w, 3), 128, dtype=np.uint8)
        # Provide state vector fallback under 'state' key if action names are known
        state = np.zeros(len(self._action_names), dtype=np.float32)
        return {
            "observation": {
                "top_camera": top,
                "wrist_camera": wrist,
                "state": state,
            },
            "state": state,
        }


def hw_to_dataset_features_compat(hw_feats, prefix: str):
    try:
        return hw_to_dataset_features(hw_feats, prefix, use_video=True)
    except TypeError:
        try:
            return hw_to_dataset_features(hw_feats, prefix, use_videos=True)
        except TypeError:
            try:
                return hw_to_dataset_features(hw_feats, prefix, use_images=True)
            except TypeError:
                return hw_to_dataset_features(hw_feats, prefix)


def _ensure_pi0_transformers_compat() -> None:
    module_name = "transformers.models.siglip.check"

    if module_name in sys.modules:
        return

    import importlib

    try:
        importlib.import_module(module_name)
        return
    except Exception:
        pass

    shim = types.ModuleType("check")

    def _compat_ok() -> bool:
        return True

    shim.check_whether_transformers_replace_is_installed_correctly = _compat_ok
    sys.modules[module_name] = shim

    try:
        siglip_pkg = importlib.import_module("transformers.models.siglip")
        setattr(siglip_pkg, "check", shim)
    except Exception:
        pass

    print("[WARN] Applied PI0 compatibility shim for transformers.models.siglip.check")


def _patch_gemma_attention_mask_alignment() -> None:
    try:
        import transformers.models.gemma.modeling_gemma as modeling_gemma
    except Exception:
        return

    if getattr(modeling_gemma, "_lerobot_mask_patch_installed", False):
        return

    original_fn = modeling_gemma.eager_attention_forward

    def _patched_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_gemma.repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            key_len = key_states.shape[-2]
            mask_len = causal_mask.shape[-1]

            if mask_len < key_len:
                causal_mask = torch.nn.functional.pad(causal_mask, (key_len - mask_len, 0), value=0.0)
            elif mask_len > key_len:
                causal_mask = causal_mask[..., -key_len:]

            attn_weights = attn_weights + causal_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    modeling_gemma.eager_attention_forward = _patched_eager_attention_forward
    modeling_gemma._lerobot_mask_patch_installed = True
    modeling_gemma._lerobot_mask_patch_original = original_fn
    print("[WARN] Applied Gemma attention mask alignment patch for PI0 compatibility")


def _patch_pi0_embed_image_dtype(policy) -> None:
    try:
        pi0_core = policy.model
        pge = pi0_core.paligemma_with_expert
        original_embed_image = pge.embed_image
    except Exception:
        return

    if getattr(pge, "_dtype_patch_installed", False):
        return

    def _embed_image_float32(image: torch.Tensor):
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        return original_embed_image(image)

    pge.embed_image = _embed_image_float32
    pge._dtype_patch_installed = True
    print("[WARN] Applied PI0 float32 image patch for SigLIP compatibility")


def _force_pi0_float32(policy) -> None:
    try:
        if hasattr(policy, "model"):
            policy.model.to(torch.float32)
        policy.to(torch.float32)
        print("[WARN] Forced PI0 policy weights to float32 for compatibility")
    except Exception as exc:
        print(f"[WARN] Could not force PI0 float32 precision: {exc}")


def resolve_policy_class(entry: dict[str, Any]):
    import importlib

    module_candidates = list(entry.get("module_candidates", [])) or [entry["module"]]
    class_candidates = list(entry.get("class_candidates", [])) or [entry["class"]]

    first_import_error = None
    last_attr_error = None

    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            if first_import_error is None:
                first_import_error = exc
            continue

        for class_name in class_candidates:
            if hasattr(module, class_name):
                return getattr(module, class_name), module_name, class_name
            last_attr_error = AttributeError(f"{module_name} has no attribute {class_name}")

    if first_import_error is not None and last_attr_error is None:
        raise first_import_error
    if last_attr_error is not None:
        raise last_attr_error
    raise ImportError("Unable to resolve policy module/class")


def load_act_policy(
    policy_path: str,
    device: str,
    from_hub: bool = False,
    revision: str | None = None,
):
    if from_hub:
        config_path = hf_hub_download(repo_id=policy_path, filename="config.json", revision=revision)
    else:
        config_path = Path(policy_path) / "config.json"

    raw_cfg = json.loads(Path(config_path).read_text())
    raw_cfg.pop("type", None)
    raw_cfg.pop("pretrained_path", None)

    raw_cfg["input_features"] = {
        k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
        for k, v in raw_cfg["input_features"].items()
    }
    raw_cfg["output_features"] = {
        k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
        for k, v in raw_cfg["output_features"].items()
    }
    raw_cfg["normalization_mapping"] = {
        k: NormalizationMode[v] for k, v in raw_cfg["normalization_mapping"].items()
    }
    raw_cfg["device"] = device

    cfg = ACTConfig(**raw_cfg)
    policy = ACTPolicy.from_pretrained(policy_path, config=cfg, revision=revision).to(device)
    policy.eval()
    return policy


def _load_local_policy_config(config_class, config_path: Path, revision: str | None = None):
    """Load a policy config from a local config.json without HF repo-id parsing."""
    raw_cfg = json.loads(config_path.read_text())
    raw_cfg.pop("type", None)
    raw_cfg.pop("pretrained_path", None)

    if "input_features" in raw_cfg and isinstance(raw_cfg["input_features"], dict):
        raw_cfg["input_features"] = {
            key: PolicyFeature(type=FeatureType[value["type"]], shape=tuple(value["shape"]))
            for key, value in raw_cfg["input_features"].items()
        }
    if "output_features" in raw_cfg and isinstance(raw_cfg["output_features"], dict):
        raw_cfg["output_features"] = {
            key: PolicyFeature(type=FeatureType[value["type"]], shape=tuple(value["shape"]))
            for key, value in raw_cfg["output_features"].items()
        }
    if "normalization_mapping" in raw_cfg and isinstance(raw_cfg["normalization_mapping"], dict):
        raw_cfg["normalization_mapping"] = {
            key: NormalizationMode[value] for key, value in raw_cfg["normalization_mapping"].items()
        }

    return config_class(**raw_cfg)


def load_policy(policy_type: str, policy_path: str, device: str, policy_from_hub: bool, revision: str | None = None):
    if policy_type == "act":
        print(f"[INIT] Loading ACT policy from: {policy_path}")
        return load_act_policy(
            policy_path=policy_path,
            device=device,
            from_hub=policy_from_hub,
            revision=revision,
        )

    entry = POLICY_REGISTRY[policy_type]
    is_pi_family = policy_type in {"pi0", "pi05", "pi0_fast"}

    if is_pi_family:
        _ensure_pi0_transformers_compat()
        _patch_gemma_attention_mask_alignment()

    if policy_type == "dit":
        _patch_dit_multi_resolution()

    cls, resolved_module, resolved_class = resolve_policy_class(entry)

    print(f"[INIT] Loading {resolved_class} ({resolved_module}) from: {policy_path}")
    kwargs = {"revision": revision}
    if is_pi_family:
        kwargs["strict"] = False

    policy_path_obj = Path(policy_path)
    if policy_path_obj.exists():
        config_class = getattr(cls, "config_class", None)
        if config_class is not None and hasattr(config_class, "from_pretrained"):
            print(f"[INIT] Loading local config from: {policy_path_obj}")
            config = _load_local_policy_config(config_class, policy_path_obj / "config.json", revision=revision)
            policy = cls.from_pretrained(
                policy_path_obj,
                config=config,
                local_files_only=True,
                **kwargs,
            ).to(device)
        else:
            policy = cls.from_pretrained(policy_path_obj, local_files_only=True, **kwargs).to(device)
    else:
        if policy_path.startswith("/"):
            raise FileNotFoundError(
                f"Local policy path does not exist: {policy_path}. If this is a Hugging Face repo id, use --policy-from-hub and pass the repo id instead."
            )
        policy = cls.from_pretrained(policy_path, **kwargs).to(device)

    if is_pi_family:
        _force_pi0_float32(policy)
        _patch_pi0_embed_image_dtype(policy)

    policy.eval()
    return policy


def _normalize_task_key(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def load_task_config(task_config_file: str | None) -> dict[str, Any]:
    """Load the task configuration used by the GUI and CLI convenience mode."""
    candidate_paths = []
    if task_config_file:
        candidate_paths.append(Path(task_config_file))
    candidate_paths.extend([Path("vla_tasks_gui_config.json"), Path("vla_tasks.json"), Path("vla_tasks_all.json")])

    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        with open(candidate, "r") as f:
            content = json.load(f)
        if isinstance(content, dict) and isinstance(content.get("tasks"), list):
            return content
        if isinstance(content, list):
            return {"tasks": [{"id": f"task_{i+1:02d}", "name": task_name} for i, task_name in enumerate(content)]}

    raise FileNotFoundError("No usable task config file found.")


def resolve_task_from_config(task_config: dict[str, Any], task_id: str) -> dict[str, Any]:
    """Resolve a task entry by id or normalized id."""
    tasks = task_config.get("tasks", [])
    target = _normalize_task_key(task_id)
    for task in tasks:
        candidate_ids = [task.get("id"), task.get("task_id"), task.get("name")]
        for candidate in candidate_ids:
            if _normalize_task_key(str(candidate)) == target:
                return task
    raise KeyError(f"Task id '{task_id}' not found in task config")


def resolve_task_variants_file(task: dict[str, Any]) -> Optional[Path]:
    """Resolve a task_variants JSON file for a task entry."""
    explicit_path = task.get("variants_file")
    if explicit_path:
        explicit = Path(explicit_path)
        if explicit.exists():
            return explicit

    task_variants_dir = Path("task_variants")
    task_id = str(task.get("id", ""))
    task_name = str(task.get("name", ""))
    candidates = []
    for raw_value in (task_id, _normalize_task_key(task_id), task_name, _normalize_task_key(task_name)):
        if raw_value:
            candidates.append(task_variants_dir / f"{raw_value}.json")

    seen = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.exists():
            return candidate
    return None


def to_action_vector(action_values, action_dim: int) -> torch.Tensor:
    """Normalize policy output to a single 1D action vector of length action_dim."""
    tensor = torch.as_tensor(action_values).detach().to("cpu")

    if tensor.ndim == 0:
        raise ValueError(f"Policy returned a scalar action ({tensor}). Expected {action_dim}D action vector.")

    if tensor.ndim == 1:
        if tensor.numel() != action_dim:
            raise ValueError(
                f"Policy returned 1D action with dim {tensor.numel()}, expected {action_dim}. Shape={tuple(tensor.shape)}"
            )
        return tensor

    if tensor.shape[-1] == action_dim:
        return tensor.reshape(-1, action_dim)[0]

    flat = tensor.flatten()
    if flat.numel() >= action_dim:
        return flat[:action_dim]

    raise ValueError(
        f"Could not parse policy action shape {tuple(tensor.shape)} into action_dim={action_dim}."
    )


def _iter_named_arrays(obj: Any, parent_key: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{parent_key}.{k}" if parent_key else str(k)
            yield from _iter_named_arrays(v, full_key)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            full_key = f"{parent_key}[{i}]"
            yield from _iter_named_arrays(v, full_key)
    elif isinstance(obj, np.ndarray):
        yield parent_key, obj


def to_hwc_uint8_image(value: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(value)
    if arr.ndim not in (2, 3):
        return None

    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    else:
        if arr.shape[2] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] != 3:
            return None

    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        if min_v >= 0.0 and max_v <= 1.0:
            arr = arr * 255.0
        elif min_v >= -1.0 and max_v <= 1.0:
            arr = (arr + 1.0) * 127.5
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def extract_top_camera_image(observation: dict[str, Any]) -> np.ndarray | None:
    candidates = []
    for key, value in _iter_named_arrays(observation):
        if value.ndim < 2:
            continue

        image = to_hwc_uint8_image(value)
        if image is None:
            continue

        key_l = key.lower()
        score = 0
        if "top" in key_l:
            score += 10
        if "camera1" in key_l:
            score += 8
        if "observation.images" in key_l:
            score += 3
        score += int(image.shape[0] * image.shape[1] / 100000)
        if image.shape[2] == 3:
            score += 3
        if float(np.std(image)) < 1.0:
            score -= 5
        candidates.append((score, key, image))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2]


def extract_molmoact2_inputs(
    observation: dict[str, Any],
    action_names: list[str],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract top/wrist RGB images and raw joint state in action-name order."""
    image_candidates: list[tuple[str, np.ndarray]] = []
    for key, value in _iter_named_arrays(observation):
        image = to_hwc_uint8_image(value)
        if image is not None:
            image_candidates.append((key.lower(), image))

    if len(image_candidates) < 2:
        raise ValueError(
            f"MolmoAct2 requires two camera images; found {len(image_candidates)} in observation keys."
        )

    def choose_camera(tokens: tuple[str, ...], excluded: set[int]) -> int | None:
        scored: list[tuple[int, int]] = []
        for idx, (key, image) in enumerate(image_candidates):
            if idx in excluded:
                continue
            score = sum(10 for token in tokens if token in key)
            score += int(image.shape[0] * image.shape[1] / 100000)
            scored.append((score, idx))
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][1]

    top_idx = choose_camera(("top", "camera1", "front", "overhead"), set())
    if top_idx is None:
        raise ValueError("Could not select a top/third-person camera for MolmoAct2.")
    wrist_idx = choose_camera(("wrist", "camera2", "side"), {top_idx})
    if wrist_idx is None:
        raise ValueError("Could not select a second camera for MolmoAct2.")

    state_values: list[float] = []
    missing: list[str] = []
    for name in action_names:
        value = observation.get(name)
        if value is None:
            value = observation.get(f"observation.{name}")
        if value is None:
            missing.append(name)
            continue
        arr = np.asarray(value)
        if arr.size != 1:
            raise ValueError(f"Joint state '{name}' must be scalar; got shape {arr.shape}.")
        state_values.append(float(arr.reshape(-1)[0]))

    if missing:
        state_vector = observation.get("observation.state", observation.get("state"))
        if state_vector is None:
            raise KeyError(f"MolmoAct2 could not find joint observation keys: {missing}")
        state = np.asarray(state_vector, dtype=np.float32).reshape(-1)
        if state.shape[0] != len(action_names):
            raise ValueError(
                f"Fallback robot state has {state.shape[0]} values; expected {len(action_names)}."
            )
    else:
        state = np.asarray(state_values, dtype=np.float32)

    if not np.isfinite(state).all():
        raise ValueError("Robot state contains NaN or infinite values.")
    return [image_candidates[top_idx][1], image_candidates[wrist_idx][1]], state


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_reference_pose_image(
    reference_pose_dir: str | None,
    iteration_idx: int,
    target_shape: tuple[int, int],
):
    if not reference_pose_dir:
        return None, None

    pose_path = Path(reference_pose_dir) / f"pic{iteration_idx}.jpg"

    # Fallbacks: accept common filenames (top.jpg, wrist.jpg, reference.jpg) or any image file in the folder
    if not pose_path.exists():
        ref_dir = Path(reference_pose_dir)
        candidates = []
        for name in (f"pic{iteration_idx}.jpg", "top.jpg", "wrist.jpg", "reference.jpg", "ref.jpg"):
            p = ref_dir / name
            if p.exists():
                candidates.append(p)
        if not candidates:
            # pick first image file in the directory
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files = sorted(ref_dir.glob(ext))
                if files:
                    candidates.extend(files)
                    break

        if not candidates:
            print(f"[WARN] Reference pose image missing for iteration {iteration_idx}: {pose_path}")
            return None, pose_path

        pose_path = candidates[0]
        print(f"[INFO] Using reference pose image: {pose_path}")

    image = cv2.imread(str(pose_path))
    if image is None:
        print(f"[WARN] Could not read reference pose image: {pose_path}")
        return None, pose_path

    width, height = target_shape
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = bgr_to_rgb(image)
    return image, pose_path


def get_observation_with_retry(
    robot: ZoomRobot,
    retries: int = 3,
    retry_delay_s: float = 0.05,
) -> dict[str, Any] | None:
    last_exc: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            return robot.get_observation()
        except TimeoutError as exc:
            last_exc = exc
            time.sleep(retry_delay_s)
        except ConnectionError as exc:
            last_exc = exc
            time.sleep(retry_delay_s)

    if last_exc is not None:
        print(f"[WARN] Observation read failed after retries: {last_exc}")
    return None


def _normalize_task_name(task_str: str) -> str:
    """Convert task string to a safe directory name."""
    # Replace spaces and special characters with underscores
    safe_name = task_str.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
    # Keep only alphanumeric and underscores
    safe_name = "".join(c if c.isalnum() or c == "_" else "" for c in safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip("_")
    # If empty or too long, use a hash instead
    if not safe_name:
        import hashlib
        safe_name = "task_" + hashlib.md5(task_str.encode()).hexdigest()[:8]
    elif len(safe_name) > 100:
        import hashlib
        safe_name = safe_name[:80] + "_" + hashlib.md5(task_str.encode()).hexdigest()[:8]
    return safe_name


def _candidate_task_names(task_id: str, task_name: str) -> list[str]:
    candidates: list[str] = []
    for value in (task_id, task_name):
        if not value:
            continue
        raw = str(value).strip()
        if raw:
            candidates.append(raw)
            candidates.append(_normalize_task_key(raw))
            if raw.startswith("task_"):
                candidates.append("task" + raw[len("task_"):])
            if raw.startswith("task") and not raw.startswith("task_") and len(raw) > 4:
                candidates.append("task_" + raw[4:])

    seen: set[str] = set()
    out: list[str] = []
    for name in candidates:
        name = str(name).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def load_reset_action_vector(action_file: str, action_names: list[str]) -> np.ndarray:
    """Load a normalized reset action vector from JSON.

    Supports:
      - list of floats (length == action dim)
      - dict keyed by action name
    """
    raw = json.loads(Path(action_file).read_text())
    if isinstance(raw, dict):
        values = []
        missing = []
        for name in action_names:
            if name in raw:
                values.append(raw[name])
            else:
                missing.append(name)
        if missing:
            raise ValueError(f"Reset action file missing keys: {missing}")
        vec = np.asarray(values, dtype=np.float32)
    elif isinstance(raw, list):
        vec = np.asarray(raw, dtype=np.float32)
    else:
        raise ValueError("Reset action file must be a JSON list or dict")

    if vec.ndim != 1 or vec.shape[0] != len(action_names):
        raise ValueError(
            f"Reset action vector must be 1D of length {len(action_names)}; got shape {vec.shape}"
        )
    return vec


def _make_run_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def print_molmoact2_dry_run_debug(
    policy: MolmoAct2PolicyAdapter,
    action_names: list[str],
    task: str,
    images: list[np.ndarray],
) -> None:
    """Print the complete model-frame -> arm-frame action conversion."""
    debug = policy.last_debug
    print("\n[MOLMOACT2 DRY RUN]")
    print(f"  task: {task!r}")
    print(f"  images: {[tuple(image.shape) for image in images]}")
    print(f"  norm_tag: {policy.norm_tag}")
    print("  transform: model = signs * arm + offsets")
    print(f"  signs:   {policy.joint_signs.tolist()}")
    print(f"  offsets: {policy.joint_offsets.tolist()}")
    print(
        "  {:<24} {:>10} {:>11} {:>12} {:>14} {:>10} {:>9}".format(
            "joint", "arm_state", "model_state", "model_out", "arm_unclamped", "arm_cmd", "delta"
        )
    )
    for idx, name in enumerate(action_names):
        clipped = " CLIPPED" if bool(debug["clipped"][idx]) else ""
        print(
            f"  {name:<24} {debug['arm_state'][idx]:>10.3f} {debug['model_state'][idx]:>11.3f} "
            f"{debug['model_action'][idx]:>12.3f} {debug['arm_target_unclamped'][idx]:>14.3f} "
            f"{debug['arm_action'][idx]:>10.3f} {debug['arm_delta'][idx]:>9.3f}{clipped}"
        )
    print(
        f"  max_step_deg: {policy.max_joint_step_deg:g}; "
        f"queued_actions_remaining: {debug['queued_actions_remaining']}"
    )
    if debug.get("new_chunk"):
        chunk = debug["predicted_chunk"]
        print(
            f"  full model-output chunk: shape={tuple(chunk.shape)}, "
            f"executing_first={debug['executed_chunk_size']}"
        )
        for step, model_action in enumerate(chunk):
            values = ", ".join(
                f"{name}={float(model_action[idx]):.3f}" for idx, name in enumerate(action_names)
            )
            print(f"    [{step:02d}] {values}")


def run_policy_phase(
    robot,
    policy,
    preprocess,
    postprocess,
    ds_features,
    task: str,
    action_names: list[str],
    device: torch.device,
    fps: int,
    max_seconds: float,
    key_reader: TerminalKeyReader,
    policy_type: str,
    robot_type: str,
    output_video_dir: str,
    policy_run_name: str,
    iteration_idx: int,
) -> bool:
    """Returns True when we should continue, False when user requested quit."""
    dt = 1.0 / float(fps)
    t_end = time.time() + max_seconds
    policy.reset()
    
    # Initialize video recording
    video_writer = None
    actions_list = []
    output_dir = Path(output_video_dir) / policy_run_name / _normalize_task_name(task)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = output_dir / f"episode_{iteration_idx:03d}.mp4"
    actions_path = output_dir / f"episode_{iteration_idx:03d}_actions.npy"

    print(
        f"[EVAL] Running {policy_type} policy for up to {max_seconds:.1f}s. "
        "Press Right Arrow to end this run early, q to quit."
    )

    while time.time() < t_end:
        t0 = time.time()

        key = key_reader.get_key()
        if key == "QUIT":
            return False
        if key == "RIGHT":
            print("[EVAL] Right Arrow detected. Returning to teleop setup.")
            break

        obs = get_observation_with_retry(robot)
        if obs is None:
            print("[EVAL] Camera stream unavailable. Ending this policy run early.")
            return False

        # Extract top camera for video recording
        top_image = extract_top_camera_image(obs)
        if top_image is not None and video_writer is None:
            h, w = top_image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                fps,
                (w, h)
            )
            if not video_writer.isOpened():
                print(f"[WARN] Could not open video writer for {video_path}")

        # Record frame to video
        if video_writer is not None and top_image is not None:
            # Convert RGB to BGR for OpenCV
            top_image_bgr = cv2.cvtColor(top_image, cv2.COLOR_RGB2BGR)
            video_writer.write(top_image_bgr)

        if policy_type == "molmoact2":
            images, state = extract_molmoact2_inputs(obs, action_names)
            action_vec = policy.predict_action(images=images, task=task, state=state)
            robot_action = {name: float(action_vec[idx]) for idx, name in enumerate(action_names)}
            actions_list.append(action_vec.copy())
            if policy.dry_run:
                print_molmoact2_dry_run_debug(policy, action_names, task, images)
        elif policy_type == "act":
            observation_frame = build_dataset_frame(
                ds_features=ds_features,
                values=obs,
                prefix=OBS_STR,
            )
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocess,
                postprocessor=postprocess,
                use_amp=policy.config.use_amp,
                task=task,
                robot_type=robot_type,
            )
            action_vec = to_action_vector(action_values, len(action_names))
            robot_action = {name: float(action_vec[idx]) for idx, name in enumerate(action_names)}
            # Save normalized action
            actions_list.append(action_vec.cpu().numpy())
        else:
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=ds_features,
                device=str(device),
                task=task,
                robot_type=robot_type,
            )
            batch = preprocess(obs_frame)
            with torch.no_grad():
                raw_action = policy.select_action(batch)
            action = postprocess(raw_action)
            robot_action = make_robot_action(action.squeeze(0), ds_features)
            # Save normalized action
            if isinstance(action, torch.Tensor):
                actions_list.append(action.squeeze(0).detach().float().cpu().numpy())
            else:
                actions_list.append(np.asarray(action).squeeze())

        if not (policy_type == "molmoact2" and policy.dry_run):
            robot.send_action(robot_action)

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # Save video file
    if video_writer is not None:
        video_writer.release()
        print(f"[EVAL] Video saved to {video_path}")

    # Save actions to file
    if actions_list:
        try:
            actions_array = np.array(actions_list)
            np.save(str(actions_path), actions_array)
            print(f"[EVAL] Actions saved to {actions_path} (shape: {actions_array.shape})")
        except Exception as e:
            print(f"[WARN] Could not save actions: {e}")

    return True


def run_teleop_setup_phase(
    robot,
    teleop: SO101Leader | None,
    key_reader: TerminalKeyReader,
    iteration_idx: int,
    total_iterations: int,
    reference_pose_dir: str | None,
    overlay_alpha: float,
    teleop_overlay_fps: int,
    zoom_factor: float,
    setup_preview_window: bool,
    reset_mode: str,
    reset_robot_action: dict[str, float] | None,
) -> bool:
    """Teleop phase before each policy run. Returns True to start policy, False to quit."""
    mode_label = {
        "leader": "leader controls follower",
        "fixed": "fixed reset action",
        "none": "manual reset (no leader or reset commands)",
    }[reset_mode]
    print(
        f"[TELEOP] Iteration {iteration_idx}/{total_iterations}: {mode_label}. "
        "Press ENTER to start policy, q to quit."
    )
    if reference_pose_dir:
        print(f"[TELEOP] Using reference pose image: pic{iteration_idx}.jpg from {reference_pose_dir}")

    dt = 1.0 / float(max(1, teleop_overlay_fps))
    last_overlay_log_t = 0.0

    overlay_ref = None
    overlay_path = None
    overlay_load_attempted = False
    overlay_timeout_warning_shown = False
    preview_window_name = "SO101 Teleop Overlay"
    preview_enabled = setup_preview_window and can_open_cv_preview_window()

    if setup_preview_window and not preview_enabled:
        print("[WARN] Setup preview window requested but GUI display is unavailable.")

    if preview_enabled:
        print("[TELEOP] Setup preview window enabled.")

    try:
        while True:
            key = key_reader.get_key()
            if key == "QUIT":
                return False
            if key == "ENTER":
                print("[TELEOP] ENTER detected. Handing off control and camera stream to policy.")
                return True

            if reset_mode == "fixed":
                if reset_robot_action is not None:
                    robot.send_action(reset_robot_action)
            elif reset_mode == "leader":
                if teleop is None:
                    raise RuntimeError("Teleop leader is not initialized but reset_mode=leader")
                action = teleop.get_action()
                if action is not None:
                    robot.send_action(action)

            now = time.time()
            should_refresh_visuals = preview_enabled
            if should_refresh_visuals and (now - last_overlay_log_t >= dt):
                obs = get_observation_with_retry(robot)
                if obs is None:
                    if not overlay_timeout_warning_shown:
                        print("[TELEOP] Camera frame timeout during overlay; keeping leader control active.")
                        overlay_timeout_warning_shown = True
                    last_overlay_log_t = now
                    continue

                overlay_timeout_warning_shown = False
                top_image = extract_top_camera_image(obs)

                if top_image is not None:
                    h, w = top_image.shape[:2]
                    if overlay_ref is None and reference_pose_dir and not overlay_load_attempted:
                        overlay_load_attempted = True
                        overlay_ref, overlay_path = load_reference_pose_image(
                            reference_pose_dir,
                            iteration_idx,
                            (w, h),
                        )

                    blended = None
                    if overlay_ref is not None:
                        blended = cv2.addWeighted(top_image, 1.0 - overlay_alpha, overlay_ref, overlay_alpha, 0)

                    if preview_enabled:
                        preview_image = blended if blended is not None else top_image
                        preview_enabled = show_setup_preview(preview_window_name, preview_image)

                last_overlay_log_t = now

            time.sleep(0.005)
    finally:
        if preview_enabled:
            close_setup_preview(preview_window_name)


def run_final_teleop_buffer_phase(
    robot,
    teleop: SO101Leader,
    key_reader: TerminalKeyReader,
    min_buffer_seconds: float,
) -> bool:
    print(
        "[TELEOP] Final post-eval teleop phase started. "
        f"Press ENTER to cut leader control after {min_buffer_seconds:.1f}s, or q to quit immediately."
    )

    t_start = time.time()

    while True:
        key = key_reader.get_key()
        elapsed = time.time() - t_start

        if key == "QUIT":
            return False

        if key == "ENTER":
            if elapsed >= min_buffer_seconds:
                print("[TELEOP] ENTER detected. Leader control released.")
                return True

            remaining = max(0.0, min_buffer_seconds - elapsed)
            print(f"[TELEOP] Buffer active for {remaining:.1f}s more before ENTER is accepted.")

        action = teleop.get_action()
        if action is not None:
            robot.send_action(action)

        time.sleep(0.005)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Unified SO101 evaluator for ACT/SmolVLA/Pi0/Pi0.5/Pi0-fast/DiT/X-VLA/MolmoAct2 with teleop-first handoff, "
            "top-camera zoom, OpenCV overlay visualization, and per-episode video recording."
        )
    )

    parser.add_argument(
        "--policy-type",
        type=str,
        default="smolvla",
        choices=list(POLICY_REGISTRY.keys()),
        help="Policy architecture to load.",
    )
    parser.add_argument(
        "--policy-family",
        type=str,
        choices=["act", "smolvla", "pi0", "pi05", "pi0.5", "pi0-fast", "pi0_fast", "pi0fast", "dit", "xvla", "x-vla", "x_vla", "molmoact2"],
        default=None,
        help="Alias for --policy-type.",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="Hugging Face repo or local path. Defaults depend on --policy-type.",
    )
    parser.add_argument(
        "--policy-from-hub",
        action="store_true",
        help="For ACT, treat --policy-path as a Hugging Face repo id.",
    )
    parser.add_argument(
        "--policy-revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision/branch/tag/commit.",
    )
    parser.add_argument(
        "--molmoact2-dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="MolmoAct2 model dtype. bfloat16 is recommended for 24 GB GPUs.",
    )
    parser.add_argument(
        "--molmoact2-norm-tag",
        default="so100_so101_molmoact2",
        help="Normalization tag stored in the MolmoAct2 checkpoint.",
    )
    parser.add_argument(
        "--molmoact2-num-steps",
        type=int,
        default=10,
        help="Number of continuous flow-solver steps.",
    )
    parser.add_argument(
        "--molmoact2-actions-per-chunk",
        type=int,
        default=1,
        help="Maximum predicted actions to execute before requesting a new chunk (default: 1 for safety).",
    )
    parser.add_argument(
        "--molmoact2-enable-cuda-graph",
        action="store_true",
        help="Enable CUDA graph capture after initial testing (disabled by default).",
    )
    parser.add_argument(
        "--molmoact2-max-joint-step-deg",
        type=float,
        default=3.0,
        help="Clamp each absolute joint command relative to current state; 0 disables clamping.",
    )
    parser.add_argument(
        "--molmoact2-dry-run",
        dest="molmoact2_dry_run",
        action="store_true",
        default=True,
        help="Run inference and record/print actions without sending commands to the robot (default).",
    )
    parser.add_argument(
        "--molmoact2-enable-hardware-actions",
        dest="molmoact2_dry_run",
        action="store_false",
        help=(
            "DANGEROUS: allow MolmoAct2 to command the follower. Use only after checking calibration "
            "and validating recorded dry-run state/action units and joint order."
        ),
    )
    parser.add_argument(
        "--molmoact2-joint-offsets",
        default="0,90,90,0,0,0",
        help=(
            "Offsets for SO-101 v3 arm coordinates -> checkpoint v2.1 coordinates "
            "(default: 0,90,90,0,0,0)."
        ),
    )
    parser.add_argument(
        "--molmoact2-joint-signs",
        default="1,-1,1,1,1,1",
        help=(
            "Signs for SO-101 v3 arm coordinates -> checkpoint v2.1 coordinates "
            "(default: 1,-1,1,1,1,1)."
        ),
    )

    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task instruction string passed to the policy. Required unless --task-id is set.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help=(
            "Task identifier to resolve from vla_tasks_gui_config.json (or --task-config-file). "
            "When set, the task instruction, reference directory, and variants file are resolved automatically."
        ),
    )
    parser.add_argument(
        "--task-config-file",
        type=str,
        default=None,
        help="Optional JSON file containing the task registry used to resolve --task-id.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default=None,
        help="Robot type passed to inference frame builders.",
    )

    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--test-iterations",
        type=int,
        default=None,
        help="Alias for --iterations (number of testing iterations).",
    )
    parser.add_argument("--policy-seconds", type=float, default=90.0)
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.5,
        help="Top camera center-crop zoom factor (default: 1.5).",
    )
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument(
        "--reference-pose-dir",
        type=str,
        default=None,
        help=(
            "Directory containing reference images pic1.jpg ... picN.jpg used during teleop "
            "overlay for scene reset guidance."
        ),
    )

    parser.add_argument(
        "--task-variants-file",
        type=str,
        default=None,
        help=(
            "Optional JSON file with a list of task instruction strings to use per-iteration. "
            "If provided, the evaluator will use the i-th entry for iteration i instead of --task."
        ),
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.4, help="Blend factor for overlay image.")
    parser.add_argument(
        "--overlay-transparency",
        type=float,
        default=None,
        help="Alias for --overlay-alpha. Use values in [0.0, 1.0].",
    )
    parser.add_argument("--teleop-overlay-fps", type=int, default=10)
    parser.add_argument(
        "--setup-preview-window",
        dest="setup_preview_window",
        action="store_true",
        help="Show an OpenCV live preview window during teleop setup.",
    )
    parser.add_argument(
        "--no-setup-preview-window",
        dest="setup_preview_window",
        action="store_false",
        help="Disable OpenCV live preview window during teleop setup.",
    )
    parser.set_defaults(setup_preview_window=True)
    parser.add_argument(
        "--final-teleop-buffer-seconds",
        type=float,
        default=0.0,
        help="Minimum post-eval teleop buffer time before ENTER can cut leader control.",
    )
    parser.add_argument(
        "--reset-mode",
        type=str,
        choices=["leader", "fixed", "none"],
        default="leader",
        help="Reset control mode: leader arm, fixed normalized action vector, or manual (none).",
    )
    parser.add_argument(
        "--reset-action-file",
        type=str,
        default=None,
        help="Path to JSON file containing a normalized reset action vector (list or dict). Required for --reset-mode fixed.",
    )

    parser.add_argument("--follower-port", type=str, default="/dev/ttyACM1")
    parser.add_argument(
        "--follower-id",
        type=str,
        default="so101_follower_arm",
        help="Follower arm id for the default target.",
    )
    parser.add_argument(
        "--eval-follower-calib-dirs",
        type=str,
        default=None,
        help=(
            "Comma-separated list of follower calibration directories to evaluate on. "
            "Defaults to calibration/robots/so101_follower."
        ),
    )
    parser.add_argument(
        "--eval-follower-ports",
        type=str,
        default=None,
        help="Comma-separated list of follower ports for each calibration dir.",
    )
    parser.add_argument(
        "--eval-follower-ids",
        type=str,
        default=None,
        help="Comma-separated list of follower ids for each calibration dir.",
    )
    parser.add_argument(
        "--primary-follower-index",
        type=int,
        default=0,
        help="Index of the arm used for policy inputs (0-based).",
    )
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--top-index", type=int, default=4)
    parser.add_argument("--wrist-index", type=int, default=6)
    parser.add_argument(
        "--eval-top-indexes",
        type=str,
        default=None,
        help=(
            "Comma-separated top camera indices per arm. If one value is provided, it is used for the primary arm."
        ),
    )
    parser.add_argument(
        "--eval-wrist-indexes",
        type=str,
        default=None,
        help=(
            "Comma-separated wrist camera indices per arm. If one value is provided, it is used for the primary arm."
        ),
    )
    parser.add_argument("--top-width", type=int, default=640)
    parser.add_argument("--top-height", type=int, default=480)
    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--wrist-fps", type=int, default=30)

    parser.add_argument(
        "--no-hardware",
        dest="no_hardware",
        action="store_true",
        help="Do not connect to robot hardware or cameras; use synthetic observations (safe dry-run).",
    )

    parser.add_argument(
        "--output-video-dir",
        type=str,
        default="policy_eval_videos", # default output directory
        help="Output directory for recording videos and normalized actions per episode.",
    )
    parser.add_argument(
        "--run-all-tasks",
        dest="run_all_tasks",
        action="store_true",
        help="Run the evaluator on all tasks from the task config in one policy instantiation.",
    )
    parser.add_argument(
        "--tasks-root",
        type=str,
        default="tasks",
        help="Root directory containing task sets (ID, OOD) with task_variants and referencePics subfolders.",
    )
    parser.add_argument(
        "--task-subset",
        type=str,
        choices=["ID", "OOD"],
        default=None,
        help="When using --run-all-tasks, only run tasks from the specified subset (ID or OOD). If None, both subsets are searched.",
    )

    args = parser.parse_args()

    if args.policy_family is not None:
        family = args.policy_family.lower()
        if family == "pi0.5":
            family = "pi05"
        if family in {"pi0-fast", "pi0_fast", "pi0fast"}:
            family = "pi0_fast"
        if family in {"x-vla", "x_vla"}:
            family = "xvla"
        args.policy_type = family

    if args.test_iterations is not None:
        args.iterations = args.test_iterations

    # Prepare tasks directory structure when requested
    tasks_root = Path(args.tasks_root)
    if args.run_all_tasks:
        for subset in ("ID", "OOD"):
            (tasks_root / subset / "task_variants").mkdir(parents=True, exist_ok=True)
            (tasks_root / subset / "referencePics").mkdir(parents=True, exist_ok=True)

    if args.overlay_transparency is not None:
        args.overlay_alpha = args.overlay_transparency

    task_config = None
    resolved_task_id = None
    resolved_task_variants_file: Optional[Path] = None

    if args.task_id and not args.run_all_tasks:
        task_config = load_task_config(args.task_config_file)
        task_entry = resolve_task_from_config(task_config, args.task_id)
        resolved_task_id = task_entry.get("id", args.task_id)

        if not args.task:
            args.task = task_entry.get("name") or task_entry.get("task")

        if not args.reference_pose_dir:
            ref_dir_value = task_entry.get("reference_dir")
            if ref_dir_value:
                args.reference_pose_dir = str(ref_dir_value)

        resolved_task_variants_file = resolve_task_variants_file(task_entry)
        if resolved_task_variants_file and not args.task_variants_file:
            args.task_variants_file = str(resolved_task_variants_file)

    if not args.run_all_tasks and not args.task:
        raise ValueError("Either --task/--task-id or --run-all-tasks must be provided.")

    if args.overlay_alpha < 0.0 or args.overlay_alpha > 1.0:
        raise ValueError("--overlay-alpha must be in [0.0, 1.0]")

    if args.final_teleop_buffer_seconds < 0.0:
        raise ValueError("--final-teleop-buffer-seconds must be >= 0.0")

    if args.molmoact2_num_steps <= 0:
        raise ValueError("--molmoact2-num-steps must be > 0")
    if args.molmoact2_actions_per_chunk <= 0:
        raise ValueError("--molmoact2-actions-per-chunk must be > 0")
    if args.molmoact2_max_joint_step_deg < 0.0:
        raise ValueError("--molmoact2-max-joint-step-deg must be >= 0")
    molmoact2_joint_offsets = parse_molmoact2_joint_transform(
        args.molmoact2_joint_offsets, "--molmoact2-joint-offsets"
    )
    molmoact2_joint_signs = parse_molmoact2_joint_transform(
        args.molmoact2_joint_signs, "--molmoact2-joint-signs"
    )
    if not np.isin(molmoact2_joint_signs, (-1.0, 1.0)).all():
        raise ValueError("--molmoact2-joint-signs values must each be -1 or 1")

    registry_entry = POLICY_REGISTRY[args.policy_type]
    policy_path = args.policy_path or registry_entry["default_path"]
    robot_type = args.robot_type or registry_entry["default_robot_type"]

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = get_safe_torch_device(device_str, log=True)

    if args.policy_type == "molmoact2":
        if args.molmoact2_dry_run:
            print(
                "[SAFETY] MolmoAct2 is in dry-run mode; policy actions will NOT be sent to the follower. "
                "Live policy motion requires --molmoact2-enable-hardware-actions."
            )
        else:
            print(
                "[SAFETY WARNING] MolmoAct2 hardware actions are ENABLED. Keep an emergency stop ready "
                "and verify that robot state/action units match the checkpoint's raw robot scale."
            )
        policy = load_molmoact2_policy(
            policy_path=policy_path,
            device=device,
            revision=args.policy_revision,
            dtype_name=args.molmoact2_dtype,
            norm_tag=args.molmoact2_norm_tag,
            num_steps=args.molmoact2_num_steps,
            actions_per_chunk=args.molmoact2_actions_per_chunk,
            enable_cuda_graph=args.molmoact2_enable_cuda_graph,
            max_joint_step_deg=args.molmoact2_max_joint_step_deg,
            dry_run=args.molmoact2_dry_run,
            joint_signs=molmoact2_joint_signs,
            joint_offsets=molmoact2_joint_offsets,
        )
    else:
        policy = load_policy(
            policy_type=args.policy_type,
            policy_path=policy_path,
            device=str(device),
            policy_from_hub=args.policy_from_hub,
            revision=args.policy_revision,
        )

    # For pi0_fast, use optimized GPU-accelerated action tokenizer processor
    _preprocessor_overrides = {"device_processor": {"device": str(device)}}
    if args.policy_type == "pi0_fast":
        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
            from processing_action_tokenizer_optimized import OptimizedUniversalActionProcessor

            _action_tokenizer_repo = "lerobot/fast-action-tokenizer"
            _snapshot = snapshot_download(repo_id=_action_tokenizer_repo, repo_type="model")

            _processor_cfg_path = Path(_snapshot) / "processor_config.json"
            _processor_cfg = {}
            if _processor_cfg_path.exists():
                with open(_processor_cfg_path, "r", encoding="utf-8") as _f:
                    _processor_cfg = json.load(_f)
            
            _bpe_tokenizer = AutoTokenizer.from_pretrained(_snapshot, use_fast=True, trust_remote_code=True)
            # Prefer repo-provided settings when present.
            _action_tokenizer_obj = OptimizedUniversalActionProcessor(
                bpe_tokenizer=_bpe_tokenizer,
                scale=float(_processor_cfg.get("scale", 10)),
                vocab_size=int(_processor_cfg.get("vocab_size", 2048)),
                min_token=int(_processor_cfg.get("min_token", -354)),
                device=str(device),
            )
            _preprocessor_overrides["action_tokenizer_processor"] = {
                "action_tokenizer_input_object": _action_tokenizer_obj
            }
        except Exception as _err:
            print(f"[WARN] Could not load optimized action tokenizer for pi0_fast: {_err}. Falling back to default.")

    if args.policy_type == "molmoact2":
        preprocess, postprocess = None, None
    else:
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            policy_path,
            preprocessor_config_filename="policy_preprocessor.json",
            postprocessor_config_filename="policy_postprocessor.json",
            preprocessor_overrides=_preprocessor_overrides,
        )

    # Load optional global per-iteration task variants (only used when not running all tasks)
    task_variants: list[str] | None = None
    if not args.run_all_tasks and args.task_variants_file:
        try:
            variants_path = Path(args.task_variants_file)
            with open(variants_path, "r") as vf:
                variants_obj = json.load(vf)

            if isinstance(variants_obj, list):
                task_variants = []
                for v in variants_obj:
                    if isinstance(v, str):
                        task_variants.append(v)
                    elif isinstance(v, dict):
                        task_variants.append(v.get("name") or v.get("task") or str(v))
                    else:
                        task_variants.append(str(v))
            elif isinstance(variants_obj, dict) and "tasks" in variants_obj and isinstance(variants_obj["tasks"], list):
                task_variants = [t if isinstance(t, str) else (t.get("name") or t.get("task") or str(t)) for t in variants_obj["tasks"]]
            else:
                print(f"[WARN] Unsupported format in --task-variants-file: {variants_path}")
                task_variants = None

            if task_variants:
                print(f"[INIT] Loaded {len(task_variants)} task variants from {variants_path}")
                if len(task_variants) < args.iterations:
                    print(f"[WARN] Fewer variants ({len(task_variants)}) than iterations ({args.iterations}); last variant will be reused for remaining iterations.")
        except Exception as e:
            print(f"[WARN] Failed to read task variants file '{args.task_variants_file}': {e}")
            task_variants = None

    def _build_camera_cfg(top_index: int, wrist_index: int):
        return {
            "top": OpenCVCameraConfig(
                index_or_path=top_index,
                width=args.top_width,
                height=args.top_height,
                fps=args.fps,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=wrist_index,
                width=args.wrist_width,
                height=args.wrist_height,
                fps=args.wrist_fps,
                fourcc="MJPG",
                warmup_s=2,
            ),
        }

    calib_dir = Path(__file__).resolve().parent / "calibration"
    default_follower_calib = calib_dir / "robots" / "so101_follower"
    leader_cfg = SO101LeaderConfig(
        port=args.leader_port,
        id="so101_leader_arm",
        calibration_dir=calib_dir / "teleoperators" / "so101_leader",
    )

    def _split_csv(value: str | None) -> list[str]:
        if not value:
            return []
        return [v.strip() for v in value.split(",") if v.strip()]

    eval_calib_dirs = _split_csv(args.eval_follower_calib_dirs)
    if not eval_calib_dirs:
        eval_calib_dirs = [str(default_follower_calib)]

    eval_ports = _split_csv(args.eval_follower_ports)
    if eval_ports and len(eval_ports) != len(eval_calib_dirs):
        raise ValueError("--eval-follower-ports must match --eval-follower-calib-dirs length")
    if not eval_ports:
        eval_ports = [args.follower_port] * len(eval_calib_dirs)

    eval_ids = _split_csv(args.eval_follower_ids)
    if eval_ids and len(eval_ids) != len(eval_calib_dirs):
        raise ValueError("--eval-follower-ids must match --eval-follower-calib-dirs length")
    if not eval_ids:
        eval_ids = [args.follower_id] * len(eval_calib_dirs)

    def _split_csv_int(value: str | None) -> list[int]:
        if not value:
            return []
        return [int(v.strip()) for v in value.split(",") if v.strip()]

    eval_top_indexes = _split_csv_int(args.eval_top_indexes)
    if eval_top_indexes and len(eval_top_indexes) not in (1, len(eval_calib_dirs)):
        raise ValueError("--eval-top-indexes must be length 1 or match --eval-follower-calib-dirs length")
    if not eval_top_indexes:
        eval_top_indexes = [args.top_index] * len(eval_calib_dirs)
    if len(eval_top_indexes) == 1:
        eval_top_indexes = [eval_top_indexes[0]] * len(eval_calib_dirs)

    eval_wrist_indexes = _split_csv_int(args.eval_wrist_indexes)
    if eval_wrist_indexes and len(eval_wrist_indexes) not in (1, len(eval_calib_dirs)):
        raise ValueError("--eval-wrist-indexes must be length 1 or match --eval-follower-calib-dirs length")
    if not eval_wrist_indexes:
        eval_wrist_indexes = [args.wrist_index] * len(eval_calib_dirs)
    if len(eval_wrist_indexes) == 1:
        eval_wrist_indexes = [eval_wrist_indexes[0]] * len(eval_calib_dirs)

    teleop: SO101Leader | None = None
    if args.reset_mode == "leader":
        teleop = SO101Leader(leader_cfg)
        print("[INIT] Connecting leader...")
        teleop.connect()
    else:
        print(f"[INIT] Leader teleop disabled (reset mode: {args.reset_mode}).")

    print(f"[INIT] Policy type : {args.policy_type}")
    print(f"[INIT] Policy path : {policy_path}")
    print(f"[INIT] Robot type  : {robot_type}")
    if resolved_task_id:
        print(f"[INIT] Task id     : {resolved_task_id}")
    print(f"[INIT] Task        : {args.task}")
    print(f"[INIT] Iterations  : {args.iterations}")
    print(f"[INIT] Zoom        : {args.zoom}")
    print(f"[INIT] Overlay alpha: {args.overlay_alpha}")
    print(f"[INIT] Output video dir: {args.output_video_dir}")

    if args.reference_pose_dir:
        print(f"[INIT] Reference poses directory: {args.reference_pose_dir}")
    if args.task_variants_file:
        print(f"[INIT] Task variants file: {args.task_variants_file}")

    try:
        with TerminalKeyReader() as key_reader:
            policy_run_name = f"{args.policy_type}_{_make_run_timestamp()}"
            print(f"[INIT] Policy run folder: {policy_run_name}")

            print("\n[INIT] Target arms (simultaneous)")
            robots: list[ZoomRobot] = []
            for idx, (calib_dir_str, port, follower_id) in enumerate(
                zip(eval_calib_dirs, eval_ports, eval_ids), start=1
            ):
                print(f"[INIT]   {idx}. calib dir: {calib_dir_str}")
                print(f"[INIT]      port     : {port}")
                print(f"[INIT]      id       : {follower_id}")
                is_primary = (idx - 1) == args.primary_follower_index
                top_index = eval_top_indexes[idx - 1]
                wrist_index = eval_wrist_indexes[idx - 1]
                follower_cfg = SO101FollowerConfig(
                    port=port,
                    id=follower_id,
                    cameras=_build_camera_cfg(top_index, wrist_index) if is_primary else {},
                    calibration_dir=Path(calib_dir_str),
                    # MolmoAct2 was trained in degrees. LeRobot 0.5.1 otherwise
                    # exposes v3 normalized joint ranges (-100..100 / 0..100),
                    # which must never be sent directly to this checkpoint.
                    use_degrees=args.policy_type == "molmoact2",
                    max_relative_target=(
                        args.molmoact2_max_joint_step_deg
                        if args.policy_type == "molmoact2" and args.molmoact2_max_joint_step_deg > 0
                        else None
                    ),
                )
                raw_robot = SO101Follower(follower_cfg)
                robots.append(ZoomRobot(raw_robot, zoom_factor=args.zoom))

            if args.primary_follower_index < 0 or args.primary_follower_index >= len(robots):
                raise ValueError("--primary-follower-index is out of range for provided arms")

            # Determine dataset feature metadata from the primary robot before touching hardware
            primary_candidate = robots[args.primary_follower_index]
            primary_raw = getattr(primary_candidate, "robot", primary_candidate)
            action_features = hw_to_dataset_features_compat(primary_raw.action_features, "action")
            obs_features = hw_to_dataset_features_compat(primary_raw.observation_features, "observation")
            ds_features = {**obs_features, **action_features}
            action_names = ds_features[ACTION]["names"]

            # If requested, substitute fake robots that do not open cameras or serial ports
            if args.no_hardware:
                print("[INIT] No-hardware mode: using synthetic fake robots (no device access)")
                fake_robots: list[Any] = []
                for r in robots:
                    raw = getattr(r, "robot", r)
                    fake = FakeRobot(
                        real_robot=raw,
                        top_shape=(args.top_height, args.top_width),
                        wrist_shape=(args.wrist_height, args.wrist_width),
                        action_names=action_names,
                    )
                    fake_robots.append(fake)
                robots = fake_robots

            multi_robot = MultiRobot(robots, primary_idx=args.primary_follower_index)

            if not args.no_hardware:
                print("[INIT] Connecting followers...")
                multi_robot.connect()
            reset_robot_action: dict[str, float] | None = None
            if args.reset_mode == "fixed":
                if not args.reset_action_file:
                    raise ValueError("--reset-action-file is required when --reset-mode fixed")
                reset_vec = load_reset_action_vector(args.reset_action_file, action_names)
                reset_tensor = torch.as_tensor(reset_vec, dtype=torch.float32)
                reset_robot_action = make_robot_action(reset_tensor, ds_features)

            finished_all_iterations = True

            # Build list of tasks to run
            tasks_to_run: list[dict[str, Any]] = []
            if args.run_all_tasks:
                task_config = load_task_config(args.task_config_file)
                tasks_to_run = list(task_config.get("tasks", []))
                
                # Filter tasks by subset if specified
                if args.task_subset:
                    filtered_tasks = []
                    for task in tasks_to_run:
                        task_id = str(task.get("id") or "").strip()
                        task_name = str(task.get("name") or task.get("task") or "").strip()
                        # Check if task has resources in the specified subset
                        candidates = _candidate_task_names(task_id, task_name)
                        has_resources = False
                        for candidate in candidates:
                            subset_variants = tasks_root / args.task_subset / "task_variants" / f"{candidate}.json"
                            subset_ref = tasks_root / args.task_subset / "referencePics" / candidate
                            if subset_variants.exists() or subset_ref.exists():
                                has_resources = True
                                break
                        if has_resources:
                            filtered_tasks.append(task)
                    tasks_to_run = filtered_tasks
                    print(f"[INIT] Filtered to {len(tasks_to_run)} tasks in {args.task_subset} subset")
            else:
                # single task: either provided by task_entry (via --task-id) or ad-hoc via --task
                if 'task_entry' in locals():
                    tasks_to_run = [task_entry]
                else:
                    tasks_to_run = [{"id": None, "name": args.task or args.task_id, "task": args.task}]

            for task_obj in tasks_to_run:
                task_id = str(task_obj.get("id") or "").strip()
                task_name_field = str(task_obj.get("name") or task_obj.get("task") or "").strip()

                # Resolve per-task variants file and reference dir.
                # If a subset is explicitly requested, only use that subset's assets.
                # Otherwise keep the historical preference order.
                per_task_variants: list[str] | None = None
                per_task_variants_path: Optional[Path] = None
                per_task_reference_dir: Optional[str] = None
                task_name_candidates = _candidate_task_names(task_id, task_name_field)
                preferred_subsets = [args.task_subset] if args.task_subset else ["ID", "OOD"]

                # look under the preferred subset(s) only
                for subset in preferred_subsets:
                    for candidate_name in task_name_candidates:
                        candidate_variants = tasks_root / subset / "task_variants" / f"{candidate_name}.json"
                        if candidate_variants.exists():
                            per_task_variants_path = candidate_variants
                            break
                    if per_task_variants_path is not None:
                        break

                # fallback to task entry variants file
                if per_task_variants_path is None:
                    candidate = resolve_task_variants_file(task_obj) if isinstance(task_obj, dict) else None
                    if candidate and candidate.exists():
                        per_task_variants_path = candidate

                # load per-task variants if available
                if per_task_variants_path is not None:
                    try:
                        with open(per_task_variants_path, "r") as vf:
                            variants_obj = json.load(vf)
                        per_task_variants = []
                        if isinstance(variants_obj, list):
                            for v in variants_obj:
                                if isinstance(v, str):
                                    per_task_variants.append(v)
                                elif isinstance(v, dict):
                                    per_task_variants.append(v.get("name") or v.get("task") or str(v))
                                else:
                                    per_task_variants.append(str(v))
                        elif isinstance(variants_obj, dict) and "tasks" in variants_obj and isinstance(variants_obj["tasks"], list):
                            per_task_variants = [t if isinstance(t, str) else (t.get("name") or t.get("task") or str(t)) for t in variants_obj["tasks"]]
                        print(f"[INIT] Loaded {len(per_task_variants)} variants for task '{task_name_field}' from {per_task_variants_path}")
                    except Exception as e:
                        print(f"[WARN] Could not load variants for task {task_name_field}: {e}")

                # resolve reference pose dir from the preferred subset(s) only
                for subset in preferred_subsets:
                    for candidate_name in task_name_candidates:
                        candidate_ref = tasks_root / subset / "referencePics" / candidate_name
                        if candidate_ref.exists():
                            per_task_reference_dir = str(candidate_ref)
                            break
                    if per_task_reference_dir:
                        break
                if not per_task_reference_dir and not args.task_subset:
                    # fallback to task entry or global arg
                    if isinstance(task_obj, dict) and task_obj.get("reference_dir"):
                        per_task_reference_dir = str(task_obj.get("reference_dir"))
                    else:
                        per_task_reference_dir = args.reference_pose_dir

                print(f"\n[RUN-TASK] Running task: id='{task_id}' name='{task_name_field}' variants={('yes' if per_task_variants else 'no')} reference_dir={per_task_reference_dir}")

                # per-task iterations loop
                for i in range(1, args.iterations + 1):
                    print(f"\n[RUN] Task {task_name_field} Iteration {i}/{args.iterations}")

                    # Determine task instruction for this iteration
                    current_task = task_name_field
                    if per_task_variants:
                        if (i - 1) < len(per_task_variants):
                            current_task = per_task_variants[i - 1]
                        else:
                            current_task = per_task_variants[-1]
                    elif task_variants:
                        # global variants (only used when not run_all_tasks)
                        if (i - 1) < len(task_variants):
                            current_task = task_variants[i - 1]
                        else:
                            current_task = task_variants[-1]

                    print(f"[RUN] Using task: {current_task}")

                    should_start_policy = run_teleop_setup_phase(
                        robot=multi_robot,
                        teleop=teleop,
                        key_reader=key_reader,
                        iteration_idx=i,
                        total_iterations=args.iterations,
                        reference_pose_dir=per_task_reference_dir,
                        overlay_alpha=args.overlay_alpha,
                        teleop_overlay_fps=args.teleop_overlay_fps,
                        zoom_factor=args.zoom,
                        setup_preview_window=args.setup_preview_window,
                        reset_mode=args.reset_mode,
                        reset_robot_action=reset_robot_action,
                    )
                    if not should_start_policy:
                        finished_all_iterations = False
                        break

                    should_continue = run_policy_phase(
                        robot=multi_robot,
                        policy=policy,
                        preprocess=preprocess,
                        postprocess=postprocess,
                        ds_features=ds_features,
                        task=current_task,
                        action_names=action_names,
                        device=device,
                        fps=args.fps,
                        max_seconds=args.policy_seconds,
                        key_reader=key_reader,
                        policy_type=args.policy_type,
                        robot_type=robot_type,
                        output_video_dir=args.output_video_dir,
                        policy_run_name=policy_run_name,
                        iteration_idx=i,
                    )
                    if not should_continue:
                        finished_all_iterations = False
                        break
                if not finished_all_iterations:
                    break

            if finished_all_iterations and teleop is not None:
                run_final_teleop_buffer_phase(
                    robot=multi_robot,
                    teleop=teleop,
                    key_reader=key_reader,
                    min_buffer_seconds=args.final_teleop_buffer_seconds,
                )

            print("[CLEANUP] Disconnecting followers...")
            try:
                multi_robot.disconnect()
            except Exception:
                pass

    except KeyboardInterrupt:
        print("\n[STOP] Keyboard interrupt.")
    finally:
        print("[CLEANUP] Disconnecting devices...")
        if teleop is not None:
            try:
                teleop.disconnect()
            except Exception:
                pass
        print("[DONE]")


if __name__ == "__main__":
    main()
