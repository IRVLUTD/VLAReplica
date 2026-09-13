#!/usr/bin/env python3
"""Test MolmoAct2 with this project's direct adapter, not lerobot-rollout.

This environment predates LeRobot's ``lerobot-rollout`` command and does not
bundle the MolmoAct2 policy. ``benchmark.py`` supplies a compatible direct
adapter, so this launcher keeps the intended two-camera setup without changing
the installed LeRobot version.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely test MolmoAct2 through the local benchmark adapter.")
    parser.add_argument("--task", required=True, help="Natural-language task sent to MolmoAct2.")
    parser.add_argument("--follower-port", default="/dev/ttyACM0")
    parser.add_argument("--top-index", type=int, default=6, help="Third-person/top camera index.")
    parser.add_argument("--side-index", type=int, default=0, help="Second camera index (normally wrist view).")
    parser.add_argument("--duration", type=int, default=30, help="Rollout duration in seconds.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--policy-path", default="allenai/MolmoAct2-SO100_101")
    parser.add_argument("--calibration-dir", default="calibration/robots/so101_follower")
    parser.add_argument("--follower-id", default="so101_follower_arm")
    parser.add_argument("--num-steps", type=int, default=10, help="MolmoAct2 flow-solver steps per action.")
    parser.add_argument("--actions-per-chunk", type=int, default=3, help="Predicted actions to execute before refreshing cameras.")
    parser.add_argument("--max-joint-step-deg", type=float, default=3.0)
    parser.add_argument("--no-hardware", action="store_true", help="Use synthetic observations instead of opening devices.")
    parser.add_argument("--enable-hardware-actions", action="store_true", help="DANGEROUS: send policy actions to the follower.")
    parser.add_argument("--enable-cuda-graph", action="store_true", help="Enable CUDA graph capture after basic testing.")
    parser.add_argument("--execute", action="store_true", help="Actually run the local benchmark command.")
    args = parser.parse_args()

    if args.enable_hardware_actions and not args.execute:
        parser.error("--enable-hardware-actions requires --execute")

    benchmark = Path(__file__).with_name("benchmark.py")
    command = [
        sys.executable,
        str(benchmark),
        "--policy-type=molmoact2",
        f"--policy-path={args.policy_path}",
        "--policy-from-hub",
        f"--task={args.task}",
        "--iterations=1",
        f"--policy-seconds={args.duration}",
        f"--fps={args.fps}",
        "--zoom=1.0",
        "--reset-mode=none",
        f"--eval-follower-calib-dirs={args.calibration_dir}",
        f"--eval-follower-ports={args.follower_port}",
        f"--eval-follower-ids={args.follower_id}",
        f"--eval-top-indexes={args.top_index}",
        f"--eval-wrist-indexes={args.side_index}",
        "--top-width=640",
        "--top-height=480",
        "--wrist-width=640",
        "--wrist-height=480",
        f"--wrist-fps={args.fps}",
        f"--molmoact2-num-steps={args.num_steps}",
        f"--molmoact2-actions-per-chunk={args.actions_per_chunk}",
        f"--molmoact2-max-joint-step-deg={args.max_joint_step_deg}",
        "--molmoact2-dry-run",
    ]
    if args.no_hardware:
        command.append("--no-hardware")
    if args.enable_hardware_actions:
        command.append("--molmoact2-enable-hardware-actions")
    if args.enable_cuda_graph:
        command.append("--molmoact2-enable-cuda-graph")

    print("MolmoAct2 local-adapter command:")
    print(shlex.join(command))
    if not args.execute:
        print("\nPreview only: no hardware command was sent.")
        print("Use --execute for a camera/robot dry run; press ENTER after manually placing the arm.")
        print("Add --no-hardware for a fully synthetic test, or --enable-hardware-actions only after dry-run validation.")
        return 0

    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
