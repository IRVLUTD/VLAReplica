# VLAReplica

__For full setup instructions with pictures and videos, refer to the [setup docs](https://irvlutd.github.io/VLAReplica/setup-docs/).__

## Repository setup

VLA-REPLICA utilizes a simple Python script for benchmarking as well as the LeRobot library for SO-101 control. 

GPU VRAM usage is heavy during inference, especially for more complex VLAs like pi0.5, so a GPU with at least 24GB VRAM is recommended.

Clone the repository, create a new virtual environment (recommended) and install prequisites listed in the ```environments.yml``` file:

```
git clone https://github.com/IRVLUTD/VLAReplica.git
cd VLAReplica
conda env create -f environment.yml
conda activate vlareplica
```

## Detect camera and USB indices

Since the camera indices on every computer can vary, utilize leRobot's find-cameras command to list out the corresponding index numbers for the RealSense and Vinmooog cameras (run in terminal):
```
lerobot-find-cameras
```
Record the camera indices for two cameras.

Since the USB serial port on every computer can vary, utilize leRobot's find-port command to list out the corresponding serial port of the SO-101 follower arm. Run the following command in a terminal:
```
lerobot-find-port
```
and then unplug the SO-101 USB cable from the computer, and then press ```Enter```. 

The terminal will output something like: ```Device port: /dev/ttyACM1```. 

Record the serial port (e.g. ```/dev/ttyACM1```) for the follower arm.

## SO-101 arm calibration

Calibration video from LeRobot: 

<figure style="text-align: center; margin: 20px auto; max-width: 800px;">
  <video 
    controls 
    preload="metadata" 
    style="width: 75%; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibrate_so101_2.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption style="color: #555; margin-top: 12px; font-weight: bold;">
    Video 1: SO-101 Arm Calibration Procedure.
  </figcaption>
</figure>

1. Calibrate the SO-101 follower according to the [LeRobot Docs](https://huggingface.co/docs/lerobot/so101?setup_motors=Command#calibrate). __Follow the video carefully, and ensure each motor is at the middle position before starting the calibration process.__ 
    * (*Note: This means for the wrist roll motor, the end-effector should be oriented so that the camera is rotated 90° and pointing towards the right side when looking at the end-effector head on*)
    * During calibration, thoroughly rotate each of the six motors to their physical joint limits. Don't forget any motors!

2. After calibration is complete, the ```calibration.json``` file is typically saved to ```~/.cache/huggingface/lerobot/calibration/robots/<your-robot-id>``` in your root folder. Copy the generated calibration JSON file into `vlareplica/calibration/robots/so101_follower` inside your repo directory.
3. Rename it to `so101_follower_arm.json`.

## Camera Calibration

We first utilize an AprilTag mounted at a defined spot with respect to the box to allow general placement of the camera mount. Then, we utilize the idea of an *image overlay* to match the camera pose to the original VLA-Replica box camera pose as closely as possible.

### AprilTag calibration

1. In a new terminal inside the virtual environment, run the calibration script (replace your-top-camera-index with the number you recorded in __Software Installation__):

    ```python calibration/camera/detect_apriltag.py --camera-index <your-top camera-index>```
    
    A GUI window will pop up, displaying the live camera feed alongside the estimated AprilTag pose. 

    <figure style="text-align: center; margin: 20px auto;">
  <img src="https://github.com/IRVLUTD/irvlutd.github.io/blob/main/VLAReplica/setup-docs/images/app/apriltag_gui.png" width=1200 style="height: auto; border-radius: 4px" alt="System overview diagram">
  <figcaption style=" color: #555; margin-top: 8px;">
    AprilTag camera calibration GUI. The live camera feed (left) and the detected AprilTag pose table (right) are shown simultaneously. Adjust the camera position until the pose values match the table below.
  </figcaption>
    </figure>

2. Reach inside the box and physically slide or tilt the camera mount along the PVC pipe until all reported values match the table below as close as possible (some error is acceptable):

    | X (m) | Y (m) | Z (m) | R (deg) | P (deg) | Y (deg) |
    | --- | --- | --- | --- | --- | --- |
    | -0.06 ± 0.01 | -0.39 ± 0.01 | 1.25 ± 0.01 | -18.5 ± 1.0 | 3.0 ± 1.0 | 2.5 ± 1.0 |

    Once satisfied, press `q` to exit the program.

### Image Overlay Calibration

Although the AprilTag pose estimator may output values close to Table A.2, there may still be slight camera misalignment. To solve this, we utilize visual overlay matching (see below) to ensure the camera view is as close as possible to VLA-REPLICA’s original view.

<figure style="text-align: center; margin: 20px auto; max-width: 800px;">
  <video 
    controls 
    preload="metadata" 
    style="width: 75%; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <source src="https://github.com/IRVLUTD/irvlutd.github.io/blob/main/VLAReplica/setup-docs/images/app/visual_calibration.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption style="color: #555; margin-top: 12px; font-weight: bold;">
    Video 2: Image Overlay Calibration Procedure.
  </figcaption>
</figure>

1. First, calibrate the top camera for the second time. Run the following (replacing `your-top-camera-id` with with the number you recorded in __Software Installation__): 
    ```python calibration/camera/overlay.py --overlay-image-folder calibration/camera/referenceImages/top --base-cam <your-top-camera-id>```

    A GUI window will pop up, overlaying the live top camera feed with a wrist view reference image. Match the view of your camera with the reference image by reaching into the box and sliding or tilting the camera mount along the PVC pipe.

2. Next, calibrate the wrist camera. Run the following (replacing `your-wrist-camera-id` with with the number you recorded in __Software Installation__): 
    ```python calibration/camera/overlay.py --overlay-image-folder calibration/camera/referenceImages/wrist --base-cam <your-wrist-camera-id>```

    A GUI window will pop up, overlaying the live wrist camera feed with a top view reference image. Slightly loosen the M3 screw on the wrist camera mount on the SO-101, and match the view of your camera with the reference image by rotating the camera mount along the end effector.

<figure style="text-align: center; margin: 20px auto;">
<img src="https://github.com/IRVLUTD/irvlutd.github.io/blob/main/VLAReplica/setup-docs/images/app/visual_calibration.png" width=1200 style="height: auto; border-radius: 4px" alt="System overview diagram">
<figcaption style=" color: #555; margin-top: 8px;">
Visual calibration GUI. Top camera (top) and wrist camera (bottom) calibration over time. The cameras are adjusted physically until the overlay match the reference image.
</figcaption>
</figure>

__Before the next step, ensure that:__

- All six pose values (x,y,z,R,P,Y) match the targets in the table:
    - | X (m) | Y (m) | Z (m) | R (deg) | P (deg) | Y (deg) |
    | --- | --- | --- | --- | --- | --- |
    | -0.06 ± 0.01 | -0.39 ± 0.01 | 1.25 ± 0.01 | -18.5 ± 1.0 | 3.0 ± 1.0 | 2.5 ± 1.0 |
- Reference images and camera views match almost identically for both top and wrist cameras.

Congrats! The environment setup is complete, and you are ready to start benchmarking your VLA models!

## Evaluation script

Use the evaluation script `benchmark.py` to run a policy across predefined ID or OOD tasks, with predefined reference images. Refer to the table below for all CLI flags.

Currently, the script supports the following models: `{act,smolvla,dit,xvla,pi0,pi05}`. Support for other VLA models will arrive soon. Feel free to modify the script to implement other VLA models of your liking.

Inside your virtual environment, run:
```
python benchmark.py \
  --policy-type pi0 \
  --policy-path lerobot/pi0_base \
  --policy-from-hub \
  --run-all-tasks \
  --task-subset ID \
  --iterations 5 \
  --eval-follower-calib-dirs calibration/robots/so101_follower \
  --eval-follower-ports /dev/ttyACM1 \
  --eval-follower-ids so101_follower_arm \
  --eval-top-indexes 4 \
  --eval-wrist-indexes 14 \
  --reset-mode fixed \
  --reset-action-file arm_reset.json
```

| Flag | Description |
| :--- | :--- |
| `--policy-type <model>` | Selects the policy family to evaluate. Currently supported models: `{act,smolvla,dit,xvla,pi0,pi05}` |
| `--policy-path <path>` | Hugging Face repo ID or local path for the policy checkpoint. |
| `--policy-from-hub` | If `--policy-path` directs to a Hugging Face repo ID, include this flag. Loads policy from Hugging Face Hub instead of local directory. |
| `--run-all-tasks` | Runs evaluation across all 10 VLA-REPLICA tasks from task config, instead of single task. |
| `--task-subset <ID or OOD>` | When using `--run-all-tasks`, restricts evaluation to ID or OOD task subset. |
| `--iterations <number>` | Number of evaluation iterations per task (we used 5 in the paper). |
| `--eval-follower-calib-dirs <path>` | Follower calibration directory. (default: `calibration/robots/so101_follower`). |
| `--eval-follower-ports <serial port>` | Serial port for the follower robot (e.g. `dev/ttyACM1`) |
| `--eval-follower-ids <id>` | Robot ID for the follower arm. (default: `so101_follower_arm`) |
| `--eval-top-indexes <index>` | Top-camera index for the active arm. |
| `--eval-wrist-indexes <index>` | Wrist-camera index for the active arm. |
| `--reset-mode fixed` | Uses a fixed reset action instead of teleoperated leader reset (we enabled this for the paper). |
| `--reset-action-file <path>` | JSON file containing the normalized reset action vector required when `--reset-mode fixed` is used. (default: `arm_reset.json`) |

## Evaluation process

1. After the script loads the corresponding policy and connects successfully to the followers, the follower arm will move to a consistent start position (predetermined in `arm_reset.json`). An openCV GUI will pop up, overlaying the live video feed from the top camera with the proper test scene (i.e. predefined object placements) for that task. 

2. Grab the corresponding objects needed for that scene (i.e. red plate and bread A for the first task) and then move the objects to their reference image positions so that the live camera and overlay image are identical to each other. 
      * <figure style="text-align: center; margin: 20px auto">
    <img src="https://github.com/IRVLUTD/irvlutd.github.io/blob/main/VLAReplica/setup-docs/images/app/benchmark_gui.png" width=1000 style="height: auto; border-radius: 4px" alt="System overview diagram">
    <figcaption style=" color: #555; margin-top: 8px;">
      benchmark.py live video evaluation GUI. The user is currently setting up the scene for the "Put bread on plate" task.
    </figcaption>
      </figure>

3. When the live video feed and overlay image match almost exactly, press `Enter` on the keyboard to start policy inference.
    * During policy evaluations for the VLA-REPLICA paper, each policy is given 90 seconds to complete the task before the iteration ends.  
    * If the policy completes the task before 90 seconds, press `right arrow (➜)` to skip to the setup phase of the next iteration. The SO-101 arm will reset back to the start position.
4. Log success and/or failure behavior for each iteration corresponding to that specific task. The full list of tasks and criteron are listed below.

## ID versus OOD evaluation

- ID tasks use scene layouts close to the training distribution to see how well the model learns.
    - There are 10 ID tasks total, with 5 variants each, for a total of __50 ID iterations__.
- OOD tasks test new colors, counts, or objects to test how well the model generalizes generalization.
    - There are 8 ID tasks total, with 5 variants each, for a total of __40 ID iterations__.

## List of Tasks & Success Criterion

The full list of tasks is located under [Task Reference](https://irvlutd.github.io/VLAReplica/setup-docs/task-reference)

| Task | Goal | Success condition |
| --- | --- | --- |
| Put bread on plate | Place the correct bread on the correct colored plate | Bread is resting on the target plate and the arm returns home |
| Put bowl on coaster | Place the correct bowl on the correct coaster | Correct bowl is on correct coaster and the arm returns home |
| Stack blocks | Stack the target block on the target block | Top block remains in contact for more than 2 seconds |
| Fold towel | Fold the towel in half | Edges are lifted and folded by more than 50% |
| Open oven | Open the oven door | Door stays open for 2+ seconds |
| Clean whiteboard | Wipe the board with the eraser | Eraser wipes 2+ times and is placed next to the board |
| Pour pepper | Pour the required number of shakes | Correct number of shakes poured and object returned |
| Lift bowl | Lift the correct bowl the required number of times | Correct lifting count is completed |
| Press button | Press the button the required number of times | Correct number of presses completed |
| Collect blocks | Put all blocks into the correct box | All blocks are in the target box and the arm returns home |

__For full setup instructions with pictures and videos, refer to the [setup docs](https://irvlutd.github.io/VLAReplica/setup-docs/).__
