from pathlib import Path

from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader

config = SO101FollowerConfig(
    port="/dev/ttyACM1", # Change to correct serial port for your SO-101
    id="so101_follower_arm", 
)

follower = SO101Follower(config)
follower.connect(calibrate=False)
follower.calibrate()

local_calibration_dir = Path(__file__).resolve().parent / config.id
local_calibration_dir.mkdir(parents=True, exist_ok=True)
local_calibration_fpath = local_calibration_dir / "calibration.json"

if hasattr(follower, "_save_calibration"):
    follower._save_calibration(local_calibration_fpath)

follower.disconnect()



# config = SO101LeaderConfig(
#     port="/dev/ttyACM3",
#     id="so101_leader_arm_videoTest",
# )

# leader = SO101Leader(config)
# leader.connect(calibrate=False)
# leader.calibrate()
# leader.disconnect()