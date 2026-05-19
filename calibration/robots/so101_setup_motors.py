from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

config = SO101FollowerConfig(
    port="/dev/ttyACM4",
    id="so101_follower_arm",
)
follower = SO101Follower(config)
follower.setup_motors()


config = SO101LeaderConfig(
    port="/dev/ttyACM5",
    id="so101_leader_arm",
)
leader = SO101Leader(config)
leader.setup_motors()