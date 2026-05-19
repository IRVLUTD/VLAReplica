from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="so101_follower_arm",
)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="so101_leader_arm",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

try:
    while True:
        action = teleop_device.get_action()
        print("Sending action:", action)
        robot.send_action(action)

except KeyboardInterrupt:
    print("Stopping Teleoperation...")
finally: 
    robot.disconnect()
    teleop_device.disconnect()
    print("Disconnected from devices.")