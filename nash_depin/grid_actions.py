"""
grid_actions.py — Grid action constants (no ROS dependency).

Kept in a separate module so that the pure decision logic of
experiments/run_e5.py can be imported and replayed without rclpy installed
(scripts/sim_frozen.py, tests/). grid_motion.py imports the same constants,
so the physical execution and the simulation cannot drift apart.

E2 Q-tables use 5 ABSOLUTE grid actions:
    0=up(-row), 1=down(+row), 2=left(-col), 3=right(+col), 4=stay
"""

import math

# Target heading (yaw, rad) per grid action, in the odometry frame.
# Convention: x_odom = col * cell, y_odom = -row * cell (row 0 at the top).
#   right (+col) -> yaw 0 ; left (-col) -> yaw pi
#   up (-row) -> yaw +pi/2 ; down (+row) -> yaw -pi/2
ACTION_YAW = {0: math.pi / 2, 1: -math.pi / 2, 2: math.pi, 3: 0.0}
ACTION_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
