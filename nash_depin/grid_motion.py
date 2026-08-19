"""
grid_motion.py — Closed-loop per-cell motion primitives (odometry).
Nash-DePIN — TurtleBot4 driver.

Replaces open-loop fixed-time motion with simple feedback control using /odom,
which the TurtleBot4 already publishes. No Nav2, no map: rotate to the target
heading, then drive until the measured distance is covered. This keeps the
per-cell error below 0.15 m across 50 episodes, which open-loop control cannot
do because drift accumulates.

Action translation (E2 Q-tables -> differential-drive robot):
    E2 Q-tables use 5 ABSOLUTE grid actions:
        0=up(-row), 1=down(+row), 2=left(-col), 3=right(+col), 4=stay
    The robot executes each as a compound primitive: rotate to heading + drive
    one cell.

Coordinate convention (calibrated per arena):
    row grows toward -Y in odometry, column grows toward +X.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


# Action constants live in grid_actions.py (no ROS dependency) so that the
# pure decision logic can be replayed without rclpy. Re-exported here.
from nash_depin.grid_actions import ACTION_YAW, ACTION_DELTA


def yaw_from_quat(q) -> float:
    """Extracts yaw from an odometry quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def ang_diff(a: float, b: float) -> float:
    """Angular difference normalized to [-pi, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


class RobotMotion(Node):
    """
    Per-cell motion controller for ONE robot.
    Subscribes to odom, publishes cmd_vel. Blocking methods (internal spin).
    """

    def __init__(self, namespace: str, cfg: dict):
        super().__init__(f"e5_motion_{namespace}")

        p = cfg["physical"]
        self.cell = p["cell_size_m"]
        self.v_lin = p["linear_speed"]
        self.v_ang = p["angular_speed"]
        self.pos_tol = p["pos_tolerance_m"]
        self.yaw_tol = p["yaw_tolerance_rad"]

        self.pub = self.create_publisher(
            Twist,
            f"/{namespace}/{cfg['ros']['cmd_vel_topic']}",
            10
        )

        # The TurtleBot4 publishes /odom with BEST_EFFORT QoS.
        # This explicit profile is required to receive odometry.
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(
            Odometry,
            f"/{namespace}/{cfg['ros']['odom_topic']}",
            self._odom_cb,
            odom_qos
        )

        self.x = self.y = self.yaw = None

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)

    def wait_odom(self, timeout=60.0) -> bool:
        """Waits for the first odometry reading."""
        t0 = time.time()
        while self.x is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.x is not None

    def _stop(self):
        self.pub.publish(Twist())

    def rotate_to(self, target_yaw: float, timeout=20.0) -> bool:
        """Rotates in place to the target yaw (P-control)."""
        t0 = time.time()

        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)

            err = ang_diff(target_yaw, self.yaw)

            if abs(err) < self.yaw_tol:
                self._stop()
                return True

            tw = Twist()
            tw.angular.z = max(-self.v_ang, min(self.v_ang, 2.0 * err))

            # Minimum speed to overcome static friction.
            if abs(tw.angular.z) < 0.15:
                tw.angular.z = math.copysign(0.15, tw.angular.z)

            self.pub.publish(tw)

        self._stop()
        return False

    def advance(self, dist: float = None, speed: float = None, timeout=20.0) -> bool:
        """Drives straight for `dist` meters measured by odometry."""
        dist = self.cell if dist is None else dist
        speed = self.v_lin if speed is None else speed

        x0, y0 = self.x, self.y
        t0 = time.time()

        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)

            traveled = math.hypot(self.x - x0, self.y - y0)

            if traveled >= dist - self.pos_tol:
                self._stop()
                return True

            tw = Twist()
            tw.linear.x = speed
            self.pub.publish(tw)

        self._stop()
        return False

    def execute_grid_action(self, action: int, speed: float = None) -> bool:
        """
        Executes a grid action: rotate to heading + drive one cell.
        action=4 (stay) does not move. Returns True if completed.
        """
        if action == 4:
            return True

        if not self.rotate_to(ACTION_YAW[action]):
            return False

        return self.advance(speed=speed)

    def current_cell(self, origin_x: float, origin_y: float):
        """Cell (row, col) estimated from odometry. Used for the drift check."""
        col = round((self.x - origin_x) / self.cell)
        fila = round(-(self.y - origin_y) / self.cell)
        return int(fila), int(col)

    def cell_error_m(self, fila: int, col: int, origin_x: float, origin_y: float) -> float:
        """Distance between the real pose and the center of the logical cell."""
        cx = origin_x + col * self.cell
        cy = origin_y - fila * self.cell
        return math.hypot(self.x - cx, self.y - cy)
