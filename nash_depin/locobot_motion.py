"""
locobot_motion.py — LoCoBot (Kobuki base) driver for the cross-hardware runs.

Interface-compatible with RobotMotion (grid_motion.py), so run_e5.py can treat
the LoCoBot exactly like a TurtleBot4.

Differences from RobotMotion / TurtleBot4:
  - No ResetPose: a per-robot origin computed from start_cell is used instead.
  - The odom and cmd_vel topics are read from cfg["locobot_ros"].
  - Odometry uses RELIABLE QoS by default for the Kobuki base.
  - A settling stop is added so the next command does not begin while the base
    is still decelerating.
  - Forward motion holds the target yaw with a soft angular correction.

Note: the Kobuki driver suspends /odom when the base is stationary, which
inflates the measured cell error during long stationary phases. This is a
hardware property, not a control failure; cross-hardware fidelity is therefore
evaluated on the discrete cell-occupancy sequence rather than metric accuracy.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from nash_depin.grid_motion import ACTION_YAW, ACTION_DELTA, yaw_from_quat, ang_diff


class LoCoBotMotion(Node):
    """
    Per-cell motion controller for the LoCoBot (Kobuki base).
    Same public interface as RobotMotion.
    """

    def __init__(self, start_cell: tuple, cfg: dict):
        """
        start_cell : (fila, col) inicial del agente.
        cfg        : full YAML configuration dictionary.
        """
        super().__init__("e6_locobot_motion")

        p = cfg["physical"]
        lp = cfg["locobot_ros"]

        self.cell = p["cell_size_m"]
        self.v_lin = lp["linear_speed"]
        self.v_ang = lp["angular_speed"]
        self.pos_tol = p["pos_tolerance_m"]
        self.yaw_tol = lp.get("yaw_tolerance_rad", 0.12)

        self.min_ang_speed = lp.get("min_angular_speed", 0.08)
        self.rotate_kp = lp.get("rotate_kp", 2.0)

        self.forward_yaw_kp = lp.get("forward_yaw_kp", 1.5)
        self.forward_max_ang = lp.get("forward_max_angular_speed", 0.25)

        self.stop_settle_s = lp.get("stop_settle_s", 0.30)
        self.stop_publish_dt_s = lp.get("stop_publish_dt_s", 0.05)

        # Per-robot origin.
        # Convention:
        #   x_odom = origin_x + col * cell
        #   y_odom = origin_y - row * cell
        row0, col0 = start_cell
        self.origin_x = -col0 * self.cell
        self.origin_y = row0 * self.cell

        cmd_topic = lp["cmd_vel_topic"]
        odom_topic = lp["odom_topic"]

        self.pub = self.create_publisher(Twist, cmd_topic, 10)

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(
            Odometry,
            odom_topic,
            self._odom_cb,
            odom_qos
        )

        self.x = None
        self.y = None
        self.yaw = None
        self._last_odom_t = 0.0

    # ------------------------------------------------------------------ #
    # Odometry
    # ------------------------------------------------------------------ #

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)
        self._last_odom_t = time.time()

    def wait_odom(self, timeout=60.0) -> bool:
        """Waits for the first odometry reading."""
        t0 = time.time()

        while self.x is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.x is not None

    def _odom_ready(self) -> bool:
        """True if valid odometry is already available."""
        return self.x is not None and self.y is not None and self.yaw is not None

    # ------------------------------------------------------------------ #
    # Stop
    # ------------------------------------------------------------------ #

    def _stop(self, settle: bool = True):
        """
        Publishes zero velocity.

        If settle=True, holds zero for a short pause.
        This lets the Kobuki base come to a physical stop before the next
        primitive starts.
        """
        tw = Twist()

        if not settle:
            self.pub.publish(tw)
            return

        t0 = time.time()

        while time.time() - t0 < self.stop_settle_s:
            self.pub.publish(tw)
            rclpy.spin_once(self, timeout_sec=self.stop_publish_dt_s)

        self.pub.publish(tw)

    # ------------------------------------------------------------------ #
    # Low-level primitives
    # ------------------------------------------------------------------ #

    def rotate_to(self, target_yaw: float, timeout=20.0) -> bool:
        """
        Rotates in place to the target yaw.

        Does not rotate if already within tolerance.
        """
        if not self._odom_ready():
            return False

        initial_err = ang_diff(target_yaw, self.yaw)
        if abs(initial_err) < self.yaw_tol:
            self._stop(settle=False)
            return True

        t0 = time.time()

        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)

            if not self._odom_ready():
                continue

            err = ang_diff(target_yaw, self.yaw)

            if abs(err) < self.yaw_tol:
                self._stop(settle=True)
                return True

            tw = Twist()

            tw.angular.z = max(
                -self.v_ang,
                min(self.v_ang, self.rotate_kp * err)
            )

            if abs(tw.angular.z) < self.min_ang_speed:
                tw.angular.z = math.copysign(self.min_ang_speed, tw.angular.z)

            self.pub.publish(tw)

        self._stop(settle=True)
        return False

    def advance(
        self,
        dist: float = None,
        speed: float = None,
        timeout=20.0,
        target_yaw: float = None
    ) -> bool:
        """
        Drives `dist` meters measured by odometry.

        Mantiene el yaw objetivo durante el avance. Esto reduce deriva lateral
        en el LoCoBot.
        """
        if not self._odom_ready():
            return False

        dist = self.cell if dist is None else dist
        speed = self.v_lin if speed is None else speed

        if dist <= self.pos_tol:
            self._stop(settle=False)
            return True

        x0, y0 = self.x, self.y

        target_yaw = self.yaw if target_yaw is None else target_yaw

        t0 = time.time()

        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)

            if (time.time() - self._last_odom_t) > 0.5:
                self._stop(settle=False)
                continue

            dx = self.x - x0
            dy = self.y - y0
            traveled = dx * math.cos(target_yaw) + dy * math.sin(target_yaw)

            if traveled >= dist - self.pos_tol:
                self._stop(settle=True)
                return True

            yaw_err = ang_diff(target_yaw, self.yaw)

            tw = Twist()
            tw.linear.x = speed

            tw.angular.z = max(
                -self.forward_max_ang,
                min(self.forward_max_ang, self.forward_yaw_kp * yaw_err)
            )

            self.pub.publish(tw)

        self._stop(settle=True)
        return False

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def execute_grid_action(self, action: int, speed: float = None) -> bool:
        """
        Executes a grid action.

        Acciones:
            0 = arriba
            1 = abajo
            2 = izquierda
            3 = derecha
            4 = quedarse
        """
        if action not in ACTION_DELTA:
            self.get_logger().error(f"Invalid action: {action}")
            self._stop(settle=True)
            return False

        if action == 4:
            self._stop(settle=False)
            return True

        if not self._odom_ready():
            return False

        target_yaw = ACTION_YAW[action]

        yaw_err = ang_diff(target_yaw, self.yaw)

        if abs(yaw_err) >= self.yaw_tol:
            if not self.rotate_to(target_yaw):
                return False
        else:
            self._stop(settle=False)

        return self.advance(speed=speed, target_yaw=target_yaw)

    def current_cell(self, origin_x: float, origin_y: float):
        """
        Cell (row, col) estimated from odometry.

        Los argumentos origin_x/origin_y se aceptan para mantener la misma
        firma que RobotMotion, pero este driver usa self.origin_x/self.origin_y.
        """
        col = round((self.x - self.origin_x) / self.cell)
        fila = round(-(self.y - self.origin_y) / self.cell)

        return int(fila), int(col)

    def cell_error_m(self, fila: int, col: int, origin_x: float, origin_y: float) -> float:
        """
        Distance between the real pose and the logical cell center.
        """
        cx = self.origin_x + col * self.cell
        cy = self.origin_y - fila * self.cell

        return math.hypot(self.x - cx, self.y - cy)
