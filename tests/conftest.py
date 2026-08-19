"""Mocks rclpy so the tests run without ROS2 installed (development PC / CI)."""
import sys, types

for name in ["rclpy"]:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.init = lambda: None
        m.shutdown = lambda: None
        m.spin_once = lambda *a, **k: None
        sys.modules[name] = m

node_mod = types.ModuleType("rclpy.node")
class _Node:
    def __init__(self, *a, **k): pass
    def create_publisher(self, *a, **k): return None
    def create_subscription(self, *a, **k): return None
    def destroy_node(self): pass
node_mod.Node = _Node
sys.modules["rclpy.node"] = node_mod

gm = types.ModuleType("geometry_msgs.msg"); gm.Twist = object
sys.modules["geometry_msgs"] = types.ModuleType("geometry_msgs")
sys.modules["geometry_msgs.msg"] = gm
nm = types.ModuleType("nav_msgs.msg"); nm.Odometry = object
sys.modules["nav_msgs"] = types.ModuleType("nav_msgs")
sys.modules["nav_msgs.msg"] = nm
