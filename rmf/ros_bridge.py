"""
Optional ROS 2 bridge — subscribes to the `fleet_states` topic and feeds
data into the FleetRegistry.  Gracefully disabled when rclpy is not available.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rmf.fleet_scanner import FleetRegistry, FleetEntry, RobotInfo

logger = logging.getLogger(__name__)


class ROS2FleetBridge:
    """
    Subscribes to the ROS 2 `fleet_states` topic and merges state into
    FleetRegistry.  Safe to instantiate even if rclpy is not installed —
    start() will log a warning and return immediately.
    """

    def __init__(self, registry: "FleetRegistry") -> None:
        self._registry = registry
        self._available = False
        try:
            import rclpy  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("rclpy not available — ROS 2 fleet_states bridge disabled")

    async def start(self) -> None:
        if not self._available:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._spin)

    def _spin(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rmf_fleet_msgs.msg import FleetState

        rclpy.init()
        node = Node("rmf_ai_agent_bridge")
        loop = asyncio.new_event_loop()

        def callback(msg: FleetState) -> None:
            from rmf.fleet_scanner import FleetEntry, RobotInfo

            robots = {}
            for r in msg.robots:
                robots[r.name] = RobotInfo(
                    name=r.name,
                    model=r.model,
                    mode=str(r.mode.mode),
                    battery_percent=r.battery_percent,
                    location={"x": r.location.x, "y": r.location.y, "level": r.location.level_name},
                )
            entry = FleetEntry(name=msg.name, robots=robots, source="ros2")
            loop.run_until_complete(self._registry.upsert(entry))

        node.create_subscription(FleetState, "fleet_states", callback, 10)
        logger.info("ROS 2 fleet_states subscriber active")
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()
