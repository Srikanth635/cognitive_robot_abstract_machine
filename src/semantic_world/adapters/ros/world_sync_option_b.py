"""
Option B: On-demand snapshot synchronization via the oldest subscriber.

This module provides a small, focused client that:
- discovers all subscribers to a topic,
- determines the oldest subscriber by reading its birth time parameters,
- requests a full snapshot from that node using a simple Trigger service,
- returns the snapshot payload as a string (e.g., JSON).

Design:
- No project-specific message definitions are required; we use example_interfaces/Trigger for portability.
- Oldest determination requires that cooperating nodes set two parameters at startup:
  - birth_time_sec (int)
  - birth_time_nanosec (int)

Testing:
- See tests in test/test_ros/test_world_sync_option_b.py for an example of two provider nodes and a client selecting the oldest.

Guidelines applied:
- Clear classes over raw functions; short but descriptive names; custom exceptions to make misuse harder.
- Avoid try/except for attribute access; keep interfaces minimal and explicit.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node as RosNode
from rclpy.topic_endpoint_info import TopicEndpointInfo
from rcl_interfaces.srv import GetParameters
from example_interfaces.srv import Trigger


class NoSubscribersFound(Exception):
    """Raised when no subscribers were found for the given topic."""


class SnapshotServiceUnavailable(Exception):
    """Raised when a target node does not expose the expected snapshot service in time."""


@dataclass(frozen=True)
class SelectedNode:
    node_name: str
    node_namespace: str

    @property
    def fully_qualified_name(self) -> str:
        if self.node_namespace == "/":
            return f"/{self.node_name}"
        return f"{self.node_namespace.rstrip('/')}/{self.node_name}"


class OldestSubscriberSelector:
    """Selects the oldest subscriber to a given topic using birth time parameters."""

    def __init__(self, node: RosNode):
        self.node = node

    def list_subscribers(self, topic_name: str) -> List[TopicEndpointInfo]:
        return list(self.node.get_subscriptions_info_by_topic(topic_name))

    def get_birth_time(self, node_name: str, node_namespace: str, timeout_sec: float = 2.0) -> Optional[Tuple[int, int]]:
        full_name = (node_namespace.rstrip('/') + '/' + node_name).replace('//', '/')
        client = self.node.create_client(GetParameters, f"{full_name}/get_parameters")
        try:
            if not client.wait_for_service(timeout_sec=timeout_sec):
                return None
            req = GetParameters.Request()
            req.names = ["birth_time_sec", "birth_time_nanosec"]
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
            result = future.result()
            if result is None or len(result.values) != 2:
                return None
            v_sec, v_nsec = result.values
            # Parameter value type 2 is integer for rcl_interfaces/ParameterType
            # Here we simply try to read integer_value; if defaults unset, it will be 0.
            if v_sec.type != v_nsec.type:
                return None
            return int(v_sec.integer_value), int(v_nsec.integer_value)
        finally:
            self.node.destroy_client(client)

    def select_oldest_subscriber(self, topic_name: str, timeout_sec: float = 2.0) -> SelectedNode:
        infos = self.list_subscribers(topic_name)
        candidates: List[Tuple[Tuple[int, int], TopicEndpointInfo]] = []
        for info in infos:
            bt = self.get_birth_time(info.node_name, info.node_namespace, timeout_sec=timeout_sec)
            if bt is not None:
                candidates.append((bt, info))
        if not candidates:
            raise NoSubscribersFound(f"No eligible subscribers with birth time found on topic: {topic_name}")
        candidates.sort(key=lambda x: (x[0][0], x[0][1]))  # sort by (sec, nsec)
        oldest_info = candidates[0][1]
        return SelectedNode(node_name=oldest_info.node_name, node_namespace=oldest_info.node_namespace)


class WorldSnapshotClient:
    """
    Requests a world snapshot from a selected node using example_interfaces/Trigger service.

    Expected service name relative to the node: "world_snapshot".
    """

    def __init__(self, node: RosNode, service_basename: str = "world_snapshot"):
        self.node = node
        self.service_basename = service_basename
        self.selector = OldestSubscriberSelector(node)

    def request_snapshot(self, selected: SelectedNode, timeout_sec: float = 5.0) -> str:
        srv_name = f"{selected.fully_qualified_name}/{self.service_basename}"
        client = self.node.create_client(Trigger, srv_name)
        try:
            if not client.wait_for_service(timeout_sec=timeout_sec):
                raise SnapshotServiceUnavailable(f"Snapshot service not available: {srv_name}")
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
            result = future.result()
            if result is None or not result.success:
                raise SnapshotServiceUnavailable(f"Snapshot request failed for: {srv_name}")
            return result.message
        finally:
            self.node.destroy_client(client)

    def sync_from_topic(self, topic_name: str, timeout_sec: float = 5.0) -> str:
        selected = self.selector.select_oldest_subscriber(topic_name, timeout_sec=timeout_sec)
        return self.request_snapshot(selected, timeout_sec=timeout_sec)
