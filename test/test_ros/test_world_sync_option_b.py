import threading
import time

import pytest

from semantic_world.testing import rclpy_node


@pytest.fixture
def provider_node_factory():
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from example_interfaces.srv import Trigger
    from std_msgs.msg import String

    class Provider:
        def __init__(self, name: str, birth_sec: int, birth_nsec: int, snapshot_message: str, topic: str):
            self.node: Node = rclpy.create_node(name)
            # Declare birth time parameters used by the selector
            self.node.declare_parameter("birth_time_sec", birth_sec)
            self.node.declare_parameter("birth_time_nanosec", birth_nsec)

            # Subscribe to the given topic so this node appears in the subscriber list
            self.subscription = self.node.create_subscription(String, topic, lambda msg: None, 10)

            # Provide a snapshot service returning the payload
            def handle_trigger(request, response):
                response.success = True
                response.message = snapshot_message
                return response

            # Expose service under the node's fully qualified name to match client expectations
            self.service = self.node.create_service(Trigger, f"{self.node.get_fully_qualified_name()}/world_snapshot", handle_trigger)

            # Spin this node in its own executor
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self.executor.spin, daemon=True, name=f"exec-{name}")
            self.thread.start()

        def shutdown(self):
            self.executor.shutdown()
            self.thread.join(timeout=2.0)
            self.node.destroy_service(self.service)
            self.node.destroy_subscription(self.subscription)
            self.node.destroy_node()

    def factory(name: str, birth_sec: int, birth_nsec: int, snapshot_message: str, topic: str):
        return Provider(name, birth_sec, birth_nsec, snapshot_message, topic)

    return factory


def test_select_oldest_and_request_snapshot(rclpy_node, provider_node_factory):
    from semantic_world.adapters.ros.world_sync_option_b import (
        OldestSubscriberSelector,
        WorldSnapshotClient,
    )

    topic = "/semantic_world/world_model"

    older = provider_node_factory("provider_older", 1, 0, "OLDER_PAYLOAD", topic)
    newer = provider_node_factory("provider_newer", 2, 0, "NEWER_PAYLOAD", topic)

    # Allow ROS graph to discover endpoints
    time.sleep(0.3)

    selector = OldestSubscriberSelector(rclpy_node)
    selected = selector.select_oldest_subscriber(topic)

    # Ensure we selected the older node
    assert selected.node_name == "provider_older"

    client = WorldSnapshotClient(rclpy_node)
    payload = client.request_snapshot(selected)
    assert payload == "OLDER_PAYLOAD"

    older.shutdown()
    newer.shutdown()


def test_no_subscribers_raises(rclpy_node):
    from semantic_world.adapters.ros.world_sync_option_b import OldestSubscriberSelector, NoSubscribersFound

    selector = OldestSubscriberSelector(rclpy_node)
    with pytest.raises(NoSubscribersFound):
        selector.select_oldest_subscriber("/semantic_world/world_model", timeout_sec=0.5)
