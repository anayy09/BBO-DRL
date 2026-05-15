"""
Cloud-Only baseline scheduler.

All tasks are offloaded to the cloud server regardless of task type,
network state, or SLA constraints.  This represents the naive maximum-
offloading policy and serves as an upper bound on transmission energy
and end-to-end latency (due to WAN round-trip time).
"""

from __future__ import annotations

from code.src.algorithms.base_scheduler import BaseScheduler
from code.src.core.task import HealthcareTask


class CloudOnlyScheduler(BaseScheduler):
    """
    Trivial scheduler: always assign every task to the cloud node.

    The cloud node is identified as the node with node_type == 'cloud',
    which has the highest node_id by construction in our topology.
    """

    def __init__(self, topology, offload_history=None):
        super().__init__(topology, offload_history)
        # Identify and cache the cloud node ID
        cloud_nodes = [
            nid
            for nid, node in topology.nodes.items()
            if node.node_type == 'cloud'
        ]
        if not cloud_nodes:
            raise RuntimeError(
                "CloudOnlyScheduler requires a 'cloud' node in the topology."
            )
        self._cloud_node_id = max(cloud_nodes)  # highest ID by convention

    def select_node(self, task: HealthcareTask) -> int:
        """Return the cloud node ID for every task."""
        node_id = self._cloud_node_id
        self.record_decision(task.device_id, node_id)
        return node_id
