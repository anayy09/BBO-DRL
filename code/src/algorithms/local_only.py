"""
Local-Only baseline scheduler.

All tasks are executed on the wearable device itself — no offloading.
This represents the lower bound on network usage and privacy risk, but
the upper bound on wearable energy consumption for compute-heavy tasks.
"""

from __future__ import annotations

from code.src.algorithms.base_scheduler import BaseScheduler
from code.src.core.task import HealthcareTask


class LocalOnlyScheduler(BaseScheduler):
    """
    Trivial scheduler: always assign the task to its originating wearable.

    node_id returned = task.device_id

    Use this as a baseline to measure the cost of NOT offloading.
    """

    def select_node(self, task: HealthcareTask) -> int:
        """Return the wearable node that generated this task."""
        node_id = task.device_id
        self.record_decision(task.device_id, node_id)
        return node_id
