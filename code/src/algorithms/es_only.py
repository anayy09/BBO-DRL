"""
ES-only ablation scheduler (replaces ES-only).

Identical to DQN-ES except the DQN top-K pre-filter is replaced by a 
full exhaustive search over the entire candidate-node set.

This baseline isolates the contribution of the DQN action-space compression
from the exact exhaustive evaluation.
"""

from __future__ import annotations

import time
from typing import List, Optional

from src.algorithms.base_scheduler import BaseScheduler
from src.core.task import HealthcareTask


class ESOnlyScheduler(BaseScheduler):
    """
    Pure Exhaustive Search over the full candidate-node set, no DQN.
    """

    def __init__(
        self,
        topology,
        seed: int = 42,
        offload_history: Optional[dict] = None,
    ):
        super().__init__(topology, offload_history)
        self._idx_to_node: List[int] = self._candidate_nodes
        self._n_nodes = len(self._idx_to_node)
        self.dispatch_times_ms: List[float] = []

    def select_node(self, task: HealthcareTask) -> int:
        if self._n_nodes == 1:
            nid = self._idx_to_node[0]
            self.record_decision(task.device_id, nid)
            return nid

        lat_bounds, eng_bounds = self.estimate_feasible_bounds(task)

        t_start = time.perf_counter()
        
        best_node = -1
        best_cost = float('inf')
        
        for node_id in self._idx_to_node:
            cost, _, _, _ = self.evaluate_node(task, node_id, lat_bounds, eng_bounds)
            if cost < best_cost:
                best_cost = cost
                best_node = node_id
                
        self.dispatch_times_ms.append((time.perf_counter() - t_start) * 1000.0)

        self.record_decision(task.device_id, best_node)
        return best_node
