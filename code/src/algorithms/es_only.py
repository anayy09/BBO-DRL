"""
BBO-only ablation scheduler  (Fix 2 — required for any hybrid claim).

Identical to BBO-DRL except the DQN top-K pre-filter (Algorithm 1, line 3)
is replaced by a full search over the entire candidate-node set.  The
Bombardier-Beetle continuous optimiser then performs the exact same
antenna-based refinement over a 1-D index in [0, N-1] rather than [0, K-1].

This baseline isolates the contribution of the BBO inner-loop optimiser
from the DQN action-space compression.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler
from src.algorithms.bbo_drl import BBOOptimizer
from src.core.task import HealthcareTask


class BBOOnlyScheduler(BaseScheduler):
    """
    Pure BBO optimiser over the full candidate-node set, no DQN.

    Parameters mirror BBODRLScheduler so that the comparison isolates the
    DQN contribution.
    """

    def __init__(
        self,
        topology,
        n_pop: int = 20,
        max_iter: int = 30,
        delta0: float = 1.0,
        seed: int = 42,
        offload_history: Optional[dict] = None,
    ):
        super().__init__(topology, offload_history)
        self._idx_to_node: List[int] = self._candidate_nodes
        self._n_nodes = len(self._idx_to_node)
        self._bbo = BBOOptimizer(
            n_pop=n_pop, max_iter=max_iter,
            delta0=delta0, seed=seed,
        )
        self.dispatch_times_ms: List[float] = []    # Fix C: per-task dispatch timing

    def select_node(self, task: HealthcareTask) -> int:
        if self._n_nodes == 1:
            nid = self._idx_to_node[0]
            self.record_decision(task.device_id, nid)
            return nid

        lat_bounds, eng_bounds = self.estimate_feasible_bounds(task)

        N = self._n_nodes

        def bbo_cost(x: np.ndarray) -> float:
            idx = int(round(float(np.clip(x[0], 0.0, N - 1.0))))
            node_id = self._idx_to_node[idx]
            cost, _, _, _ = self.evaluate_node(
                task, node_id, lat_bounds, eng_bounds,
            )
            return cost

        # Fix C: time the full dispatch decision (BBO search only, no DQN/Bellman)
        t_start = time.perf_counter()
        bounds: List[Tuple[float, float]] = [(0.0, float(N - 1))]
        best_pos, _ = self._bbo.optimize(bbo_cost, bounds, dim=1)
        self.dispatch_times_ms.append((time.perf_counter() - t_start) * 1000.0)

        best_idx = int(round(float(np.clip(best_pos[0], 0.0, N - 1.0))))
        node_id = self._idx_to_node[best_idx]

        self.record_decision(task.device_id, node_id)
        return node_id
