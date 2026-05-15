"""
Ant Colony Optimization (ACO) scheduler for task offloading.

Each ant selects a destination node using a pheromone-guided stochastic
probability rule (analogous to the ACS/AS-based edge selection):

  P(j | i) ∝ τ_{device,j}^α  ·  η_j^β

where:
  τ_{device,j} = pheromone level for device → node j
  η_j          = 1 / estimated_latency_j   (heuristic desirability)
  α            = pheromone exponent (default 1)
  β            = heuristic exponent  (default 2)

Pheromone update (after all ants complete their tour):
  τ_{device,j} ← (1 - ρ) · τ_{device,j} + Δτ_j
  Δτ_j = Σ_{ants that chose j}  1 / F(x_{ant})

References:
  Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization
  by a colony of cooperating agents. IEEE Transactions on Systems, Man, and
  Cybernetics-Part B (Cybernetics), 26(1), 29–41.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from code.src.algorithms.base_scheduler import BaseScheduler
from code.src.core.task import HealthcareTask


class ACOScheduler(BaseScheduler):
    """
    Ant Colony Optimization scheduler.

    Parameters
    ----------
    topology          : NetworkTopology
    n_ants            : int   — colony size per iteration (default 20)
    max_iter          : int   — ACO iterations per scheduling call (default 30)
    evaporation_rate  : float — ρ, pheromone evaporation (default 0.1)
    alpha_pheromone   : float — α, pheromone weight in probability (default 1)
    beta_heuristic    : float — β, heuristic weight in probability (default 2)
    tau_init          : float — initial pheromone value (default 1.0)
    seed              : int
    offload_history   : dict
    """

    def __init__(
        self,
        topology,
        n_ants: int = 20,
        max_iter: int = 30,
        evaporation_rate: float = 0.1,
        alpha_pheromone: float = 1.0,
        beta_heuristic: float = 2.0,
        tau_init: float = 1.0,
        seed: int = 42,
        offload_history: Optional[dict] = None,
    ):
        super().__init__(topology, offload_history)
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.rho = evaporation_rate
        self.alpha = alpha_pheromone
        self.beta = beta_heuristic
        self.tau_init = tau_init
        self._rng = np.random.default_rng(seed)

        self._n_nodes = len(self._candidate_nodes)
        self._idx_to_node: List[int] = self._candidate_nodes

        # Pheromone matrix: shape (n_devices, n_candidate_nodes)
        # Indexed by device_id (mapped via _device_to_row)
        # We maintain a dict: device_id → np.ndarray of shape (n_nodes,)
        self._pheromones: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Pheromone management
    # ------------------------------------------------------------------

    def _get_pheromones(self, device_id: int) -> np.ndarray:
        """Retrieve or initialise pheromone vector for a device."""
        if device_id not in self._pheromones:
            self._pheromones[device_id] = np.full(
                self._n_nodes, self.tau_init, dtype=float
            )
        return self._pheromones[device_id]

    def _update_pheromones(
        self,
        device_id: int,
        ant_choices: List[int],
        ant_costs: List[float],
    ) -> None:
        """
        Pheromone evaporation + deposit:
          τ_j ← (1-ρ)·τ_j + Σ_{ants→j} (1/F_ant)
        """
        tau = self._pheromones[device_id]
        # Evaporate
        tau *= (1.0 - self.rho)
        # Deposit
        for idx, cost in zip(ant_choices, ant_costs):
            if cost > 1e-12:
                tau[idx] += 1.0 / cost
            else:
                tau[idx] += 1.0 / 1e-12   # cap very low cost

        # Floor pheromones to avoid collapse
        tau = np.clip(tau, 1e-6, None)
        self._pheromones[device_id] = tau

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def select_node(self, task: HealthcareTask) -> int:
        """Run ACO to select the best node for this task."""
        if self._n_nodes == 1:
            node_id = self._idx_to_node[0]
            self.record_decision(task.device_id, node_id)
            return node_id

        lat_bounds, eng_bounds = self.estimate_feasible_bounds(task)

        best_node_id, _ = self._run_aco(task, lat_bounds, eng_bounds)
        self.record_decision(task.device_id, best_node_id)
        return best_node_id

    # ------------------------------------------------------------------
    # ACO core
    # ------------------------------------------------------------------

    def _run_aco(
        self,
        task: HealthcareTask,
        lat_bounds: Tuple[float, float],
        eng_bounds: Tuple[float, float],
    ) -> Tuple[int, float]:
        """
        Execute ACO for one task.

        Returns
        -------
        (best_node_id, best_cost)
        """
        # Pre-compute heuristic values η_j = 1 / latency_j
        heuristics = self._compute_heuristics(task, lat_bounds, eng_bounds)

        tau = self._get_pheromones(task.device_id)

        global_best_idx = -1
        global_best_cost = float('inf')

        for _ in range(self.max_iter):
            ant_choices: List[int] = []
            ant_costs: List[float] = []

            for _ in range(self.n_ants):
                idx, cost = self._ant_select(tau, heuristics, task, lat_bounds, eng_bounds)
                ant_choices.append(idx)
                ant_costs.append(cost)

                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_idx = idx

            self._update_pheromones(task.device_id, ant_choices, ant_costs)
            tau = self._pheromones[task.device_id]

        if global_best_idx < 0:
            global_best_idx = 0

        best_node_id = self._idx_to_node[global_best_idx]
        return best_node_id, global_best_cost

    # ------------------------------------------------------------------
    # Ant selection rule
    # ------------------------------------------------------------------

    def _ant_select(
        self,
        tau: np.ndarray,
        heuristics: np.ndarray,
        task: HealthcareTask,
        lat_bounds: Tuple[float, float],
        eng_bounds: Tuple[float, float],
    ) -> Tuple[int, float]:
        """
        Probabilistic node selection for one ant:
          P(j) = (τ_j^α · η_j^β) / Σ_k (τ_k^α · η_k^β)
        """
        attractiveness = (tau ** self.alpha) * (heuristics ** self.beta)
        total = attractiveness.sum()

        if total < 1e-12:
            probs = np.ones(self._n_nodes) / self._n_nodes
        else:
            probs = attractiveness / total

        idx = int(self._rng.choice(self._n_nodes, p=probs))
        node_id = self._idx_to_node[idx]
        cost, _, _, _ = self.evaluate_node(task, node_id, lat_bounds, eng_bounds)
        return idx, cost

    # ------------------------------------------------------------------
    # Heuristic pre-computation
    # ------------------------------------------------------------------

    def _compute_heuristics(
        self,
        task: HealthcareTask,
        lat_bounds: Tuple[float, float],
        eng_bounds: Tuple[float, float],
    ) -> np.ndarray:
        """
        η_j = 1 / estimated_latency_j  (heuristic desirability).
        Uses a fast latency estimate (ignores queue delay for speed).
        """
        heuristics = np.zeros(self._n_nodes, dtype=float)
        src_node = self.topology.get_node(task.device_id)

        for k, node_id in enumerate(self._idx_to_node):
            dst_node = self.topology.get_node(node_id)
            try:
                uplink_rate = self.topology.get_uplink_rate(task.device_id, node_id)
                prop_delay = self.topology.get_propagation_delay(task.device_id, node_id)
                t_tx = task.data_size_bits / max(uplink_rate, 1.0)
                t_proc = task.cpu_cycles / max(dst_node.hardware.cpu_freq_hz, 1.0)
                lat = t_tx + prop_delay + t_proc
                heuristics[k] = 1.0 / max(lat, 1e-6)
            except Exception:
                heuristics[k] = 1e-6

        return heuristics
