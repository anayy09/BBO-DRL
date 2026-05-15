"""
Particle Swarm Optimization (PSO) scheduler for task offloading.

Standard PSO formulation adapted for discrete node selection:

  v_k(t+1) = w · v_k(t)
            + c1 · r1 · (pbest_k - x_k(t))
            + c2 · r2 · (gbest   - x_k(t))
  x_k(t+1) = x_k(t) + v_k(t+1)

Position is continuous in [0, n_nodes-1]; discretised to integer node index
for fitness evaluation.  Velocities are clamped to [-v_max, +v_max].

References:
  Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization.
  IEEE ICNN, 1942–1948.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from code.src.algorithms.base_scheduler import BaseScheduler
from code.src.core.task import HealthcareTask


class PSOScheduler(BaseScheduler):
    """
    PSO-based task offloading scheduler.

    Parameters
    ----------
    topology        : NetworkTopology
    n_particles     : int   — swarm size (default 30)
    max_iter        : int   — iterations per scheduling decision (default 50)
    inertia         : float — w, inertia weight (default 0.7)
    c1              : float — cognitive coefficient (default 1.5)
    c2              : float — social coefficient (default 1.5)
    seed            : int   — random seed for reproducibility
    offload_history : dict  — per-device offloading history
    """

    def __init__(
        self,
        topology,
        n_particles: int = 30,
        max_iter: int = 50,
        inertia: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int = 42,
        offload_history: Optional[dict] = None,
    ):
        super().__init__(topology, offload_history)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self._rng = np.random.default_rng(seed)

        # Candidate nodes include all non-wearable nodes
        self._n_nodes = len(self._candidate_nodes)
        # Map from index → actual node_id
        self._idx_to_node: List[int] = self._candidate_nodes

        # Velocity maximum: span half the search space per step
        self._v_max = float(self._n_nodes) / 2.0

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def select_node(self, task: HealthcareTask) -> int:
        """
        Run PSO to find the best offloading node for this task.

        Returns
        -------
        node_id : int
        """
        if self._n_nodes == 1:
            node_id = self._idx_to_node[0]
            self.record_decision(task.device_id, node_id)
            return node_id

        # Estimate normalisation bounds once per task
        lat_bounds, eng_bounds = self.estimate_feasible_bounds(task)

        best_node_id, _ = self._run_pso(task, lat_bounds, eng_bounds)
        self.record_decision(task.device_id, best_node_id)
        return best_node_id

    # ------------------------------------------------------------------
    # PSO core
    # ------------------------------------------------------------------

    def _run_pso(
        self,
        task: HealthcareTask,
        lat_bounds: Tuple[float, float],
        eng_bounds: Tuple[float, float],
    ) -> Tuple[int, float]:
        """
        Execute PSO optimisation.

        Returns
        -------
        (best_node_id, best_cost)
        """
        n = self._n_nodes

        # Initialise positions uniformly in [0, n-1], velocities in [-v_max, v_max]
        positions = self._rng.uniform(0.0, float(n - 1), size=self.n_particles)
        velocities = self._rng.uniform(-self._v_max, self._v_max, size=self.n_particles)

        # Evaluate initial fitness
        costs = np.array([
            self._fitness(positions[k], task, lat_bounds, eng_bounds)
            for k in range(self.n_particles)
        ])

        pbest_pos = positions.copy()
        pbest_cost = costs.copy()

        gbest_idx = int(np.argmin(pbest_cost))
        gbest_pos = float(pbest_pos[gbest_idx])
        gbest_cost = float(pbest_cost[gbest_idx])

        for _ in range(self.max_iter):
            r1 = self._rng.random(self.n_particles)
            r2 = self._rng.random(self.n_particles)

            # Velocity update
            velocities = (
                self.inertia * velocities
                + self.c1 * r1 * (pbest_pos - positions)
                + self.c2 * r2 * (gbest_pos - positions)
            )
            # Clamp velocities
            velocities = np.clip(velocities, -self._v_max, self._v_max)

            # Position update + boundary clamping
            positions = np.clip(positions + velocities, 0.0, float(n - 1))

            # Evaluate fitness
            costs = np.array([
                self._fitness(positions[k], task, lat_bounds, eng_bounds)
                for k in range(self.n_particles)
            ])

            # Update personal bests
            improved = costs < pbest_cost
            pbest_pos = np.where(improved, positions, pbest_pos)
            pbest_cost = np.where(improved, costs, pbest_cost)

            # Update global best
            min_idx = int(np.argmin(pbest_cost))
            if pbest_cost[min_idx] < gbest_cost:
                gbest_cost = float(pbest_cost[min_idx])
                gbest_pos = float(pbest_pos[min_idx])

        best_idx = int(round(np.clip(gbest_pos, 0, n - 1)))
        best_node_id = self._idx_to_node[best_idx]
        return best_node_id, gbest_cost

    # ------------------------------------------------------------------
    # Fitness function
    # ------------------------------------------------------------------

    def _fitness(
        self,
        position: float,
        task: HealthcareTask,
        lat_bounds: Tuple[float, float],
        eng_bounds: Tuple[float, float],
    ) -> float:
        """
        Convert continuous position to discrete node index and evaluate cost.
        """
        idx = int(round(np.clip(position, 0, self._n_nodes - 1)))
        node_id = self._idx_to_node[idx]
        cost, _, _, _ = self.evaluate_node(task, node_id, lat_bounds, eng_bounds)
        return cost
