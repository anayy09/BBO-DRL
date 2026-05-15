"""
Hybrid Slime Mould Algorithm + Harris Hawks Optimization (SMA-HHO) scheduler.

Combines two state-of-the-art swarm intelligence algorithms:

SMA (Slime Mould Algorithm):
  - Adaptive weight W(t) based on fitness ranking
  - Oscillatory exploration: X_A/X_B selected from the swarm
  - W(best) = 1 + rÂ·log(bF/(bF+wF)+1)  for top half
    W(others) = 1 - rÂ·log(bF/(bF+wF)+1) for bottom half

HHO (Harris Hawks Optimization):
  - Rabbit energy E = 2Â·E0Â·(1 - t/T) models escaping prey
  - |E| â‰¥ 1 â†’ exploration (random perch or rabbit tracking)
  - 0.5 â‰¤ |E| < 1 â†’ soft besiege (gradual approach)
  - |E| < 0.5 â†’ hard besiege (rapid dive)

Hybrid integration:
  At each iteration, each agent updates via SMA first, then the HHO
  energy modulates whether the agent accepts the SMA move (exploration)
  or performs a targeted HHO dive (exploitation).

References:
  Li, S. et al. (2020). Slime mould algorithm: A new method for stochastic
  optimization. Future Generation Computer Systems, 111, 300â€“323.

  Heidari, A.A. et al. (2019). Harris hawks optimization: Algorithm and
  applications. Future Generation Computer Systems, 97, 849â€“872.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler
from src.core.task import HealthcareTask


class HSHHOScheduler(BaseScheduler):
    """
    Hybrid SMA-HHO scheduler.

    Parameters
    ----------
    topology      : NetworkTopology
    n_agents      : int   â€” population size (default 30)
    max_iter      : int   â€” iterations per scheduling call (default 50)
    seed          : int
    offload_history : dict
    """

    def __init__(
        self,
        topology,
        n_agents: int = 30,
        max_iter: int = 50,
        seed: int = 42,
        offload_history: Optional[dict] = None,
    ):
        super().__init__(topology, offload_history)
        self.n_agents = n_agents
        self.max_iter = max_iter
        self._rng = np.random.default_rng(seed)

        self._n_nodes = len(self._candidate_nodes)
        self._idx_to_node: List[int] = self._candidate_nodes

        # Search space bounds: continuous [0, n_nodes - 1]
        self._lb = 0.0
        self._ub = float(self._n_nodes - 1)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def select_node(self, task: HealthcareTask) -> int:
        """Run HS-HHO to select the best node for this task."""
        if self._n_nodes == 1:
            node_id = self._idx_to_node[0]
            self.record_decision(task.device_id, node_id)
            return node_id

        lat_bounds, eng_bounds = self.estimate_feasible_bounds(task)
        best_node_id, _ = self._run_hs_hho(task, lat_bounds, eng_bounds)
        self.record_decision(task.device_id, best_node_id)
        return best_node_id

    # ------------------------------------------------------------------
    # HS-HHO core
    # ------------------------------------------------------------------

    def _run_hs_hho(
        self,
        task: HealthcareTask,
        lat_bounds: Tuple[float, float],
        eng_bounds: Tuple[float, float],
    ) -> Tuple[int, float]:
        """
        Hybrid SMA-HHO main loop.

        Returns
        -------
        (best_node_id, best_cost)
        """
        T = self.max_iter
        n = self.n_agents
        lb, ub = self._lb, self._ub

        # Initialise positions uniformly
        X = self._rng.uniform(lb, ub, size=n)
        fitness = np.array([self._evaluate_pos(p, task, lat_bounds, eng_bounds) for p in X])

        # Best (rabbit) position
        best_idx = int(np.argmin(fitness))
        X_rabbit = float(X[best_idx])
        F_rabbit = float(fitness[best_idx])

        for t in range(1, T + 1):
            # --- SMA weight computation ---
            sorted_idx = np.argsort(fitness)
            best_fit = float(fitness[sorted_idx[0]])
            worst_fit = float(fitness[sorted_idx[-1]])

            W = self._sma_weights(fitness, sorted_idx, best_fit, worst_fit)

            # --- HHO energy (escaping energy) ---
            E0 = self._rng.uniform(-1.0, 1.0)  # random initial energy âˆˆ [-1,1]
            E = 2.0 * E0 * (1.0 - t / T)       # decreasing over time

            # --- Agent position update ---
            new_X = np.empty(n, dtype=float)
            for k in range(n):
                if abs(E) >= 1.0:
                    # ---- Exploration: SMA-dominated ----
                    new_X[k] = self._sma_exploration(X, X_rabbit, W[k], k, t, T)
                elif 0.5 <= abs(E) < 1.0:
                    # ---- Soft besiege (HHO): gradual approach ----
                    new_X[k] = self._hho_soft_besiege(X[k], X_rabbit, E, t, T)
                else:
                    # ---- Hard besiege (HHO): rapid convergence ----
                    new_X[k] = self._hho_hard_besiege(
                        X, X_rabbit, E, fitness, task, lat_bounds, eng_bounds
                    )

            # Clamp to bounds
            new_X = np.clip(new_X, lb, ub)

            # Evaluate new positions
            new_fitness = np.array([
                self._evaluate_pos(p, task, lat_bounds, eng_bounds) for p in new_X
            ])

            # Greedy selection: keep new position only if it improves fitness
            improved = new_fitness < fitness
            X = np.where(improved, new_X, X)
            fitness = np.where(improved, new_fitness, fitness)

            # Update rabbit (global best)
            cur_best_idx = int(np.argmin(fitness))
            if fitness[cur_best_idx] < F_rabbit:
                X_rabbit = float(X[cur_best_idx])
                F_rabbit = float(fitness[cur_best_idx])

        best_node_idx = int(round(np.clip(X_rabbit, lb, ub)))
        best_node_id = self._idx_to_node[best_node_idx]
        return best_node_id, F_rabbit

    # ------------------------------------------------------------------
    # SMA weight computation
    # ------------------------------------------------------------------

    def _sma_weights(
        self,
        fitness: np.ndarray,
        sorted_idx: np.ndarray,
        best_fit: float,
        worst_fit: float,
    ) -> np.ndarray:
        """
        Compute oscillatory SMA weight W for each agent.

        W(i) = 1 + r Â· log( (bF)/(bF + wF) + 1 )  for top half (rank â‰¤ n/2)
        W(i) = 1 - r Â· log( (bF)/(bF + wF) + 1 )  for bottom half

        where r âˆˆ [0,1] is a uniform random number per agent.
        """
        n = len(fitness)
        W = np.zeros(n, dtype=float)
        denom = max(abs(best_fit) + abs(worst_fit), 1e-12)
        log_term = math.log(abs(best_fit) / denom + 1.0)

        for rank, idx in enumerate(sorted_idx):
            r = self._rng.random()
            if rank < n // 2:
                W[idx] = 1.0 + r * log_term    # top half: weight > 1
            else:
                W[idx] = 1.0 - r * log_term    # bottom half: weight < 1
        return W

    # ------------------------------------------------------------------
    # SMA exploration move
    # ------------------------------------------------------------------

    def _sma_exploration(
        self,
        X: np.ndarray,
        X_rabbit: float,
        w: float,
        k: int,
        t: int,
        T: int,
    ) -> float:
        """
        SMA position update (exploration phase):
          p = tanh(|F_k - F_best|) / (|F_k| + Îµ)   â€” simplified to p=t/T
          if rand < p:
              X_k = X_rabbit - r1 * (X_rabbit - w * X_A)  â€” approach food
          else:
              X_k = X_rand               â€” random walk
        """
        n = len(X)
        p = t / T  # simplified oscillation probability

        if self._rng.random() < p:
            # Select two random agents (excluding self)
            candidates = [i for i in range(n) if i != k]
            if len(candidates) >= 2:
                idx_A, idx_B = self._rng.choice(candidates, size=2, replace=False)
                X_A, X_B = float(X[idx_A]), float(X[idx_B])
            else:
                X_A = X_B = X_rabbit
            r1 = self._rng.random()
            return float(X_rabbit - r1 * (X_rabbit - w * X_A + (1 - w) * X_B))
        else:
            # Random exploration between bounds
            return float(self._rng.uniform(self._lb, self._ub))

    # ------------------------------------------------------------------
    # HHO soft besiege
    # ------------------------------------------------------------------

    def _hho_soft_besiege(
        self,
        x: float,
        x_rabbit: float,
        E: float,
        t: int,
        T: int,
    ) -> float:
        """
        Soft besiege with progressive rapid dives (0.5 â‰¤ |E| < 1):
          Î”X = X_rabbit - X_k
          X_new = (X_rabbit - EÂ·|JÂ·X_rabbit - X_k|)

        J = 2Â·(1 - r) â€” random jump strength.
        """
        J = 2.0 * (1.0 - self._rng.random())
        delta = abs(J * x_rabbit - x)
        return float(x_rabbit - E * delta)

    # ------------------------------------------------------------------
    # HHO hard besiege
    # ------------------------------------------------------------------

    def _hho_hard_besiege(
        self,
        X: np.ndarray,
        X_rabbit: float,
        E: float,
        fitness: np.ndarray,
        task: HealthcareTask,
        lat_bounds: Tuple[float, float],
        eng_bounds: Tuple[float, float],
    ) -> float:
        """
        Hard besiege with phased rapid dives (|E| < 0.5):
          Generate candidate dive:  Y = X_rabbit - EÂ·|X_rabbit - X_k|
          Generate candidate jump:  Z = Y + LF(D) â€” LÃ©vy flight perturbation
          Accept whichever has better fitness.
        """
        # Use mean position as representative current agent position
        x_mean = float(np.mean(X))
        Y = X_rabbit - abs(E) * abs(X_rabbit - x_mean)
        Z = Y + self._levy_flight(self._n_nodes)

        Y_clipped = float(np.clip(Y, self._lb, self._ub))
        Z_clipped = float(np.clip(Z, self._lb, self._ub))

        f_Y = self._evaluate_pos(Y_clipped, task, lat_bounds, eng_bounds)
        f_Z = self._evaluate_pos(Z_clipped, task, lat_bounds, eng_bounds)

        return Y_clipped if f_Y <= f_Z else Z_clipped

    # ------------------------------------------------------------------
    # LÃ©vy flight
    # ------------------------------------------------------------------

    def _levy_flight(self, dim: float) -> float:
        """
        LÃ©vy flight step size:
          LF = 0.01 Â· u / |v|^{1/Î²}
        where u ~ N(0, Ïƒ_uÂ²), v ~ N(0, 1), Î² = 1.5.
        """
        beta = 1.5
        sigma_u = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2.0)
            / (math.gamma((1 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
        ) ** (1.0 / beta)

        u = self._rng.normal(0.0, sigma_u)
        v = self._rng.normal(0.0, 1.0)
        step = 0.01 * u / (abs(v) ** (1.0 / beta) + 1e-12)
        return float(step * dim)

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def _evaluate_pos(
        self,
        pos: float,
        task: HealthcareTask,
        lat_bounds: Tuple[float, float],
        eng_bounds: Tuple[float, float],
    ) -> float:
        idx = int(round(np.clip(pos, self._lb, self._ub)))
        node_id = self._idx_to_node[idx]
        cost, _, _, _ = self.evaluate_node(task, node_id, lat_bounds, eng_bounds)
        return cost

