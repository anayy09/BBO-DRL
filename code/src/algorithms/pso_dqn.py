"""
PSO+DQN hybrid scheduler — control condition for Fix A (Fix2.md).

Same algorithmic structure as BBO-DRL (Algorithm 1) with exactly one change:
  - DQN top-K pre-filter is IDENTICAL to BBO-DRL (same epsilon-greedy logic)
  - BBO inner-loop search (lines 4–13 of Algorithm 1) is REPLACED by a PSO
    search over the same K-node subspace AK

PSO hyperparameters are identical to the standalone PSO baseline
(n_particles=30, max_iter=50, inertia=0.7, c1=c2=1.5) as required by Fix A.
DQN architecture, state vector, reward function, and online training are
bit-for-bit identical to BBODRLScheduler, isolating the inner optimiser.

The dispatch_times_ms list records per-task scheduling overhead (state
observation through dispatch decision, excluding the Bellman update)
for Fix C.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler
from src.algorithms.bbo_drl import DQNNetwork, ReplayBuffer
from src.core.cost_function import compute_cost
from src.core.task import HealthcareTask


class PSODQNScheduler(BaseScheduler):
    """
    PSO+DQN hybrid task offloading scheduler.

    DQN compresses the search space to top-K candidate nodes (identical to
    BBODRLScheduler).  PSO refines the selection within that K-node subspace
    (replaces the BBO inner loop).  Everything else — reward, replay, Bellman
    update, epsilon schedule — is identical to BBODRLScheduler.
    """

    STATE_DIM = 6

    def __init__(
        self,
        topology,
        n_candidate_nodes: int = 3,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        gamma: float = 0.95,
        lr: float = 0.001,
        replay_capacity: int = 10_000,
        batch_size: int = 32,
        target_sync_freq: int = 50,
        n_particles: int = 30,
        pso_max_iter: int = 50,
        pso_inertia: float = 0.7,
        pso_c1: float = 1.5,
        pso_c2: float = 1.5,
        seed: int = 42,
        offload_history: Optional[dict] = None,
    ):
        super().__init__(topology, offload_history)

        self.n_candidate_nodes = min(n_candidate_nodes, len(self._candidate_nodes))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_sync_freq = target_sync_freq

        self.n_particles = n_particles
        self.pso_max_iter = pso_max_iter
        self.pso_inertia = pso_inertia
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2

        self._n_nodes = len(self._candidate_nodes)
        self._idx_to_node: List[int] = self._candidate_nodes

        self._online_net = DQNNetwork(
            state_dim=self.STATE_DIM,
            action_dim=self._n_nodes,
            hidden_dim=64,
            seed=seed,
        )
        self._target_net = DQNNetwork(
            state_dim=self.STATE_DIM,
            action_dim=self._n_nodes,
            hidden_dim=64,
            seed=seed,
        )
        self._target_net.copy_weights_from(self._online_net)

        self._replay = ReplayBuffer(capacity=replay_capacity)
        self._step_count = 0
        self._total_loss = 0.0
        self._rng = np.random.default_rng(seed)
        self._pso_rng = np.random.default_rng(seed + 1)

        self._max_rate_bps = 1e9
        self._max_load = 100
        self._max_rtt_s = 0.5

        self.epsilon_history: List[float] = []
        self.dispatch_times_ms: List[float] = []    # Fix C: per-task dispatch timing

    # ------------------------------------------------------------------
    # State construction — identical to BBODRLScheduler
    # ------------------------------------------------------------------

    def get_state(self, task: HealthcareTask, node_id: int) -> np.ndarray:
        try:
            rate = self.topology.get_uplink_rate(task.device_id, node_id)
        except Exception:
            rate = 0.0
        bw_norm = min(rate / self._max_rate_bps, 1.0)

        node = self.topology.get_node(node_id)
        cpu_util = min(node.current_load / self._max_load, 1.0)

        try:
            prop = self.topology.get_propagation_delay(task.device_id, node_id)
        except Exception:
            prop = 0.25
        rtt_norm = min(2.0 * prop / self._max_rtt_s, 1.0)

        node_type = node.node_type
        if node_type == 'cloud':
            battery_norm = 1.0
        elif node_type == 'fog':
            battery_norm = 0.8
        elif node_type == 'edge':
            battery_norm = 0.6
        else:
            battery_norm = 0.3

        return np.array([
            bw_norm, cpu_util, rtt_norm, battery_norm,
            float(task.ci_score), float(task.attack_probability),
        ], dtype=float)

    # ------------------------------------------------------------------
    # Reward — identical to BBODRLScheduler
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        task: HealthcareTask,
        node_id: int,
        latency_s: float,
        energy_j: float,
        privacy_risk: float,
    ) -> float:
        lat_bounds = (0.0, task.max_delay_s * 2.0)
        eng_bounds = (0.0, 0.1)
        cost = compute_cost(latency_s, energy_j, privacy_risk,
                            task.ci_score, lat_bounds, eng_bounds)
        sla_met = latency_s <= task.max_delay_s
        phi = float(task.ci_score)
        if sla_met:
            sla_term = phi * 0.5
        else:
            vr = (latency_s - task.max_delay_s) / max(task.max_delay_s, 1e-6)
            sla_term = -phi * min(vr, 2.0)
        return float(-cost + sla_term)

    # ------------------------------------------------------------------
    # DQN top-K selection — identical to BBODRLScheduler
    # ------------------------------------------------------------------

    def _dqn_select_top_k(self, state: np.ndarray) -> List[int]:
        K = self.n_candidate_nodes
        if self._rng.random() < self.epsilon:
            indices = self._rng.choice(
                self._n_nodes, size=min(K, self._n_nodes), replace=False,
            ).tolist()
        else:
            q_values = self._online_net.forward(state)
            top_k_idx = np.argsort(q_values)[::-1][:K]
            indices = top_k_idx.tolist()
        return indices

    # ------------------------------------------------------------------
    # PSO refinement within K-node subspace (replaces BBO)
    # ------------------------------------------------------------------

    def _pso_refine(
        self,
        task: HealthcareTask,
        candidate_indices: List[int],
        lat_bounds: Tuple[float, float],
        eng_bounds: Tuple[float, float],
    ) -> int:
        K = len(candidate_indices)
        if K == 1:
            return self._idx_to_node[candidate_indices[0]]

        v_max = float(K) / 2.0
        positions = self._pso_rng.uniform(0.0, float(K - 1), size=self.n_particles)
        velocities = self._pso_rng.uniform(-v_max, v_max, size=self.n_particles)

        def _fitness(pos: float) -> float:
            idx_in_k = int(round(float(np.clip(pos, 0.0, K - 1.0))))
            node_id = self._idx_to_node[candidate_indices[idx_in_k]]
            cost, _, _, _ = self.evaluate_node(task, node_id, lat_bounds, eng_bounds)
            return cost

        costs = np.array([_fitness(positions[k]) for k in range(self.n_particles)])
        pbest_pos = positions.copy()
        pbest_cost = costs.copy()
        gbest_idx = int(np.argmin(pbest_cost))
        gbest_pos = float(pbest_pos[gbest_idx])
        gbest_cost = float(pbest_cost[gbest_idx])

        for _ in range(self.pso_max_iter):
            r1 = self._pso_rng.random(self.n_particles)
            r2 = self._pso_rng.random(self.n_particles)
            velocities = (
                self.pso_inertia * velocities
                + self.pso_c1 * r1 * (pbest_pos - positions)
                + self.pso_c2 * r2 * (gbest_pos - positions)
            )
            velocities = np.clip(velocities, -v_max, v_max)
            positions = np.clip(positions + velocities, 0.0, float(K - 1))
            costs = np.array([_fitness(positions[k]) for k in range(self.n_particles)])
            improved = costs < pbest_cost
            pbest_pos = np.where(improved, positions, pbest_pos)
            pbest_cost = np.where(improved, costs, pbest_cost)
            min_idx = int(np.argmin(pbest_cost))
            if pbest_cost[min_idx] < gbest_cost:
                gbest_cost = float(pbest_cost[min_idx])
                gbest_pos = float(pbest_pos[min_idx])

        best_idx_in_k = int(round(float(np.clip(gbest_pos, 0.0, K - 1.0))))
        best_global_idx = candidate_indices[best_idx_in_k]
        return self._idx_to_node[best_global_idx]

    # ------------------------------------------------------------------
    # DQN Bellman update — identical to BBODRLScheduler
    # ------------------------------------------------------------------

    def update_policy(self, batch_size: int = 32) -> float:
        if len(self._replay) < batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self._replay.sample(batch_size)
        q_current = self._online_net.forward(states)
        q_next = self._target_net.forward(next_states)
        max_q_next = q_next.max(axis=1)
        targets = q_current.copy()
        for i in range(batch_size):
            td_target = rewards[i] + (1.0 - dones[i]) * self.gamma * max_q_next[i]
            targets[i, actions[i]] = td_target
        loss = self._online_net.update(states, targets, lr=self.lr)
        self._total_loss += loss
        return loss

    # ------------------------------------------------------------------
    # Main scheduling interface
    # ------------------------------------------------------------------

    def select_node(self, task: HealthcareTask) -> int:
        if self._n_nodes == 1:
            nid = self._idx_to_node[0]
            self.record_decision(task.device_id, nid)
            return nid

        lat_bounds, eng_bounds = self.estimate_feasible_bounds(task)
        representative_node = self._idx_to_node[0]
        state = self.get_state(task, representative_node)

        # Fix C: time dispatch decision (DQN top-K + PSO refine), not Bellman update
        t_dispatch_start = time.perf_counter()

        top_k_indices = self._dqn_select_top_k(state)
        best_node_id = self._pso_refine(task, top_k_indices, lat_bounds, eng_bounds)

        self.dispatch_times_ms.append((time.perf_counter() - t_dispatch_start) * 1000.0)

        _, latency_s, energy_j, privacy_risk = self.evaluate_node(
            task, best_node_id, lat_bounds, eng_bounds,
        )

        action_idx = self._idx_to_node.index(best_node_id)
        reward = self.compute_reward(task, best_node_id, latency_s, energy_j, privacy_risk)
        next_state = self.get_state(task, best_node_id)
        self._replay.push(state, action_idx, reward, next_state, False)

        if len(self._replay) >= self.batch_size:
            self.update_policy(self.batch_size)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        self._step_count += 1

        if self._step_count % self.target_sync_freq == 0:
            self._target_net.copy_weights_from(self._online_net)

        self.record_decision(task.device_id, best_node_id)
        return best_node_id

    def get_diagnostics(self) -> dict:
        return {
            'epsilon': self.epsilon,
            'replay_size': len(self._replay),
            'step_count': self._step_count,
            'avg_loss': self._total_loss / max(self._step_count, 1),
        }
