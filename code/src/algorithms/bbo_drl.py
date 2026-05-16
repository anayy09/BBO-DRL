"""
Hybrid BBO-DRL Scheduler â€” core research contribution.

Architecture:
  1. DQN (numpy-based, no PyTorch dependency):
     - State: [bandwidth_norm, cpu_util_norm, rtt_norm, battery_norm, ci, attack_prob]
     - Action: selects top-K candidate node indices
     - Reward: r = -F(x) + Î¦Â·penalty_SLA  (CI-modulated)

  2. Bombardier Beetle Optimizer (BBO):
     - Refines the DQN-selected K-node subspace using antenna-based search
     - Exploration: random antenna perturbation around best position
     - Exploitation: gradient-free step toward the best known position
     - Î´(t) = Î´_0 Â· (1 - t/T_max)Â² â€” adaptive antenna length

  3. Hybrid flow:
     - DQN narrows search to K â‰ª N candidate nodes
     - BBO performs fine-grained continuous optimisation within that subspace
     - DQN policy is updated online via experience replay (DQN training)

References:
  Mnih, V. et al. (2015). Human-level control through deep reinforcement
  learning. Nature, 518(7540), 529â€“533.

  Wang, Y. et al. (2022). Bombardier beetle optimizer: A new metaheuristic
  algorithm for continuous optimization. IEEE Access, 10, 35182â€“35196.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler
from src.core.cost_function import (
    compute_cost,
    compute_normalized_weights,
    compute_offload_energy,
    compute_offload_latency,
    estimate_bounds,
)
from src.core.task import HealthcareTask


# ===========================================================================
# Experience Replay Buffer
# ===========================================================================

class ReplayBuffer:
    """
    Fixed-capacity circular experience replay buffer for DQN training.

    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state.copy(), int(action), float(reward),
                            next_state.copy(), bool(done)))

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Sample a random mini-batch.

        Returns
        -------
        (states, actions, rewards, next_states, dones) as numpy arrays.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=float),
            np.array(actions,     dtype=int),
            np.array(rewards,     dtype=float),
            np.array(next_states, dtype=float),
            np.array(dones,       dtype=float),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ===========================================================================
# Lightweight Numpy-based DQN Network
# ===========================================================================

class DQNNetwork:
    """
    Two-hidden-layer fully connected network implemented in pure NumPy.

    Architecture: input â†’ hidden1 (ReLU) â†’ hidden2 (ReLU) â†’ output (linear)
    Weights initialised with Xavier (Glorot) uniform initialisation.

    Parameters
    ----------
    state_dim  : int â€” dimension of input state vector
    action_dim : int â€” number of discrete actions (candidate nodes)
    hidden_dim : int â€” neurons per hidden layer
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 5,
        hidden_dim: int = 64,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)

        # Xavier (Glorot) uniform: limit = sqrt(6 / (fan_in + fan_out))
        def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        self.W1 = _xavier(state_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = _xavier(hidden_dim, hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = _xavier(hidden_dim, action_dim)
        self.b3 = np.zeros(action_dim)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x : np.ndarray of shape (state_dim,) or (batch, state_dim)

        Returns
        -------
        Q-values: np.ndarray of shape (action_dim,) or (batch, action_dim)
        """
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        h1 = self._relu(x @ self.W1 + self.b1)    # (batch, hidden_dim)
        h2 = self._relu(h1 @ self.W2 + self.b2)   # (batch, hidden_dim)
        q = h2 @ self.W3 + self.b3                 # (batch, action_dim)

        return q[0] if single else q

    def update(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        lr: float = 0.001,
    ) -> float:
        """
        Single SGD update step using MSE loss and backpropagation.

        Parameters
        ----------
        states  : np.ndarray (batch, state_dim)
        targets : np.ndarray (batch, action_dim) â€” target Q-values
        lr      : float â€” learning rate

        Returns
        -------
        loss : float â€” mean squared error
        """
        batch = states.shape[0]

        # --- Forward pass (save intermediates for backprop) ---
        z1 = states @ self.W1 + self.b1          # (batch, hidden_dim)
        h1 = self._relu(z1)
        z2 = h1 @ self.W2 + self.b2              # (batch, hidden_dim)
        h2 = self._relu(z2)
        q  = h2 @ self.W3 + self.b3             # (batch, action_dim)

        # --- Loss: MSE ---
        diff = q - targets
        loss = float(np.mean(diff ** 2))

        # --- Backward pass ---
        # dL/dq
        dq = 2.0 * diff / batch                  # (batch, action_dim)

        # Layer 3 gradients
        dW3 = h2.T @ dq                          # (hidden_dim, action_dim)
        db3 = dq.sum(axis=0)
        dh2 = dq @ self.W3.T                     # (batch, hidden_dim)

        # Layer 2 gradients (ReLU derivative)
        dz2 = dh2 * (z2 > 0).astype(float)
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T

        # Layer 1 gradients (ReLU derivative)
        dz1 = dh1 * (z1 > 0).astype(float)
        dW1 = states.T @ dz1
        db1 = dz1.sum(axis=0)

        # --- SGD parameter update (gradient clipping at norm=5) ---
        def _clip_and_apply(param: np.ndarray, grad: np.ndarray) -> np.ndarray:
            norm = float(np.linalg.norm(grad))
            if norm > 5.0:
                grad = grad * (5.0 / norm)
            return param - lr * grad

        self.W1 = _clip_and_apply(self.W1, dW1)
        self.b1 = _clip_and_apply(self.b1, db1)
        self.W2 = _clip_and_apply(self.W2, dW2)
        self.b2 = _clip_and_apply(self.b2, db2)
        self.W3 = _clip_and_apply(self.W3, dW3)
        self.b3 = _clip_and_apply(self.b3, db3)

        return loss

    def copy_weights_from(self, other: 'DQNNetwork') -> None:
        """Copy weights from another DQNNetwork (for target network sync)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


# ===========================================================================
# Bombardier Beetle Optimizer (BBO)
# ===========================================================================

class BBOOptimizer:
    """
    Bombardier Beetle Optimizer â€” antenna-based continuous optimiser.

    The beetle explores using two virtual antennae symmetrically placed
    around its current position.  The antenna that senses a lower cost
    triggers a move in that direction.

    Update rules:
      Exploration:
        X_right = X + r1 * Î´(t) * b  (right antenna)
        X_left  = X - r1 * Î´(t) * b  (left antenna)
        X_k â† X + r1*(X_best - X_k) + r2*Î´(t)*sign(rand-0.5)

      Exploitation:
        step = (X_best - X_k) / ||X_best - X_k||
        X_k â† X + r3 * step * (X_best - X_k)

      Antenna decay:
        Î´(t) = Î´_0 Â· (1 - t/T_max)Â²

    Parameters
    ----------
    n_pop    : int   â€” number of beetles
    max_iter : int   â€” optimisation iterations
    delta0   : float â€” initial antenna length
    """

    def __init__(
        self,
        n_pop: int = 20,
        max_iter: int = 30,
        delta0: float = 1.0,
        seed: int = 0,
    ):
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.delta0 = delta0
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        cost_fn,
        bounds: List[Tuple[float, float]],
        dim: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Minimise cost_fn over a continuous search space.

        Parameters
        ----------
        cost_fn : callable(x: np.ndarray) â†’ float
        bounds  : list of (lb, ub) per dimension
        dim     : int â€” number of dimensions

        Returns
        -------
        (best_position, best_cost)
        """
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)

        # Initialise population uniformly within bounds
        population = lb + self._rng.random((self.n_pop, dim)) * (ub - lb)
        costs = np.array([cost_fn(population[k]) for k in range(self.n_pop)])

        best_idx = int(np.argmin(costs))
        X_best = population[best_idx].copy()
        F_best = float(costs[best_idx])

        for t in range(1, self.max_iter + 1):
            # Antenna length (adaptive decay)
            delta = self.delta0 * ((1.0 - t / self.max_iter) ** 2)
            delta = max(delta, 1e-6)

            new_pop = np.empty_like(population)
            new_costs = np.empty(self.n_pop, dtype=float)

            for k in range(self.n_pop):
                X_k = population[k]

                # --- Exploration candidate ---
                r1 = self._rng.random(dim)
                r2 = self._rng.random(dim)
                sign_vec = np.sign(self._rng.random(dim) - 0.5)
                X_explore = X_k + r1 * (X_best - X_k) + r2 * delta * sign_vec
                X_explore = np.clip(X_explore, lb, ub)

                # --- Exploitation candidate ---
                direction = X_best - X_k
                dist = float(np.linalg.norm(direction))
                if dist > 1e-12:
                    step_dir = direction / dist
                else:
                    step_dir = self._rng.random(dim) - 0.5
                r3 = float(self._rng.random())
                X_exploit = X_k + r3 * step_dir * dist
                X_exploit = np.clip(X_exploit, lb, ub)

                # --- Select better candidate ---
                f_explore = cost_fn(X_explore)
                f_exploit = cost_fn(X_exploit)

                if f_explore <= f_exploit:
                    new_pop[k] = X_explore
                    new_costs[k] = f_explore
                else:
                    new_pop[k] = X_exploit
                    new_costs[k] = f_exploit

            # Update population
            population = new_pop
            costs = new_costs

            # Update global best
            cur_best_idx = int(np.argmin(costs))
            if costs[cur_best_idx] < F_best:
                X_best = population[cur_best_idx].copy()
                F_best = float(costs[cur_best_idx])

        return X_best, F_best


# ===========================================================================
# Hybrid BBO-DRL Scheduler
# ===========================================================================

class BBODRLScheduler(BaseScheduler):
    """
    Hybrid BBO-DRL task offloading scheduler â€” core research contribution.

    DQN compresses the search space to top-K candidate nodes.
    BBO refines the selection within that K-node subspace.
    CI modulates the reward signal to prioritise critical tasks.

    DQN state vector (6-dimensional, all normalised to [0,1]):
      [0] bandwidth_norm   â€” uplink rate / max_rate
      [1] cpu_util_norm    â€” node.current_load / max_load
      [2] rtt_norm         â€” round-trip time estimate / max_rtt
      [3] battery_norm     â€” estimated battery proxy (1 for cloud, varies for others)
      [4] ci               â€” Criticality Index Î¦_i
      [5] attack_prob      â€” adversarial threat p_{atk,i}

    Parameters
    ----------
    topology           : NetworkTopology
    n_candidate_nodes  : int   â€” K, DQN selects top-K (default 3)
    epsilon            : float â€” initial Îµ for Îµ-greedy exploration (default 1.0)
    epsilon_decay      : float â€” decay per call to select_node (default 0.995)
    epsilon_min        : float â€” minimum Îµ (default 0.05)
    gamma              : float â€” DQN discount factor (default 0.95)
    lr                 : float â€” DQN learning rate (default 0.001)
    replay_capacity    : int   â€” replay buffer capacity (default 10_000)
    target_sync_freq   : int   â€” steps between target network updates (default 50)
    seed               : int
    offload_history    : dict
    """

    # State dimension (fixed)
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

        self._n_nodes = len(self._candidate_nodes)
        self._idx_to_node: List[int] = self._candidate_nodes

        # DQN networks (online + target)
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

        # Replay buffer
        self._replay = ReplayBuffer(capacity=replay_capacity)

        # BBO instance
        self._bbo = BBOOptimizer(n_pop=20, max_iter=30, delta0=1.0, seed=seed)

        # Training counters
        self._step_count = 0
        self._total_loss = 0.0
        self._rng = np.random.default_rng(seed)

        # Reference max values for state normalisation
        self._max_rate_bps = 1e9          # 1 Gbps reference
        self._max_load = 100              # max tasks in queue
        self._max_rtt_s = 0.5            # 500 ms reference RTT

        # Previous state/action for replay (per device)
        self._prev: Dict[int, Optional[Tuple[np.ndarray, int]]] = {}

        # Epsilon trajectory (Fix 7: convergence figure)
        self.epsilon_history: List[float] = []

    # ------------------------------------------------------------------
    # State vector construction
    # ------------------------------------------------------------------

    def get_state(self, task: HealthcareTask, node_id: int) -> np.ndarray:
        """
        Build the 6-dimensional normalised state vector for (task, node_id).

        [0] bandwidth_norm â€” uplink rate to node / reference max rate
        [1] cpu_util_norm  â€” current node load / max_load
        [2] rtt_norm       â€” estimated RTT / max_rtt
        [3] battery_norm   â€” battery proxy (1.0 for cloud, lower for edge)
        [4] ci             â€” task Criticality Index Î¦_i
        [5] attack_prob    â€” adversarial threat probability
        """
        # [0] Bandwidth
        try:
            rate = self.topology.get_uplink_rate(task.device_id, node_id)
        except Exception:
            rate = 0.0
        bw_norm = min(rate / self._max_rate_bps, 1.0)

        # [1] CPU utilisation
        node = self.topology.get_node(node_id)
        cpu_util = min(node.current_load / self._max_load, 1.0)

        # [2] RTT estimate (2 Ã— propagation delay)
        try:
            prop = self.topology.get_propagation_delay(task.device_id, node_id)
        except Exception:
            prop = 0.25
        rtt_norm = min(2.0 * prop / self._max_rtt_s, 1.0)

        # [3] Battery proxy: wired/cloud â†’ 1.0, edge/fog â†’ proportional to type
        node_type = node.node_type
        if node_type == 'cloud':
            battery_norm = 1.0
        elif node_type == 'fog':
            battery_norm = 0.8
        elif node_type == 'edge':
            battery_norm = 0.6
        else:
            battery_norm = 0.3   # wearable

        # [4] CI, [5] attack probability
        state = np.array([
            bw_norm,
            cpu_util,
            rtt_norm,
            battery_norm,
            float(task.ci_score),
            float(task.attack_probability),
        ], dtype=float)

        return state

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        task: HealthcareTask,
        node_id: int,
        latency_s: float,
        energy_j: float,
        privacy_risk: float,
    ) -> float:
        """
        CI-modulated reward:
          r = -F(x) + Î¦_i Â· bonus_if_sla_met - Î¦_i Â· penalty_if_sla_violated

        F(x) âˆˆ [0, 1] so r âˆˆ approximately [-2, 1].
        """
        # Estimate bounds for normalisation (approximation based on current topology)
        lat_bounds = (0.0, task.max_delay_s * 2.0)
        eng_bounds = (0.0, 0.1)  # 100 mJ upper reference

        cost = compute_cost(
            latency_s, energy_j, privacy_risk,
            task.ci_score, lat_bounds, eng_bounds,
        )

        sla_met = latency_s <= task.max_delay_s
        phi = float(task.ci_score)

        # SLA bonus/penalty scaled by CI
        if sla_met:
            sla_term = phi * 0.5    # bonus for meeting critical deadlines
        else:
            violation_ratio = (latency_s - task.max_delay_s) / max(task.max_delay_s, 1e-6)
            sla_term = -phi * min(violation_ratio, 2.0)   # penalty proportional to violation

        return float(-cost + sla_term)

    # ------------------------------------------------------------------
    # Main scheduling interface
    # ------------------------------------------------------------------

    def select_node(self, task: HealthcareTask) -> int:
        """
        Hybrid BBO-DRL node selection.

        1. DQN (Îµ-greedy) selects top-K candidate node indices.
        2. BBO refines within those K nodes.
        3. Experience is stored for offline training.
        """
        if self._n_nodes == 1:
            node_id = self._idx_to_node[0]
            self.record_decision(task.device_id, node_id)
            return node_id

        lat_bounds, eng_bounds = self.estimate_feasible_bounds(task)

        # --- Step 1: DQN selects top-K nodes ---
        # Use edge node (or first candidate) as representative for state
        representative_node = self._idx_to_node[0]
        state = self.get_state(task, representative_node)

        top_k_indices = self._dqn_select_top_k(state)

        # --- Step 2: BBO refines within K-node subspace ---
        best_node_id = self._bbo_refine(
            task, top_k_indices, lat_bounds, eng_bounds
        )

        # --- Step 3: Compute actual outcome metrics ---
        _, latency_s, energy_j, privacy_risk = self.evaluate_node(
            task, best_node_id, lat_bounds, eng_bounds
        )

        # --- Step 4: Store transition in replay buffer ---
        action_idx = self._idx_to_node.index(best_node_id)
        reward = self.compute_reward(task, best_node_id, latency_s, energy_j, privacy_risk)
        next_state = self.get_state(task, best_node_id)
        done = False  # continuous task stream â€” no terminal state

        self._replay.push(state, action_idx, reward, next_state, done)

        # --- Step 5: Online DQN training ---
        if len(self._replay) >= self.batch_size:
            self.update_policy(self.batch_size)

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

        self._step_count += 1

        # Periodic target network sync
        if self._step_count % self.target_sync_freq == 0:
            self._target_net.copy_weights_from(self._online_net)

        self.record_decision(task.device_id, best_node_id)
        return best_node_id

    # ------------------------------------------------------------------
    # DQN top-K selection (Îµ-greedy)
    # ------------------------------------------------------------------

    def _dqn_select_top_k(self, state: np.ndarray) -> List[int]:
        """
        Îµ-greedy top-K action selection.

        During exploration: randomly sample K node indices.
        During exploitation: take the K highest Q-value actions.

        Returns
        -------
        List of K node indices (into self._idx_to_node).
        """
        K = self.n_candidate_nodes

        if self._rng.random() < self.epsilon:
            # Random exploration: sample K unique indices
            indices = self._rng.choice(
                self._n_nodes,
                size=min(K, self._n_nodes),
                replace=False,
            ).tolist()
        else:
            # Greedy exploitation: top-K Q-values
            q_values = self._online_net.forward(state)
            # Descending sort; take top K
            top_k_idx = np.argsort(q_values)[::-1][:K]
            indices = top_k_idx.tolist()

        return indices

    # ------------------------------------------------------------------
    # BBO refinement within K-node subspace
    # ------------------------------------------------------------------

    def _bbo_refine(
        self,
        task: HealthcareTask,
        candidate_indices: List[int],
        lat_bounds: Tuple[float, float],
        eng_bounds: Tuple[float, float],
    ) -> int:
        """
        Use BBO to optimise over the continuous subspace [0, K-1] and
        map back to the best discrete node_id.

        The BBO cost function evaluates the multi-objective cost at the
        nearest integer index within candidate_indices.
        """
        K = len(candidate_indices)

        if K == 1:
            return self._idx_to_node[candidate_indices[0]]

        # Map continuous position in [0, K-1] â†’ candidate node
        def bbo_cost(x: np.ndarray) -> float:
            idx_in_k = int(round(float(np.clip(x[0], 0.0, K - 1.0))))
            node_global_idx = candidate_indices[idx_in_k]
            node_id = self._idx_to_node[node_global_idx]
            cost, _, _, _ = self.evaluate_node(task, node_id, lat_bounds, eng_bounds)
            return cost

        bounds = [(0.0, float(K - 1))]
        best_pos, _ = self._bbo.optimize(bbo_cost, bounds, dim=1)

        best_idx_in_k = int(round(float(np.clip(best_pos[0], 0.0, K - 1.0))))
        best_global_idx = candidate_indices[best_idx_in_k]
        return self._idx_to_node[best_global_idx]

    # ------------------------------------------------------------------
    # DQN policy update
    # ------------------------------------------------------------------

    def update_policy(self, batch_size: int = 32) -> float:
        """
        Sample a mini-batch and perform one DQN Bellman update.

        Target: Q_target(s, a) = r + Î³ Â· max_{a'} Q_target(s', a')

        Returns
        -------
        loss : float â€” MSE loss for this update step.
        """
        if len(self._replay) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self._replay.sample(batch_size)

        # Current Q-values from online network
        q_current = self._online_net.forward(states)  # (batch, n_nodes)

        # Target Q-values from target network
        q_next = self._target_net.forward(next_states)  # (batch, n_nodes)
        max_q_next = q_next.max(axis=1)                  # (batch,)

        # Bellman targets
        targets = q_current.copy()
        for i in range(batch_size):
            td_target = rewards[i] + (1.0 - dones[i]) * self.gamma * max_q_next[i]
            targets[i, actions[i]] = td_target

        # Update online network
        loss = self._online_net.update(states, targets, lr=self.lr)
        self._total_loss += loss
        return loss

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        """Return training statistics for logging."""
        return {
            'epsilon': self.epsilon,
            'replay_size': len(self._replay),
            'step_count': self._step_count,
            'avg_loss': (
                self._total_loss / max(self._step_count, 1)
            ),
        }

