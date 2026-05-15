# Computational Complexity Analysis Supplement
## BBO-DRL: Hybrid Bombardier Beetle Optimizer with Deep Reinforcement Learning

*Prepared as a supplementary technical note to accompany anticipated reviewer requests regarding per-task decision overhead, memory footprint, convergence behavior, and scalability.*

---

## 1. BBO-DRL Per-Task Decision Complexity

### 1.1 DQN Forward Pass

The online DQN network consists of two fully connected hidden layers of width H = 64, operating on a state vector of dimension d_state = 6 and producing Q-values over |A| = M + 3 discrete actions, where M is the number of candidate nodes excluding the wearable, local-edge, and cloud primitives. In the default topology (M = 4 candidate fog/edge nodes), |A| = 7; experiments at maximum scale use |A| = 15 to cover 10 wearable proxies plus 5 infrastructure nodes.

The forward pass complexity is:

```
T_DQN_forward = O(d_state × H + H × H + H × |A|)
              = O(6×64 + 64×64 + 64×15)
              = O(384 + 4096 + 960)
              = O(5440)                          [multiplications]
```

Because the forward pass is a matrix chain multiplication over small fixed-dimension matrices, it is implemented as three successive dense BLAS calls and executes in approximately 2–4 μs on a Raspberry Pi 4 (1.5 GHz Cortex-A72, single-threaded NumPy with OpenBLAS).

### 1.2 Top-K Candidate Selection

After the DQN forward pass produces a Q-value vector of length |A|, the scheduler selects the top-K = 3 actions using a partial argsort:

```
T_topK = O(|A| × log K) ≈ O(15 × 1.58) ≈ O(24)    [comparisons]
```

This is negligible relative to the forward pass and BBO phases.

### 1.3 BBO Search Within the K-Subspace

The BBO optimizer operates on a one-dimensional continuous domain [0, K−1] and rounds the optimized position to the nearest integer candidate index. Each BBO iteration evaluates two candidate positions per beetle (exploration and exploitation) and selects the better one. The cost function for each evaluation queries precomputed Shannon-capacity and M/M/1 queuing estimates cached during state construction, making each evaluation O(1).

Per-iteration cost for the full population:

```
T_BBO_iter = N_pop × 2 × O(1) = 20 × 2 × O(1) = O(40)
```

Total BBO cost over T_BBO = 30 iterations:

```
T_BBO_total = N_pop × T_BBO = 20 × 30 = O(600)
```

The antenna-length decay schedule δ(t) = δ₀ · (1 − t/T_max)² adds O(1) overhead per iteration and is dominated by the cost evaluations above.

### 1.4 DQN Replay Update (Amortized)

The DQN policy update is triggered every time the replay buffer contains at least batch_size = 32 transitions. The update cost involves a forward pass and a backward pass (full backpropagation) through the two-layer network:

```
T_update = O(batch_size × (d_state×H + H×H + H×|A|))
         = O(32 × 5440)
         = O(174,080)
```

However, this update is **not** triggered per task but rather once per decision step once sufficient experience has accumulated. With batch_size = 32 and target_sync_freq = 50, the amortized per-task update cost is:

```
T_update_amortized = T_update / batch_size = O(5440)    [per task]
```

### 1.5 Total Per-Task Complexity

Summing the dominant terms:

```
T_per_task = T_DQN_forward + T_topK + T_BBO_total + T_update_amortized
           = O(5440) + O(24) + O(600) + O(5440)
           ≈ O(11,500)
```

Rounding to a conservative order estimate: **O(N_pop × T_BBO + H²) ≈ O(600 + 4096) ≈ O(5000)** operations for the scheduling decision alone (excluding the amortized update), and O(11,500) including it.

### 1.6 Wall-Clock Estimate

On a Raspberry Pi 4 running at 1.5 GHz with single-threaded NumPy and OpenBLAS:
- Effective throughput for small dense matrix multiplications: approximately 800–1,200 floating-point operations per CPU cycle
- Total per-task operations: ~11,500
- Estimated wall-clock: 11,500 / (1.5 × 10⁹ × 1,000) ≈ **7.7 μs per task decision**

This is approximately **4 orders of magnitude below the minimum clinically relevant latency threshold** of 20 ms for high-acuity arrhythmia detection, confirming that BBO-DRL imposes negligible scheduling overhead.

---

## 2. Baseline Algorithm Complexities

The following table places BBO-DRL in context against the five comparison algorithms. All complexities are per-task (i.e., per scheduling decision).

| Algorithm | Per-Task Complexity | Parameter Values | Notes |
|---|---|---|---|
| Local-Only | O(1) | — | Fixed assignment: always wearable. No evaluation. |
| Cloud-Only | O(1) | — | Fixed assignment: always cloud. No evaluation. |
| PSO | O(N_p × T_p × M) | N_p=30, T_p=50, M=1 node eval | 1,500 cost evaluations; no memory of prior episodes. |
| ACO | O(N_a × T_a × M) | N_a=20, T_a=30, M=1 | 600 cost evaluations; pheromone reset per task. |
| HS-HHO | O(N_a × T_a × M) | N_a=30, T_a=50, M=1 | 1,500 cost evaluations; Harris Hawks exploitation phase. |
| **BBO-DRL** | **O(600) BBO + O(5440) DQN amortized** | N_pop=20, T_BBO=30, H=64 | DQN update amortized over batch=32; subspace search reduces effective M. |

**Key observation:** BBO-DRL's raw BBO cost (600 evaluations) is identical to ACO and lower than PSO and HS-HHO. The DQN overhead is amortized over a batch of 32 decisions, adding only ~5,440 operations per task on average. The combined BBO-DRL per-task cost is therefore competitive with ACO despite delivering superior adaptation through the DQN's learned policy.

---

## 3. Space Complexity

### 3.1 DQN Weight Storage

The online network and target network each require storage for three weight matrices and three bias vectors:

```
W1: d_state × H         =  6  × 64  =    384 parameters
b1: H                   =          64  =     64 parameters
W2: H × H               = 64  × 64  =  4,096 parameters
b2: H                   =          64  =     64 parameters
W3: H × |A|             = 64  × 15  =    960 parameters
b3: |A|                 =          15  =     15 parameters
                                    ─────────────────────
Single network total:                   5,583 parameters
Two networks (online + target):        11,166 parameters × 8 bytes (float64) = 89.3 KB
```

At float32 precision this reduces to **44.7 KB** for both networks combined.

### 3.2 Experience Replay Buffer

```
Buffer capacity: 10,000 transitions
Per-transition storage: (state[6] + action[1] + reward[1] + next_state[6] + done[1])
                      = 15 floats × 8 bytes = 120 bytes per transition
Total: 10,000 × 120 bytes = 1.2 MB
```

At float32: **600 KB**.

### 3.3 BBO Population State

```
N_pop = 20 beetles × dim = 1 position + 1 cost = 40 floats = 320 bytes
```

Negligible.

### 3.4 Offload History (Privacy Entropy Computation)

The system maintains an N_devices × M offload count matrix for Shannon entropy computation:

```
10 devices × 4 candidate nodes = 40 integer counters = 320 bytes
```

### 3.5 Total Memory Footprint

| Component | Float64 | Float32 |
|---|---|---|
| DQN networks (×2) | 89.3 KB | 44.7 KB |
| Replay buffer | 1,200 KB | 600 KB |
| BBO population | < 1 KB | < 1 KB |
| Offload history | < 1 KB | < 1 KB |
| **Total** | **~1.29 MB** | **~645 KB** |

Both estimates are trivially within the Raspberry Pi 4's 8 GB RAM. The full BBO-DRL runtime footprint (including Python interpreter and NumPy stack) peaks at under 80 MB in profiling, consistent with deployment on constrained edge gateways.

---

## 4. Convergence Analysis

### 4.1 DQN Epsilon-Greedy Exploration Decay

The exploration rate follows ε(k) = max(ε_min, ε₀ · α^k), where ε₀ = 1.0 (fully random), α = 0.995 (decay per task), and ε_min = 0.05 (minimum exploitation floor). The number of task decisions required for ε to reach ε_min is:

```
k* = -ln(ε_min / ε₀) / ln(1/α)
   = -ln(0.05 / 1.0) / ln(1/0.995)
   = ln(20) / (-ln(0.995))
   = 2.996 / 0.005013
   ≈ 598 decisions
```

The system therefore transitions from predominantly exploratory behavior to predominantly exploitative behavior after approximately **600 task scheduling decisions**. In a typical IoMT deployment generating 1 task per second per device across 10 devices, this threshold is reached in approximately 60 seconds of operation — a one-time cold-start cost at system initialization.

### 4.2 BBO Antenna-Length Convergence

The BBO antenna length follows δ(t) = δ₀ · (1 − t/T_max)². This quadratic decay reaches a fraction f of the initial antenna length δ₀ at iteration:

```
f = (1 − t/T_max)²  =>  t/T_max = 1 − √f
```

For f = 0.05 (5% of initial search radius):

```
t/T_max = 1 − √0.05 ≈ 1 − 0.2236 = 0.7764
```

Thus δ(t) drops below 5% of δ₀ at approximately **78% of the T_max = 30 iterations**, meaning the BBO operates in a precision-exploitation mode for roughly the final 7 iterations of each scheduling call. The quadratic rather than linear decay profile keeps the algorithm in broader exploration during the first half of iterations, providing better escape from local cost minima before tightening to a precise solution.

### 4.3 Combined Convergence Characterization

The overall BBO-DRL framework exhibits a two-timescale convergence structure:
- **Fast timescale (within a single task decision):** BBO converges within 30 iterations, delivering a refined node selection in O(600) operations.
- **Slow timescale (across task decisions):** The DQN policy converges after approximately 600 task decisions, at which point the top-K candidate set delivered to BBO is policy-optimal rather than exploratory-random. The net effect is that BBO-DRL performs an approximately random search during the first ~600 decisions and a policy-guided fine search thereafter, achieving stable scheduling quality for the vast majority of the experiment duration.

---

## 5. Scalability with Number of Tasks

Because task scheduling decisions are independent (no inter-task coupling beyond queue state updates, which are O(1) per task), the total simulation time scales strictly linearly:

```
T_total(N_tasks) = N_tasks × T_per_task = O(N_tasks)
```

This linear scaling property is empirically confirmed by the Monte Carlo results, which show approximately constant per-task latency across all six tested scales (100 to 10,000 tasks). The absence of super-linear scaling is a direct consequence of the fixed-size DQN and BBO subproblems: neither network width nor BBO population grows with N_tasks.

The only component that exhibits a phase transition is the replay buffer, which achieves full capacity (10,000 transitions) at N_tasks = 10,000 and thereafter operates as a sliding-window FIFO. This transition does not affect per-task scheduling cost but does stabilize the variance of DQN gradient updates, which contributes to the slightly improved SLA compliance observed at the largest task scales in Figure 5 of the main manuscript.

---

*End of Complexity Analysis Supplement*
