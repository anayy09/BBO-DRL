"""
Central hyperparameter configuration for the Q1 BBO-DRL paper.

All Monte Carlo runs, ablations, and statistical analyses import from
this single file so that the experimental protocol is reproducible and
unambiguous.

Hyperparameter values are also surfaced in Table II of the manuscript
and in the supplementary complexity_analysis.md.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED:      int = 42
PER_RUN_SEED_FN   = lambda run_id, n_tasks: GLOBAL_SEED + run_id * 1000 + n_tasks


# ---------------------------------------------------------------------------
# Monte Carlo protocol  (Fix 1, Fix 5)
# ---------------------------------------------------------------------------
N_RUNS:          int        = 30
TASK_SCALES:     list[int]  = [100, 500, 1000, 2000, 5000]
PRIMARY_SCALE:   int        = 1000          # scale used for Table III, ablations
N_WEARABLES:     int        = 10
N_FOG_NODES:     int        = 3


# ---------------------------------------------------------------------------
# BBO-DRL hyperparameters  (Fix 7: explicit epsilon decay schedule)
# ---------------------------------------------------------------------------
# Epsilon-greedy exploration schedule:
#   epsilon(t+1) = max(epsilon_min, epsilon(t) * epsilon_decay)
# i.e., a discrete-step geometric (exponential) decay applied once per
# scheduling decision.  Closed-form: epsilon(t) = epsilon_0 * decay^t
# (clipped at epsilon_min).
EPSILON_INIT:        float = 1.0
EPSILON_DECAY:       float = 0.995          # per scheduling decision
EPSILON_MIN:         float = 0.05
# Pre-computed convergence task counts under this schedule
# (epsilon_0 * decay^t = target  ->  t = log(target/epsilon_0) / log(decay))
#   t(eps=0.10) = log(0.10 / 1.0) / log(0.995) = 460
#   t(eps=0.05) = log(0.05 / 1.0) / log(0.995) = 598
EPSILON_T_AT_0_10:   int   = 460
EPSILON_T_AT_0_05:   int   = 598

# DQN architecture / training
DQN_HIDDEN_DIM:      int   = 64
DQN_LR:              float = 0.001
DQN_GAMMA:           float = 0.95
DQN_BATCH_SIZE:      int   = 32
DQN_REPLAY_CAPACITY: int   = 10_000
DQN_TARGET_SYNC:     int   = 50              # steps between target-net updates

# BBO inner search
BBO_POP:             int   = 20
BBO_MAX_ITER:        int   = 30
BBO_DELTA0:          float = 1.0
BBO_TOP_K:           int   = 3               # DQN top-K pre-filter (Algorithm 1, line 3)


# ---------------------------------------------------------------------------
# CI-adaptive weight functions  (Fix 6: ablation defines four conditions)
# ---------------------------------------------------------------------------
# Default ("proposed") non-linear weights are defined analytically in
# core/cost_function.py with constants ALPHA_E, BETA_L, GAMMA_P.
ALPHA_E:             float = 3.0
BETA_L:              float = 4.0
GAMMA_P:             float = 2.0

# Threshold separating high-criticality vs low-criticality regime for the
# step-weight ablation condition.
STEP_CI_THRESHOLD:   float = 0.5


# ---------------------------------------------------------------------------
# Privacy guard (Fix 8)
# ---------------------------------------------------------------------------
# A flow is classified as a "traffic-analysis attack" when its empirical
# offload-entropy ratio H/H_max falls below this threshold for the source
# device.  Threshold tuned to maximise F1 on a 20% calibration split.
PRIVACY_ENTROPY_THRESHOLD: float = 0.55


# ---------------------------------------------------------------------------
# Real-trace evaluation (Fix 10)
# ---------------------------------------------------------------------------
MITBIH_N_RUNS:       int   = 30
MITBIH_PAYLOAD_BITS: int   = int(7.2 * 1024 * 8)   # 7.2 KB = 58_982 bits
MITBIH_DEADLINE_S:   float = 0.150
MITBIH_RHO:          float = 0.9


# ---------------------------------------------------------------------------
# Statistical testing  (Fix 4)
# ---------------------------------------------------------------------------
# Significance tested on four metrics across all baselines vs BBO-DRL.
# Bonferroni correction: alpha_corrected = 0.05 / (n_baselines * n_metrics)
STAT_ALPHA:          float = 0.05
STAT_METRICS:        list[str] = [
    'avg_latency_ms', 'avg_energy_mj',
    'avg_privacy_risk', 'sla_violation_pct',
]
STAT_BASELINES:      list[str] = ['PSO', 'ACO', 'HS-HHO', 'BBO-only', 'DQN-only']
# Bonferroni denominator = len(STAT_BASELINES) * len(STAT_METRICS) = 20


# ---------------------------------------------------------------------------
# Algorithm registry  (Fix 2: includes ablations)
# ---------------------------------------------------------------------------
def get_full_algorithm_registry():
    """
    Return the complete algorithm registry including BBO-only and DQN-only
    ablations.  Imported lazily to avoid circular imports at module load.
    """
    from src.algorithms.aco import ACOScheduler
    from src.algorithms.bbo_drl import BBODRLScheduler
    from src.algorithms.bbo_only import BBOOnlyScheduler
    from src.algorithms.cloud_only import CloudOnlyScheduler
    from src.algorithms.dqn_only import DQNOnlyScheduler
    from src.algorithms.hs_hho import HSHHOScheduler
    from src.algorithms.local_only import LocalOnlyScheduler
    from src.algorithms.pso import PSOScheduler

    return {
        'BBO-DRL':    BBODRLScheduler,
        'BBO-only':   BBOOnlyScheduler,
        'DQN-only':   DQNOnlyScheduler,
        'PSO':        PSOScheduler,
        'ACO':        ACOScheduler,
        'HS-HHO':     HSHHOScheduler,
        'Local-Only': LocalOnlyScheduler,
        'Cloud-Only': CloudOnlyScheduler,
    }


def summary() -> str:
    """Return a human-readable summary of all hyperparameters."""
    return (
        f"BBO-DRL Q1 hyperparameter summary\n"
        f"  Monte Carlo: N_RUNS={N_RUNS}  scales={TASK_SCALES}\n"
        f"  Epsilon: init={EPSILON_INIT} decay={EPSILON_DECAY} "
        f"min={EPSILON_MIN}\n"
        f"           eps<0.10 after {EPSILON_T_AT_0_10} tasks; "
        f"eps<0.05 after {EPSILON_T_AT_0_05} tasks\n"
        f"  DQN: hidden={DQN_HIDDEN_DIM}  lr={DQN_LR}  gamma={DQN_GAMMA}  "
        f"batch={DQN_BATCH_SIZE}\n"
        f"  BBO: pop={BBO_POP}  iter={BBO_MAX_ITER}  K={BBO_TOP_K}  "
        f"delta0={BBO_DELTA0}\n"
        f"  CI weights (non-linear, default): "
        f"alpha_E={ALPHA_E} beta_L={BETA_L} gamma_P={GAMMA_P}\n"
        f"  Privacy guard entropy threshold: {PRIVACY_ENTROPY_THRESHOLD}\n"
        f"  Statistical tests: alpha={STAT_ALPHA}  "
        f"Bonferroni denom={len(STAT_BASELINES)*len(STAT_METRICS)}\n"
    )


if __name__ == '__main__':
    print(summary())
