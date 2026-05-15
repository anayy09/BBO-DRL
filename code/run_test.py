"""Quick integration test: run all 6 schedulers on 500 synthetic tasks."""
import sys
BASE = r"c:\Users\sinha\OneDrive - University of Florida\Papers\Bio-Inspired Adaptive Task Offloading System"
sys.path.insert(0, BASE)

from code.src.simulation.topology import build_healthcare_topology
from code.src.simulation.environment import OffloadingEnvironment
from code.src.algorithms.bbo_drl import BBODRLScheduler
from code.src.algorithms.pso import PSOScheduler
from code.src.algorithms.aco import ACOScheduler
from code.src.algorithms.hs_hho import HSHHOScheduler
from code.src.algorithms.local_only import LocalOnlyScheduler
from code.src.algorithms.cloud_only import CloudOnlyScheduler
from code.src.data_ingestion.event_generator import generate_synthetic_tasks
from code.src.core.task import HealthcareTask

topo = build_healthcare_topology(n_wearables=10, n_fog_nodes=3)
tasks_raw = generate_synthetic_tasks(500, seed=42)
tasks = [
    HealthcareTask(
        task_id=t.task_id,
        device_id=t.device_id % 10,
        timestamp=t.timestamp,
        data_size_bits=t.data_size_bits,
        cpu_cycles=t.cpu_cycles,
        max_delay_s=t.max_delay_s,
        privacy_sensitivity=t.privacy_sensitivity,
        ci_score=t.ci_score,
        attack_probability=t.attack_probability,
        source=t.source,
    )
    for t in tasks_raw
]

header = f"{'Algorithm':<12} {'Avg Lat (ms)':>14} {'Avg Energy (mJ)':>16} {'Privacy Risk':>13} {'SLA Viols':>10}"
print(header)
print('-' * 68)

schedulers = [
    ('LocalOnly', LocalOnlyScheduler),
    ('CloudOnly', CloudOnlyScheduler),
    ('PSO', PSOScheduler),
    ('ACO', ACOScheduler),
    ('HS-HHO', HSHHOScheduler),
    ('BBO-DRL', BBODRLScheduler),
]

for name, sched_cls in schedulers:
    sched = sched_cls(topo)
    env = OffloadingEnvironment(topo, sched, n_tasks=500, seed=42)
    results = env.run(tasks)
    avg_lat = sum(x['latency_ms'] for x in results) / len(results)
    avg_eng = sum(x['energy_mj'] for x in results) / len(results)
    avg_priv = sum(x['privacy_risk'] for x in results) / len(results)
    sla_v = sum(1 for x in results if x['sla_violated'])
    row = f"{name:<12} {avg_lat:>14.2f} {avg_eng:>16.4f} {avg_priv:>13.4f} {sla_v:>10}/{len(results)}"
    print(row)

print("\nAll schedulers completed successfully.")
