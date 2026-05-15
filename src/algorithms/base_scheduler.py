"""
Abstract base class for all task-offloading schedulers.

Every concrete scheduler must implement `select_node(task)` and may
optionally override `evaluate_node` if it needs custom cost components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from src.core.cost_function import (
    compute_cost,
    compute_local_energy,
    compute_local_latency,
    compute_offload_energy,
    compute_offload_latency,
    compute_privacy_risk,
    estimate_bounds,
)
from src.core.network import NetworkTopology
from src.core.task import HealthcareTask


class BaseScheduler(ABC):
    """
    Abstract base for all scheduling algorithms.

    Parameters
    ----------
    topology : NetworkTopology
        The current network graph.
    offload_history : dict, optional
        Per-device history of offloading decisions.
        Structure: {device_id: {node_id: count}}
    """

    def __init__(
        self,
        topology: NetworkTopology,
        offload_history: Optional[Dict[int, Dict[int, int]]] = None,
    ):
        self.topology = topology
        # offload_history[device_id][node_id] = number of tasks sent there
        self.offload_history: Dict[int, Dict[int, int]] = offload_history or {}

        # Pre-compute sorted list of all non-wearable candidate node IDs
        self._candidate_nodes = sorted(
            n.node_id
            for n in topology.nodes.values()
            if n.node_type != 'wearable'
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def select_node(self, task: HealthcareTask) -> int:
        """
        Return the node_id to which this task should be assigned.
        Must be implemented by every concrete scheduler.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared evaluation helper
    # ------------------------------------------------------------------

    def evaluate_node(
        self,
        task: HealthcareTask,
        node_id: int,
        latency_bounds: Tuple[float, float],
        energy_bounds: Tuple[float, float],
    ) -> Tuple[float, float, float, float]:
        """
        Compute (cost, latency_s, energy_j, privacy_risk) for assigning
        *task* to *node_id*.

        For the wearable's own node (local execution):
          - Latency  = C_i / f_local
          - Energy   = κ · C_i · f²
          - Privacy  = 0  (data never leaves the device)

        For remote nodes:
          - Latency  = t_tx + t_prop + t_queue + t_proc
          - Energy   = P_tx · t_tx + P_idle · (L - t_tx)
          - Privacy  = ρ · (1 - H / H_max)

        Parameters
        ----------
        task           : HealthcareTask
        node_id        : int — target node
        latency_bounds : (l_min, l_max) in seconds   — for normalisation
        energy_bounds  : (e_min, e_max) in joules    — for normalisation

        Returns
        -------
        (cost, latency_s, energy_j, privacy_risk)
        """
        topo = self.topology
        src_node = topo.get_node(task.device_id)
        dst_node = topo.get_node(node_id)

        is_local = (node_id == task.device_id)

        # ---- Latency ----
        if is_local:
            latency_s = compute_local_latency(
                task.cpu_cycles,
                src_node.hardware.cpu_freq_hz,
            )
        else:
            uplink_rate = topo.get_uplink_rate(task.device_id, node_id)
            prop_delay = topo.get_propagation_delay(task.device_id, node_id)
            queue_delay = topo.get_queue_delay(node_id)
            latency_s = compute_offload_latency(
                task.data_size_bits,
                task.cpu_cycles,
                uplink_rate,
                prop_delay,
                queue_delay,
                dst_node.hardware.cpu_freq_hz,
            )

        # ---- Energy ----
        if is_local:
            energy_j = compute_local_energy(
                task.cpu_cycles,
                src_node.hardware.cpu_freq_hz,
                src_node.hardware.kappa,
            )
        else:
            uplink_rate = topo.get_uplink_rate(task.device_id, node_id)
            energy_j = compute_offload_energy(
                task.data_size_bits,
                uplink_rate,
                latency_s,
                src_node.hardware.tx_power_w,
                src_node.hardware.idle_power_w,
            )

        # ---- Privacy risk ----
        if is_local:
            privacy_risk = 0.0
        else:
            device_history = self.offload_history.get(task.device_id, {})
            n_available = len(self._candidate_nodes)
            privacy_risk = compute_privacy_risk(
                task.privacy_sensitivity,
                device_history,
                n_available,
            )

        # ---- Composite cost ----
        cost = compute_cost(
            latency_s,
            energy_j,
            privacy_risk,
            task.ci_score,
            latency_bounds,
            energy_bounds,
        )

        return cost, latency_s, energy_j, privacy_risk

    # ------------------------------------------------------------------
    # Shared bound estimation
    # ------------------------------------------------------------------

    def estimate_feasible_bounds(
        self,
        task: HealthcareTask,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Quick sweep over all candidate nodes to get min/max latency &
        energy — used as normalisation bounds in compute_cost.
        """
        latencies: list[float] = []
        energies: list[float] = []

        src_node = self.topology.get_node(task.device_id)

        # Include local execution
        lat_local = compute_local_latency(
            task.cpu_cycles, src_node.hardware.cpu_freq_hz
        )
        eng_local = compute_local_energy(
            task.cpu_cycles,
            src_node.hardware.cpu_freq_hz,
            src_node.hardware.kappa,
        )
        latencies.append(lat_local)
        energies.append(eng_local)

        for nid in self._candidate_nodes:
            dst_node = self.topology.get_node(nid)
            try:
                uplink_rate = self.topology.get_uplink_rate(task.device_id, nid)
                prop_delay = self.topology.get_propagation_delay(task.device_id, nid)
                queue_delay = self.topology.get_queue_delay(nid)
                lat = compute_offload_latency(
                    task.data_size_bits,
                    task.cpu_cycles,
                    uplink_rate,
                    prop_delay,
                    queue_delay,
                    dst_node.hardware.cpu_freq_hz,
                )
                eng = compute_offload_energy(
                    task.data_size_bits,
                    uplink_rate,
                    lat,
                    src_node.hardware.tx_power_w,
                    src_node.hardware.idle_power_w,
                )
                if math.isfinite(lat) and math.isfinite(eng):
                    latencies.append(lat)
                    energies.append(eng)
            except Exception:
                pass

        return estimate_bounds(latencies, energies)

    # ------------------------------------------------------------------
    # History tracking
    # ------------------------------------------------------------------

    def record_decision(self, device_id: int, node_id: int) -> None:
        """Update the offloading history after a scheduling decision."""
        if device_id not in self.offload_history:
            self.offload_history[device_id] = {}
        hist = self.offload_history[device_id]
        hist[node_id] = hist.get(node_id, 0) + 1

    @property
    def candidate_nodes(self):
        return self._candidate_nodes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nodes={len(self._candidate_nodes)})"


import math  # noqa: E402 — placed at bottom to avoid circular import issues
