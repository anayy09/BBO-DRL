"""
Healthcare simulation topology builder.

Topology structure:
  N wearables  â†’  1 edge gateway  â†’  K fog nodes  â†’  1 cloud

Physical placement:
  - Wearables: randomly within 50 m radius of edge gateway (origin)
  - Edge gateway: at origin (0, 0) km
  - Fog nodes: at 1â€“5 km from edge
  - Cloud: at 50 km (abstracted WAN distance)

Node ID assignment:
  - Wearables:     0 â€¦ N-1
  - Edge gateway:  N
  - Fog nodes:     N+1 â€¦ N+K
  - Cloud:         N+K+1
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from src.core.hardware_profiles import (
    CLOUD_SERVER,
    EDGE_GATEWAY_RPI4,
    FOG_NODE,
    WEARABLE_ESP32,
)
from src.core.network import NetworkLink, NetworkNode, NetworkTopology


# ---------------------------------------------------------------------------
# Topology builder
# ---------------------------------------------------------------------------

def build_healthcare_topology(
    n_wearables: int = 10,
    n_fog_nodes: int = 3,
    seed: int = 42,
) -> NetworkTopology:
    """
    Build the IoT-Edge-Cloud healthcare simulation topology.

    Parameters
    ----------
    n_wearables : int
        Number of wearable sensor devices (patients).
    n_fog_nodes : int
        Number of intermediate fog/MEC compute nodes.
    seed : int
        Random seed for reproducible node placement.

    Returns
    -------
    NetworkTopology instance with all nodes and directed links registered.
    """
    rng = random.Random(seed)
    topology = NetworkTopology()

    # ------------------------------------------------------------------
    # 1. Wearable nodes (IoT layer)
    # ------------------------------------------------------------------
    for i in range(n_wearables):
        # Random placement within 50 m radius of edge gateway
        angle_rad = rng.uniform(0.0, 2.0 * math.pi)
        radius_km = rng.uniform(0.005, 0.05)  # 5â€“50 m converted to km
        x_km = radius_km * math.cos(angle_rad)
        y_km = radius_km * math.sin(angle_rad)

        wearable = NetworkNode(
            node_id=i,
            name=f"Wearable-{i}",
            node_type='wearable',
            hardware=WEARABLE_ESP32,
            position_km=(x_km, y_km),
            current_load=0,
            arrival_rate=0.0,
        )
        topology.add_node(wearable)

    # ------------------------------------------------------------------
    # 2. Edge gateway (MEC layer) â€” at origin
    # ------------------------------------------------------------------
    edge_id = n_wearables
    edge_gateway = NetworkNode(
        node_id=edge_id,
        name="Edge-Gateway-RPi4",
        node_type='edge',
        hardware=EDGE_GATEWAY_RPI4,
        position_km=(0.0, 0.0),
        current_load=0,
        arrival_rate=0.0,
    )
    topology.add_node(edge_gateway)

    # ------------------------------------------------------------------
    # 3. Fog nodes (fog/MEC layer) â€” 1â€“5 km from edge
    # ------------------------------------------------------------------
    fog_angles = [
        (2.0 * math.pi * k / n_fog_nodes) for k in range(n_fog_nodes)
    ]
    fog_ids: List[int] = []
    for k in range(n_fog_nodes):
        fog_id = n_wearables + 1 + k
        radius_km = rng.uniform(1.0, 5.0)
        x_km = radius_km * math.cos(fog_angles[k])
        y_km = radius_km * math.sin(fog_angles[k])

        fog_node = NetworkNode(
            node_id=fog_id,
            name=f"Fog-Node-{k}",
            node_type='fog',
            hardware=FOG_NODE,
            position_km=(x_km, y_km),
            current_load=0,
            arrival_rate=0.0,
        )
        topology.add_node(fog_node)
        fog_ids.append(fog_id)

    # ------------------------------------------------------------------
    # 4. Cloud server â€” modelled as 50 km WAN distance from edge
    # ------------------------------------------------------------------
    cloud_id = n_wearables + n_fog_nodes + 1
    cloud_node = NetworkNode(
        node_id=cloud_id,
        name="Cloud-Server",
        node_type='cloud',
        hardware=CLOUD_SERVER,
        position_km=(50.0, 0.0),   # 50 km east of edge
        current_load=0,
        arrival_rate=0.0,
    )
    topology.add_node(cloud_node)

    # ------------------------------------------------------------------
    # 5. Add directed links
    # ------------------------------------------------------------------
    # 5a. Wearable â†’ Edge gateway (Wi-Fi, NLOS typical indoor Î±=3.0)
    for i in range(n_wearables):
        wearable_pos = topology.nodes[i].position_km
        edge_pos = topology.nodes[edge_id].position_km
        dist_km = _euclidean_km(wearable_pos, edge_pos)
        topology.add_link(NetworkLink(
            source_id=i,
            dest_id=edge_id,
            distance_km=max(dist_km, 0.001),   # minimum 1 m
            path_loss_exponent=3.0,              # NLOS indoor/body area
        ))

    # 5b. Wearable â†’ direct to each fog node (multi-hop via edge; modelled
    #     as path loss over combined distance with NLOS, Î±=3.5)
    for i in range(n_wearables):
        for fog_id in fog_ids:
            wearable_pos = topology.nodes[i].position_km
            fog_pos = topology.nodes[fog_id].position_km
            dist_km = _euclidean_km(wearable_pos, fog_pos)
            topology.add_link(NetworkLink(
                source_id=i,
                dest_id=fog_id,
                distance_km=max(dist_km, 0.001),
                path_loss_exponent=3.5,   # NLOS multi-hop
            ))

    # 5c. Wearable â†’ Cloud (via edge + WAN; high path-loss, long distance)
    for i in range(n_wearables):
        topology.add_link(NetworkLink(
            source_id=i,
            dest_id=cloud_id,
            distance_km=50.0,
            path_loss_exponent=2.0,   # LOS modelled for WAN fibre
        ))

    # 5d. Edge â†’ Fog nodes (5G NR backhaul, LOS, Î±=2.5)
    for fog_id in fog_ids:
        edge_pos = topology.nodes[edge_id].position_km
        fog_pos = topology.nodes[fog_id].position_km
        dist_km = _euclidean_km(edge_pos, fog_pos)
        topology.add_link(NetworkLink(
            source_id=edge_id,
            dest_id=fog_id,
            distance_km=dist_km,
            path_loss_exponent=2.5,
        ))

    # 5e. Edge â†’ Cloud (fibre WAN)
    topology.add_link(NetworkLink(
        source_id=edge_id,
        dest_id=cloud_id,
        distance_km=50.0,
        path_loss_exponent=2.0,
    ))

    # 5f. Fog â†’ Cloud (fibre backhaul)
    for fog_id in fog_ids:
        fog_pos = topology.nodes[fog_id].position_km
        cloud_pos = topology.nodes[cloud_id].position_km
        dist_km = _euclidean_km(fog_pos, cloud_pos)
        topology.add_link(NetworkLink(
            source_id=fog_id,
            dest_id=cloud_id,
            distance_km=dist_km,
            path_loss_exponent=2.0,
        ))

    print(
        f"[Topology] Built: {n_wearables} wearables | 1 edge gateway | "
        f"{n_fog_nodes} fog nodes | 1 cloud -> "
        f"{len(topology.nodes)} nodes, {len(topology.links)} directed links."
    )
    return topology


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _euclidean_km(pos_a: Tuple[float, float], pos_b: Tuple[float, float]) -> float:
    """Euclidean distance in kilometres between two (x, y) km positions."""
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Convenience accessor â€” node ID helpers
# ---------------------------------------------------------------------------

def get_wearable_ids(topology: NetworkTopology) -> List[int]:
    return [nid for nid, n in topology.nodes.items() if n.node_type == 'wearable']


def get_edge_ids(topology: NetworkTopology) -> List[int]:
    return [nid for nid, n in topology.nodes.items() if n.node_type == 'edge']


def get_fog_ids(topology: NetworkTopology) -> List[int]:
    return [nid for nid, n in topology.nodes.items() if n.node_type == 'fog']


def get_cloud_id(topology: NetworkTopology) -> int:
    cloud_nodes = [nid for nid, n in topology.nodes.items() if n.node_type == 'cloud']
    if not cloud_nodes:
        raise RuntimeError("No cloud node found in topology.")
    return cloud_nodes[0]

