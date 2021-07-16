# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .topology import Topology

def _copy_links(remote_bw, num_local, num_dist, local_links):
    return [[remote_bw if src // num_local != dst // num_local else local_links[dst % num_local][src % num_local]
        for src in range(num_dist)] for dst in range(num_dist)]

def _copy_switches(num_local, num_copies, local_switches):
    switches = []
    for srcs, dsts, bw, name in local_switches:
        for i in range(num_copies):
            dist_srcs = [src + i * num_local for src in srcs]
            dist_dsts = [dst + i * num_local for dst in dsts]
            switches.append((dist_srcs, dist_dsts, bw, f'copy_{i}_{name}_local'))
    return switches

def distributed_fully_connected(local_topology, num_copies, remote_bw):
    num_local = local_topology.num_nodes()
    num_dist = num_local * num_copies

    links = _copy_links(remote_bw, num_local, num_dist, local_topology.links)
    switches = _copy_switches(num_local, num_copies, local_topology.switches)

    return Topology(f'DistributedFullyConnected(local={local_topology.name},copies={num_copies},bw={remote_bw})', links, switches)

def distributed_hub_and_spoke(local_topology, num_copies, remote_bw):
    num_local = local_topology.num_nodes()
    num_dist = num_local * num_copies

    links = _copy_links(remote_bw, num_local, num_dist, local_topology.links)
    switches = _copy_switches(num_local, num_copies, local_topology.switches)

    for i in range(num_copies):
        local_ranks = [j + i * num_local for j in range(num_local)]
        remote_ranks = [k for k in range(num_dist) if k // num_local != i]
        switches.append((local_ranks, remote_ranks, remote_bw, f'copy_{i}_out_remote'))
        switches.append((remote_ranks, local_ranks, remote_bw, f'copy_{i}_in_remote'))
    
    return Topology(f'DistributedHubAndSpoke(local={local_topology.name},copies={num_copies},bw={remote_bw})', links, switches)

# Distributed template with only local connections
# Specify the node topology separately
def distributed_node_local(local_topology, num_copies, node_topology, remote_bw):
    num_local = local_topology.num_nodes()
    num_dist = num_local * num_copies

    local_links = local_topology.links
    links = []

    # Local links
    for n in range(num_copies):
        for r in range(num_local):
            rank_links = [0 for k in range(0, n * num_local)]
            rank_links += local_links[r].copy()
            rank_links += [0 for k in range(0, (num_copies-n-1) * num_local)]
            links.append(rank_links)

    # Remote links 
    for src, dst in node_topology:
        links[src][dst] = 1

    # Switches - TODO: Assumes every rank is connected - we don't check
    switches = _copy_switches(num_local, num_copies, local_topology.switches)
    for i in range(num_copies):
        local_ranks = [j + i * num_local for j in range(num_local)]
        remote_ranks = [k for k in range(num_dist) if k // num_local != i]
        switches.append((local_ranks, remote_ranks, remote_bw, f'copy_{i}_out_remote'))
        switches.append((remote_ranks, local_ranks, remote_bw, f'copy_{i}_in_remote'))

    return Topology(f'DistributedNodeLocal(local={local_topology.name},copies={num_copies},bw={remote_bw})', links, switches)

