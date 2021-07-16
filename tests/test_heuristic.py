import argparse

import sccl.collectives as collectives
import sccl.strategies as strategies
import sccl.topologies.nvidia as nvtp
import sccl.topologies.generic as gtp
import sccl.topologies.distributed as dtp
import sccl.heuristic as heuristics
from sccl.instance import *

# TODO: Make certain buddies form a fully connected topology between nodes.
def _check_buddies(buddies_2d, num_local, num_remote):
    # Check there are n(n-1)/2 buddies
    if (len(buddies_2d) < num_remote * (num_remote-1)/2):
        print("Not enough connections between nodes for fully connected")
    connected = [False] * num_remote

    return True
    

def _make_buddies(buddies_2d, num_local):
    buddies = []
    for src, dst in buddies_2d:
        src_copy, src_rank = src
        dst_copy, dst_rank = dst
        src_id = src_copy * num_local + src_rank
        dst_id = dst_copy * num_local + dst_rank
        buddies += [(src_id, dst_id), (dst_id, src_id)]
    return buddies


def restricted_topology(num_local, collective, use_heuristic):
    num_copies = num_local + 1
    local_topo = gtp.hub_and_spoke(num_local)
    # For now buddies are symmetric and hardcoded
    if num_local == 3:
        buddies_2d = [
            ((0,0), (1,0)),
            ((0,1), (2,1)),
            ((0,2), (3,2)),
            ((1,1), (3,1)),
            ((1,2), (2,2)),
            ((2,0), (3,0))
        ]
    elif num_local == 2:
        buddies_2d = [
            ((0,0), (1,0)),
            ((0,1), (2,1)),
            ((1,1), (2,0))
        ]
    buddies = _make_buddies(buddies_2d, num_local)
    # Make the remote BW across the nodes higher since we are only exposing one of the links out of num_local links up the bw
    topology = dtp.distributed_node_local(local_topo, num_copies, buddies, num_local)
    print(f'Buddy links: {buddies}')
    print(f'Links: {topology.links}')
    print(f'Switches: {topology.switches}')

    if collective == 'alltoall':
        collective = collectives.alltoall(topology.num_nodes())
    else:
        collective = collectives.allgather(topology.num_nodes())

    base_instance = Instance(None).set(extra_memory=num_copies*num_local)
    if use_heuristic:
        h = heuristics.Heuristic(buddies)
    else:
        h = None

    # initial_steps = 2
    # result = strategies.solve_least_steps(topology, collective, initial_steps, base_instance, logging=True, heuristic=h)
    # print(result)

    algorithms = []
    for alg in strategies.solve_all_latency_bandwidth_tradeoffs(topology, collective, base_instance=base_instance, 
                logging=True, heuristic=h):
        algorithms.append(alg)
        print(alg)
    

def baseline(num_local, collective):
    num_copies = num_local + 1
    local_topo = gtp.hub_and_spoke(num_local)
    topology = dtp.distributed_hub_and_spoke(local_topo, num_copies, 1)
    print(f'Links: {topology.links}')
    print(f'Switches: {topology.switches}')

    if collective == 'alltoall':
        collective = collectives.alltoall(topology.num_nodes())
    else:
        collective = collectives.allgather(topology.num_nodes())

    # base_instance = Instance(None)
    # initial_steps = 1
    # result = strategies.solve_least_steps(topology, collective, initial_steps, base_instance, logging=True)

    algorithms = []
    for alg in strategies.solve_all_latency_bandwidth_tradeoffs(topology, collective, logging=True):
        algorithms.append(alg)
        print(alg)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-local', dest='num_local', type=int)
    parser.add_argument('--no-topology-restriction', dest='topology_restriction', action='store_false')
    parser.add_argument('--no-heuristic', dest='heuristic', action='store_true')
    parser.add_argument('--collective', dest='collective', type=str, choices=['alltoall', 'allgather'])
    parser.set_defaults(topology_restriction=True, heuristic=True, collective='alltoall')
    args = parser.parse_args()
    num_local = args.num_local
    collective = args.collective

    print(f'Nodes:{num_local}, Copies:{num_local+1}, Restricted topology:{args.topology_restriction}, Heuristic:{args.heuristic}, Collective:{collective}')
    if args.topology_restriction:
        restricted_topology(num_local, collective, args.heuristic)
    else:
        baseline(num_local, collective)


