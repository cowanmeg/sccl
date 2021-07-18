import argparse
import datetime 
import time

import sccl.collectives as collectives
import sccl.strategies as strategies
import sccl.topologies.nvidia as nvtp
import sccl.topologies.generic as gtp
import sccl.topologies.distributed as dtp
import sccl.heuristic as heuristics
from sccl.instance import *

def _make_buddies(buddies_2d, num_local):
    buddies = []
    for src, dst in buddies_2d:
        src_copy, src_rank = src
        dst_copy, dst_rank = dst
        src_id = src_copy * num_local + src_rank
        dst_id = dst_copy * num_local + dst_rank
        buddies.append((src_id, dst_id))
    return buddies

# Every node is fully connected
# Each local rank has one symmetric link to a rank in another node
# Only works for the configuration num_local + 1 == num_copies
def fully_connected_symmetric(num_local, num_copies):
    buddies_2d = []
    connected = set([])
    for n in range(num_copies):
        for r in range(n, num_local):
            node_dst = r+1 
            rank_dst = n
            buddies_2d.append(((n, r), (node_dst, rank_dst)))
            buddies_2d.append(((node_dst, rank_dst), (n,r)))
    return buddies_2d

def restricted_topology(num_local, collective, use_heuristic):
    num_copies = num_local + 1
    local_topo = gtp.hub_and_spoke(num_local)

    buddies_2d = fully_connected_symmetric(num_local, num_copies)
    buddies = _make_buddies(buddies_2d, num_local)
    # Make the remote BW across the nodes higher since we are only exposing one of the links out of num_local links up the bw
    topology = dtp.distributed_node_local(local_topo, num_copies, buddies, num_local)
    print(f'Buddy links: {buddies}')
    print(f'Links: {topology.links}')
    print(f'Switches: {topology.switches}')

    if collective == 'alltoall':
        c = collectives.alltoall(topology.num_nodes(), topology.num_nodes())
    else:
        c = collectives.allgather(topology.num_nodes())

    base_instance = Instance(None).set(extra_memory=num_copies)
    if use_heuristic:
        h = heuristics.Heuristic(buddies)
    else:
        h = None

    # initial_steps = 2
    # result = strategies.solve_least_steps(topology, c, initial_steps, base_instance, logging=True, heuristic=h)
    # print(result)

    algorithms = []
    start = time.time()
    for alg in strategies.solve_all_latency_bandwidth_tradeoffs(topology, c, base_instance=base_instance, 
                logging=True, heuristic=h):
        algorithms.append(alg)
        stop = time.time()
        print(f'Total elapsed time {stop-start}s')
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
    
    ts = datetime.datetime.now()
    print("Start time:", ts.strftime("%a %m-%d-%Y %H:%M:%S"))
    
    if args.topology_restriction:
        restricted_topology(num_local, collective, args.heuristic)
    else:
        baseline(num_local, collective)


