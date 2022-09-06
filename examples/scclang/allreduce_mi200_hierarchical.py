# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

# Ring all reduce for AMD MI200

num_local_gpus = 16

rings = [
    [0,4,5,7,6,2,3,11,10,14,15,13,12,8,9,1],
    [0,4,5,6,7,3,2,9,8,12,13,14,15,11,10,1],
    [0,1,9,8,12,13,15,14,10,11,3,2,6,7,5,4],
    [0,1,10,11,15,14,13,12,8,9,2,3,7,6,5,4],
    [0,4,5,7,6,2,3,11,10,14,15,13,12,8,9,1],
    [0,1,9,8,12,13,15,14,10,11,3,2,6,7,5,4],
    [0,8,9,13,12,14,15,11,10,2,3,7,6,4,5,1],
    [0,1,5,4,6,7,3,2,10,11,15,14,12,13,9,8],
]

# r = ring index
# g = gpu index
def rank(r, n, g):
    return rings[r][g] + n * num_local_gpus

        
def allreduce_ring(num_nodes, instances, protocol, schedule):
    size = num_nodes * num_local_gpus
    chunks_per_node = len(rings)
    num_chunks = chunks_per_node * num_nodes
    topology = fully_connected(size)
    collective = AllReduce(size, num_chunks, True)
    offset=1

    with SCCLProgram("allreduce_ring_mi200", topology, collective, instances,
        protocol=protocol, threadblock_policy=ThreadblockPolicy.auto, interleaved_replication=False, instr_fusion=True):
        # Intra-node reduce scatter
        for n in range(num_nodes):
            for index in range(num_chunks):        
                x = index % chunks_per_node
                c = chunk(rank(x, n, (x+offset)%num_local_gpus), Buffer.input, index)
                for step in range(1, num_local_gpus):
                    c = chunk(rank(x, n, (x+offset+step)%num_local_gpus), Buffer.input, index).reduce(c)

        # Inter-node allreduce (reduce scatter + allgather)
        for index in range(num_chunks):
            g = index % chunks_per_node
            n = index // chunks_per_node
            c = chunk(rank(g, n, g), Buffer.input, index)
            for step in range(1, num_nodes):
                c = chunk(rank(g, (n+step)%num_nodes, g), Buffer.input, index).reduce(c)
            for step in range(0, num_nodes-1):
                c = c.copy(rank(g, (n+step)%num_nodes, g), Buffer.input, index) 

        # Intra-node allgather
        for n in range(num_nodes):
            for index in range(num_chunks):        
                g = index % chunks_per_node
                c = chunk(rank(g, n, g), Buffer.input, index)
                for step in range(1, num_local_gpus):
                    c = c.copy(rank(g, n, (g+step)%num_local_gpus), Buffer.input, index)

        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help ='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: LL128')
parser.add_argument('--schedule', type=str, default='auto', choices=['auto', 'manual'], help ='Scheduling policy. Default: auto')
args = parser.parse_args()

assert args.num_nodes > 1, "Multi-node allreduce. num_nodes > 1"
allreduce_ring(args.num_nodes, args.instances, args.protocol, args.schedule)
