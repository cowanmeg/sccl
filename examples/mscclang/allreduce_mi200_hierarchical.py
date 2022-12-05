# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

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
    [0,8,9,13,12,14,15,11,10,2,3,7,6,4,5,1],
    [0,1,5,4,6,7,3,2,10,11,15,14,12,13,9,8],
    [0,4,5,6,7,3,2,9,8,12,13,14,15,11,10,1],
    [0,1,10,11,15,14,13,12,8,9,2,3,7,6,5,4],
]

# r = ring index
# g = gpu index
def ring2rank(r, n, g):
    return rings[r][g] + n * num_local_gpus

def rank(n, g):
    return n * num_local_gpus + g
        
def allreduce_ring(num_nodes, instances, num_rings, protocol):
    size = num_nodes * num_local_gpus
    num_chunks = num_rings* num_local_gpus
    topology = fully_connected(size)
    collective = AllReduce(size, num_chunks, True)

    with MSCCLProgram("allreduce_ring_mi200", topology, collective, instances,
        protocol=protocol, threadblock_policy=ThreadblockPolicy.manual, interleaved_replication=False, instr_fusion=True):
        # Intra-node reduce scatter - each gpu should have 12 chunks that are contiguous
        for n in range(num_nodes):
            for g in range(num_local_gpus):
                for r in range(num_rings):
                    index = (rings[r].index(g) + 1) % num_local_gpus
                    offset = r + g * num_rings
                    c = chunk(ring2rank(r, n, index), Buffer.input, offset)
                    for step in range(1, num_local_gpus):
                        c = chunk(ring2rank(r, n, (index+step)%num_local_gpus), Buffer.input, offset).reduce(c, sendtb=r, recvtb=r, ch=r)


        # Inter-node allreduce (reduce scatter + allgather)
        intranode_rings = num_rings // num_nodes
        for g in range(num_local_gpus):
            for i in range(0, num_rings):
                offset = g * num_rings + i
                n = i % num_nodes
                c = chunk(rank(n, g), Buffer.input, offset)
                x = (i//2) % intranode_rings
                for step in range(1, num_nodes):
                    c = chunk(rank((n+step)%num_nodes, g), Buffer.input, offset).reduce(c, sendtb=num_rings*2+x, recvtb=num_rings*2+x, ch=x)
                for step in range(0, num_nodes-1):
                    c = c.copy(rank((n+step)%num_nodes, g), Buffer.input, offset, sendtb=num_rings*2+x, recvtb=num_rings*2+x, ch=x)        

        # Intra-node allgather
        for n in range(num_nodes):
            for g in range(num_local_gpus):
                for r in range(num_rings):
                    index = rings[r].index(g)
                    offset = r + g * num_rings
                    c = chunk(ring2rank(r, n, index), Buffer.input, offset)
                    for step in range(1, num_local_gpus):
                        c = c.copy(ring2rank(r, n, (index+step)%num_local_gpus), Buffer.input, offset, sendtb=num_rings+r, recvtb=num_rings+r, ch=num_rings+r)          
        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help ='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('num_rings', type=int, default=12, choices=range(1, 13), help='Number of rings [1-12]')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

assert args.num_nodes > 1, "Multi-node allreduce. num_nodes > 1"
assert 12 % args.num_nodes == 0, "Algorithm requires the number of nodes to divide 12"
allreduce_ring(args.num_nodes, args.instances, args.num_rings, args.protocol)
