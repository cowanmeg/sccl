# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

# Ring all reduce for AMD MI200

num_local_gpus = 16

# 12 rings - first 8 all end 
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
def rank(r, g):
    return rings[r][g]

        
def allreduce_ring(instances, protocol):
    num_rings = len(rings)
    num_chunks = num_rings * num_local_gpus
    topology = fully_connected(num_local_gpus)
    collective = AllReduce(num_local_gpus, num_chunks, True)
    offset=1
    with SCCLProgram("allreduce_ring_mi200", topology, collective, instances,
        protocol=protocol, threadblock_policy=ThreadblockPolicy.manual, instr_fusion=True, interleaved_replication=False):
        for g in range(num_local_gpus):
            for r in range(num_rings):
                index = rings[r].index(g)
                offset = g * num_rings + r
                c = chunk(rank(r, index), Buffer.input, offset)
                for step in range(1, num_local_gpus):
                    c = chunk(rank(r, (index+step)%num_local_gpus), Buffer.input, offset).reduce(c, sendtb=r, recvtb=r, ch=r)
                for step in range(0, num_local_gpus-1):
                    c = c.copy(rank(r, (index+step)%num_local_gpus), Buffer.input, offset, sendtb=r, recvtb=r, ch=r)       

        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: LL128')
args = parser.parse_args()

allreduce_ring(args.instances, args.protocol)
