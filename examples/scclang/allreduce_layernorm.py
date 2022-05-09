# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce


def allreduce_allpairs(instances):
    size = 8
    chunksperloop = 64
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, False)
    with SCCLProgram("allreduce_layernorm", topology, collective, instances, protocol="LL", 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual):
        
        # Each rank sends the nth chunk to the nth rank into scratch space
        for r1 in range(size):
            for r2 in range(size):
                index = r2 * 8
                c = chunk(r1, Buffer.input, index, size=8)
                if r1 != r2:
                    c.send(r2, 'scratch', sendtb=r2, recvtb=r1, ch=0)
                else:
                    c.compute('residual-add', tb=r2)

        # Each rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for r in range(size):
            for index in range(0, 56):
                c = chunk(r, 'scratch', index)
                c.reduce(r, Buffer.input, r*8 + (index % 8), sendtb=(index % 8), ch=0)
            # Layernorm on the fully reduced chunk
            # Assumes layernorm is inplace and the input and output are same size. 
            c = chunk(r, Buffer.input, r*8, size=8)
            c.compute('layernorm', tb=0) # dst buffer, index
        
        # Each rank sends the fully reduced nth chunk to all other gpus
        # TODO: Allgather is half the size now. Figure out where to add this to language.
        for r1 in range(size):
            for r2 in range(size):
                index = r1 * 8
                c = chunk(r1, Buffer.input, index, 4)
                c.send(r2, Buffer.output, r1*4, sendtb=r2, recvtb=r1)
                
        XML()
        # Check() # TODO: Checks for computation

parser = argparse.ArgumentParser()
parser.add_argument('instances', type=int, help='number of instances')
# parser.add_argument('threadblocks', type=int, default=0, help='number of threadblocks per instance')

args = parser.parse_args()

allreduce_allpairs(args.instances)