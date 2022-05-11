# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce


def resadd_ar_layernorm_v1(instances):
    size = 8
    chunksperloop = 64
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, False)

    
    with SCCLProgram("allreduce_layernorm", topology, collective, 1, protocol="LL", 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual, instr_fusion=True):

        instances = 1
        # Each rank sends the nth chunk to the nth rank into scratch space
        for r1 in range(size):
            for r2 in range(size):
                index = r2 * 8 
                c = chunk(r1, Buffer.input, index, size=8)
                if r1 != r2:
                    c.send(r2, 'scratch', sendtb=r2, recvtb=r1, ch=0)
                else:
                    c.compute('residual-add', Buffer.input, index, tb=0)

        # Each rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for r in range(size):
            for index in range(0, 56):
                c = chunk(r, 'scratch', index)
                c.reduce(r, Buffer.input, r*8 + (index % 8), sendtb=(index % 8), ch=0)

            # Layernorm on the fully reduced chunk
            # Assumes layernorm is inplace and the input and output are same size. 
            # Layernorm output size is half the input size
            c = chunk(r, Buffer.input, r*8, size=8)
            c.compute('layernorm', Buffer.output, r*4, tb=0) # dst buffer, index
        
        # Each rank sends the fully reduced nth chunk to all other gpus
        # instance_size = chunksperloop // size // instances // 2
        for r1 in range(size):
            for r2 in range(size):
                if r1 != r2:
                    index = r1 * 4
                    c = chunk(r1, Buffer.output, index, 4)
                    c.send(r2, Buffer.output, index, sendtb=r2, recvtb=r1)
                
        XML()

def resadd_ar_layernorm_v2(instances):
    size = 8
    chunksperloop = 64
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, False)

    
    with SCCLProgram("allreduce_layernorm", topology, collective, 1, protocol="LL", 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual, instr_fusion=True):

        instances = 1
        # Each rank sends the nth chunk to the nth rank into scratch space
        for r1 in range(size):
            for r2 in range(size):
                index = r2 * 8 
                c = chunk(r1, Buffer.input, index, size=8)
                if r1 != r2:
                    c.send(r2, 'scratch', sendtb=r2, recvtb=r1, ch=0)

        # Each rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for r in range(size):
            for index in range(0, 56):
                c = chunk(r, 'scratch', index)
                c.reduce(r, Buffer.input, r*8 + (index % 8), sendtb=(index % 8), ch=0)
            c = chunk(r, Buffer.input, r*8, size=8)
            c.send(r, 'scratch', 0, sendtb=0) # AllReduce output: Temporarily being stored in the scratch

            # Layernorm on the fully reduced chunk
            # Assumes layernorm is inplace and the input and output are same size. 
            c = c.compute('residual-add', Buffer.input, r*8, tb=0)
            # Layernorm output size is half the input size
            c.compute('layernorm', Buffer.output, r*4, tb=0) # dst buffer, index
        
        # Each rank sends the fully reduced nth chunk to all other gpus
        # instance_size = chunksperloop // size // instances // 2
        for r1 in range(size):
            for r2 in range(size):
                if r1 != r2:
                    index = r1 * 4
                    c = chunk(r1, Buffer.output, index, 4)
                    c.send(r2, Buffer.output, index, sendtb=r2, recvtb=r1)
                
        XML()

def resadd_ar(instances):
    size = 8
    chunksperloop = 64
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, False)
    with SCCLProgram("resadd_allreduce", topology, collective, instances, protocol="LL", 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual):
        
        # Each rank sends the nth chunk to the nth rank into scratch space
        for r1 in range(size):
            for r2 in range(size):
                index = r2 * 8
                c = chunk(r1, Buffer.input, index, size=8)
                if r1 != r2:
                    c.send(r2, 'scratch', sendtb=r2, recvtb=r1, ch=0)
                else:
                    c.compute('residual-add', Buffer.input, index, tb=r2)

        # Each rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for r in range(size):
            for index in range(0, 56):
                c = chunk(r, 'scratch', index)
                c.reduce(r, Buffer.input, r*8 + (index % 8), sendtb=(index % 8), ch=0)
        
        # Each rank sends the fully reduced nth chunk to all other gpus
        instance_size = chunksperloop // size // instances 
        for r1 in range(size):
            for r2 in range(size):
                index = r1 * 8
                c = chunk(r1, Buffer.input, index, 8)
                c.send(r2, Buffer.output, index, sendtb=r2, recvtb=r1)
                
        XML()


parser = argparse.ArgumentParser()
# parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('version', type=str, choices=['no-layernorm', 'save-ar', 'save-ar-resadd'], help='which version of fused allreduce/layernorm')
args = parser.parse_args()
# assert args.instances >= 1 and args.instances <= 4
if args.version == 'save-ar':
    resadd_ar_layernorm_v2(1)
elif args.version == 'save-ar-resadd':
    resadd_ar_layernorm_v1(1)
else:
    resadd_ar(1)