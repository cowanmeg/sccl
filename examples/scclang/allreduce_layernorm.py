# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import Collective, AllReduce

class FusedAllReduceAddLayernorm(Collective):

    def __init__(self, num_ranks, chunk_factor, inplace):
        Collective.__init__(self, num_ranks, chunk_factor, inplace)
        self.name = 'custom'

    def init_buffers(self):
        chunks_per_node = self.chunk_factor
        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = []
            output_buffer = [None] * chunks_per_node
            for c in range(chunks_per_node):
                # Chunks start at rank r index c, and ends on all ranks (-1) at index r
                input_buffer.append(Chunk(r, c, -1, c))
            # Input and output buffer are the same.
            if self.inplace:
                buffers = {Buffer.input : input_buffer, 
                           Buffer.output : input_buffer}
            else:
                buffers = {Buffer.input : input_buffer, 
                           Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers

    def check(self, prog):
        chunks_per_node = self.chunk_factor
        expected_chunks = []
        buf = Buffer.input if self.inplace else Buffer.output

        for c in range(chunks_per_node):
            chunk = ReduceChunk(-1, [])
            for r in range(self.num_ranks):
                chunk = chunk.reduce(-1, Chunk(r, c))
            expected_chunks.append(chunk)

        correct = True
        for r in range(self.num_ranks):
            output = prog.buffers[r][buf]
            for c in range(chunks_per_node):
                chunk = output[c]
                if chunk is None or chunk != expected_chunks[c]:
                    print(f'Rank {r} chunk {c} is incorrect should be ReduceChunk index {c} from all ranks, given {chunk}')
                    correct = False
        return correct

    def get_buffer_index(self, rank, buffer, index):
        if self.inplace and buffer == Buffer.output:
            return Buffer.input, index
        else:
            return buffer, index



def resadd_ar_layernorm_v1(instances):
    size = 8
    chunksperloop = 64 * instances
    topology = fully_connected(size)
    collective = FusedAllReduceAddLayernorm(size, chunksperloop, False)

    with SCCLProgram("allreduce_layernorm", topology, collective, 1, protocol="LL", 
        interleaved_replication=True, threadblock_policy=ThreadblockPolicy.manual, instr_fusion=True):

        # Each rank sends the nth chunk to the nth rank into scratch space
        instance_size = chunksperloop // size // instances
        half_instance_size = chunksperloop // size // instances // 2
        chunk_size = chunksperloop // size
        half_chunk_size = chunksperloop // size // 2

        for r1 in range(size):
            for r2 in range(size):
                for i in range(instances):
                    index = r2 * chunk_size + i * instance_size
                    c = chunk(r1, Buffer.input, index, size=instance_size)
                    if r1 != r2:
                        if r1 == 0:
                            print(f'{r1}->{r2} sending {index} {instance_size}')
                        c.send(r2, 'scratch', sendtb=r2*instances+i, recvtb=r1*instances+i, ch=i)
                    else:
                        c.compute('residual-add', Buffer.input, index, tb=r1*instances + i, ch=i)

        # Each rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for r in range(size):
            for y in range(size-1):
                for x in range(0, chunk_size):
                    for i in range(0, instances):
                        print(index)
                        index = y * chunk_size + x * instance_size + i
                        c = chunk(r, 'scratch', index)
                        c.reduce(r, Buffer.input, r*chunk_size + x *instance_size + i, sendtb=(index % 8) * instances + i, ch=i)


        # for i in range(instances):
        #     for r in range(size):
        #         for ci in range(0, 56):
        #             index = ci +
        #             c = chunk(r, 'scratch', index)
        #             c.reduce(r, Buffer.input, r*chunk_size + (ci % 8) * instance_size + i, sendtb=(index % 8) * instances + i, ch=i)

        # for r in range(size):
        #     # Layernorm on the fully reduced chunk
        #     # Layernorm output size is half the input size
        #     c = chunk(r, Buffer.input, r*chunk_size, size=chunk_size)
        #     c.compute('layernorm', Buffer.output, r*half_chunk_size, tb=r * instances) # dst buffer, index
    
        # # Each rank sends the fully reduced nth chunk to all other gpus
        
        # for r1 in range(size):
        #     for r2 in range(size):
        #         for i in range(instances):
        #             if r1 != r2:
        #                 index = r1 * half_chunk_size + i * half_instance_size
        #                 c = chunk(r1, Buffer.output, index, half_instance_size)
                        c.send(r2, Buffer.output, index, sendtb=r2*instances+i, recvtb=r1*instances+i, ch=i)
                
        XML()

def resadd_ar_layernorm_v2(instances):
    size = 8
    chunksperloop = 64
    topology = fully_connected(size)
    collective = FusedAllReduceAddLayernorm(size, chunksperloop, False)

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

def ar_resadd(instances):
    size = 8
    chunksperloop = 64
    topology = fully_connected(size)
    collective = FusedAllReduceAddLayernorm(size, chunksperloop, True)
    with SCCLProgram("resadd_allreduce", topology, collective, instances, protocol="LL", 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual):
        
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
            c.send(r, 'scratch', r*8, sendtb=0) # Copy AllReduce -> output buffer

            # Layernorm on the fully reduced chunk
            # Assumes layernorm is inplace and the input and output are same size. 
            c = c.compute('residual-add', Buffer.output, r*8, tb=0)
        
        # Each rank sends the fully reduced nth chunk to all other gpus
        instance_size = chunksperloop // size // instances 
        for r1 in range(size):
            for r2 in range(size):
                if r1 != r2:
                    index = r1 * 8
                    c = chunk(r1, Buffer.output, index, 8)
                    c.send(r2, Buffer.output, index, sendtb=r2, recvtb=r1)
                
        XML()

def resadd_ar(instances):
    size = 8
    chunksperloop = 64 * instances
    topology = fully_connected(size)
    # collective=AllReduce(size, chunksperloop, True)
    collective = FusedAllReduceAddLayernorm(size, chunksperloop, True)
    with SCCLProgram("resadd_allreduce", topology, collective, 1, protocol="LL", 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual):
        
        # Each rank sends the nth chunk to the nth rank into scratch space
        csize = 8 * instances
        isize = 8
        for r1 in range(size):
            for r2 in range(size):
                for i in range(instances):
                    index = r2 * csize + i * isize
                    c = chunk(r1, Buffer.input, index, size=8)
                    if r1 != r2:
                        c.send(r2, 'scratch', sendtb=r2*instances+i, recvtb=r1*instances+i, ch=i)
                    else:
                        c.compute('residual-add', Buffer.input, index, tb=r2*instances+i, ch=i)

        # Each rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for r in range(size):
            for cc in range(0, 56):
                for i in range(instances):
                    index = cc * instances + i
                    c = chunk(r, 'scratch', index)
                    dstindx =  r*csize + (cc % 8) * instances + i
                    tb = index % 16
                    c.reduce(r, Buffer.input, dstindx, sendtb=tb, ch=i)
        
        # Each rank sends the fully reduced nth chunk to all other gpus
        for i in range(instances):
            for r1 in range(size):
                for r2 in range(size):
                    if r1 != r2:
                        index = r1 * csize + i * isize
                        c = chunk(r1, Buffer.input, index, 8)
                        c.send(r2, Buffer.input, index, sendtb=r2*instances+i, recvtb=r1*instances+i, ch=i)
                
        XML()
        Check()

def allreduce_ring(size, instances, channels, protocol):
    topology = fully_connected(size)
    collective = AllReduce(size, size, True)
    with SCCLProgram(f"allreduce_ring_{channels}channelsperring", topology, collective, instances,
         protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        # Reduce ring
        for step in range(0, size-1):
            for index in range(0, size):
                rank = (index + step) % size
                c = chunk(rank, Buffer.input, index)
                next_rank = (index + step + 1) % size
                channel = index%channels
                c = c.reduce(next_rank, Buffer.input, index, ch=channel, recvtb=channel, sendtb=channel)
        # Propagate ring
        for step in range(-1, size-2):
            for index in range(0, size):
                rank = (index + step) % size
                c = chunk(rank, Buffer.input, index)
                next_rank = (index + step + 1) % size
                channel = index%channels
                c = c.send(next_rank, Buffer.input, index, ch=channel, recvtb=channel, sendtb=channel)
                # After we get the fully reduced chunk - handle the residual add
                if step == size-3:
                    c.compute('residual-add', Buffer.input, index, tb=channel, ch=channel)
               
        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('channels', type=int, help='number of channels per ring')
parser.add_argument('version', type=str, choices=['inplace-add', 'outofplace-add'], help='which version of fused allreduce/layernorm')
args = parser.parse_args()
assert args.instances >= 1 and args.instances <= 4
# if args.version == 'inplace-add':
#     resadd_ar(args.instances)
# else:
#     ar_resadd(args.instances)
allreduce_ring(8, args.instances, args.channels, 'Simple')