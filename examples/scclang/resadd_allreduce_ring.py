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

def allreduce_ring(size, instances, channels, protocol):
    topology = fully_connected(size)
    collective = FusedAllReduceAddLayernorm(size, size, True)
    with SCCLProgram(f"allreduce_ring_resadd", topology, collective, instances,
         protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        # Reduce ring
        for step in range(0, size-1):
            for index in range(0, size):
                rank = (index + step) % size
                c = chunk(rank, Buffer.input, index)
                channel = index%channels
                if step == 0:
                    c = c.compute('residual-add', Buffer.input, index, tb=channel, ch=channel)
                next_rank = (index + step + 1) % size
                c = c.reduce(next_rank, Buffer.input, index, ch=channel, recvtb=channel, sendtb=channel)
        # Propagate ring
        for step in range(-1, size-2):
            for index in range(0, size):
                rank = (index + step) % size
                c = chunk(rank, Buffer.input, index)
                next_rank = (index + step + 1) % size
                channel = index%channels
                c = c.send(next_rank, Buffer.input, index, ch=channel, recvtb=channel, sendtb=channel)
               
        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('channels', type=int, help='number of channels per ring')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: LL128')

args = parser.parse_args()

allreduce_ring(8, args.instances, args.channels, args.protocol)