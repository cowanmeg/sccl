# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.language import *
from lxml import etree as ET
from dataclasses import dataclass, field
import sccl.collectives as scclcoll
from sccl.strategies import *
from sccl.path_encoding import *
# from sccl.language.chunk_dag import *

import sys
from enum import Enum
from collections import defaultdict


class SCCLSketch(SCCLProgram):

    def __init__(self, name, topo, collective, instances, protocol='Simple'):
        self.name = name
        self.topo = topo
        self.collective = collective       
        self.instances = instances
        self.protocol = protocol
        assert protocol == 'Simple' or protocol == 'LL' or protocol == 'LL128', \
            f'Given protocol: {protocol}. Must be either Simple, LL, LL128'

        # Initialize the input buffers
        num_ranks = topo.num_nodes()
        self.ranks = []
        for r in range(num_ranks):
            self.ranks.append(SymbolicProcess(self, r))
        self.buffers = collective.init_buffers()
        self.chunk_dag = ChunkDAG()
        self.syn_collective = scclcoll.allgather(topo.num_nodes())
        for r in range(num_ranks):
            buf = self.buffers[r][Buffer.input]
            for chunk in buf:
                ref = self.get_ref(r, Buffer.input, chunk.origin_index, 1)
                self.chunk_dag.init_chunk(chunk, ref)
        
    def get_ref(self, rank, buffer, index, size):
        # TODO This will only work for concrete references??
        return SymbolicRef(buffer, index, size, self, [rank], self.buffers[rank][buffer][index])

    def send(self, sender, dst, buffer, index, size):
        assert size == 1, "Sketches only support 1 chunk sends."
        chunk = sender.chunk
        ref = SymbolicRef(buffer, index, size, self, dst, chunk)
        self.chunk_dag.add_send(chunk, sender, ref, -1, -1, 0)
        # Concrete sender and destination - update the buffer
        if len(sender.rank) == 1 and len(dst) == 1:
            self.buffers[dst[0]][buffer][index] = chunk
        # Symbolic destination TODO - keep track of where it has been...
        # what to do here...
        return ref

    # Lower program and holes to constraints for sccl to target
    def synthesize(self):
        print("Starting to synthesize!")
        logging = True
        encoding = PathEncoding(self.topo, self.syn_collective)
        self.chunk_dag.lower_constraints(encoding)
        initial_steps = self.chunk_dag.max_steps
        print(f"Setting initial steps to {initial_steps} based off sketch")
        result = solve_least_steps(self.topo, self.syn_collective, initial_steps=initial_steps, logging=logging, encoding=encoding)
        print(result)
        # TODO: Is there a way I can fill in the source file?

@dataclass
class SymbolicProcess():
    sketch: SCCLSketch
    rank: int

    def input(self, index, size=1):
        return self.sketch.get_ref(self.rank, Buffer.input, index, size)


@dataclass
class SymbolicRef(ChunkRef):
    sketch: SCCLSketch
    rank: list
    chunk: Chunk 

    # Send current takes a dst or dst hole
    # concrete buffer, and index
    def send(self, dst, buffer=None, index=-1, sendtb=-1, recvtb=-1, ch=0):
        # Change dst into a list if a scalar is given 
        if type(dst) is not list:
            dst = [dst]
        else: # Just in case there are duplicate destinations in this path
            dst_set = set(dst)
            dst = list(dst_set)
        return self.sketch.send(self, dst, buffer, index, self.size)

    def reduce(self, dst, buffer, index=-1, sendtb=-1, recvtb=-1, ch=0):
        print("Can't support reduce in a sketch yet")
        sys.exit(1)

    def group(self, other):
        print("Can't support group in a sketch yet")
        sys.exit(1)

    def split(self, num):
        print("Can't support split in a sketch yet")
        sys.exit(1)


class ChunkInstruction(Enum):
    start = 'start'
    reduce = 'reduce'
    send = 'send'

    def __str__(self):
        return self.value

@dataclass
class ChunkOp:
    inst: ChunkInstruction
    ref: SymbolicRef
    sendtb: int # For lowering to RankInstructions
    recvtb: int #  For lowering to RankInstructions
    ch: int # For lowering to RankInstructions
    steps = 0 
    prev: list = field(default_factory=list) # Previous ChunkOps
    next: list = field(default_factory=list) # Next ChunkOps
    visited = False

    def __repr__(self):
        return f'ChunkOp({self.inst} {self.ref.rank} {self.ref.buffer} {self.ref.index} {self.next})'

    def __lt__(self, other):
        return self.steps < other.steps

def _same_location(op, ref):
    return op.ref.rank == ref.rank and op.ref.buffer == ref.buffer and op.ref.index == ref.index

class ChunkDAG:
    def __init__(self):
        self.chunk_paths = {} # chunk -> ChunkOp. Stores the entry point to where every chunk is created
        self.max_steps = -1

    # Initialize the ChunkDAG with starting chunks
    def init_chunk(self, chunk, ref):
        op = ChunkOp(ChunkInstruction.start, ref, -1, -1, -1)
        self.chunk_paths[chunk] = op

    def _find_prev_op_for_chunk(self, chunk, ref):
        prev_op = None
        frontier = [self.chunk_paths[chunk]]
        while len(frontier) > 0:
            current_op = frontier[0]
            if _same_location(current_op, ref):
                prev_op = current_op
            frontier = frontier[1:] + current_op.next
        return prev_op

    def add_send(self, chunk, src, dst, sendtb, recvtb, ch):
        op = ChunkOp(ChunkInstruction.send, dst, sendtb, recvtb, ch)

        # Find the previous operation for this chunk
        prev_op = self._find_prev_op_for_chunk(chunk, src)
        prev_op.next.append(op)
        op.prev.append(prev_op)

    def add_reduce(self, chunk1, chunk2, reduce_chunk, src, dst, sendtb, recvtb, ch):
        op = ChunkOp(ChunkInstruction.reduce, dst, sendtb, recvtb, ch)

        # Find the previous operations that reduce builds off
        prev_op_src = self._find_prev_op_for_chunk(chunk1, src)
        prev_op_dst = self._find_prev_op_for_chunk(chunk2, dst)
        prev_op_src.next.append(op)
        prev_op_dst.next.append(op)
        op.prev = op.prev + [prev_op_src, prev_op_dst]

        # Reduce operations create new chunks, so keep a pointer to a new chunk
        self.chunk_paths[reduce_chunk] = op

    def _complete_metadata(self):
        frontier = []
        for chunk, op in self.chunk_paths.items():
            if len(op.prev) == 0: 
                frontier.append(op)

        while len(frontier) > 0:
            current_op = frontier[0]
            for next_op in current_op.next:
                next_op.steps = max(next_op.steps, current_op.steps + 1)
                self.max_steps = max(self.max_steps, next_op.steps)
            frontier = frontier[1:] + current_op.next

    # Adds constraints from the sketch on top of the base encoding
    # Walk the ChunkDAG adding in information
    def lower_constraints(self, encoding):
        self._complete_metadata()
        s = encoding.solver
        s.push()

        for chunk, op in self.chunk_paths.items():
            # TODO: SCCLang chunk -> SCCL chunk id
            chunk_id = chunk.origin_rank
            src = [chunk.origin_rank]
            frontier = op.next

            while len(frontier) > 0:
                op = frontier[0]
                dst = op.ref.rank
                step = op.steps
                # Concrete source
                if len(src) == 1:
                    # Concrete source -> concrete destination
                    if len(dst) == 1:
                        s.add(send(chunk_id, src, dst[0]) == True)
                        s.add(start(chunk_id, dst[0]) == step)
                    # Concrete source -> symbolic destination
                    else:
                        opts = [And(send(chunk_id, src, d), start(chunk_id, d) == step) for d in dst]
                        s.add(Xor(*opts))
                # Symbolic source -> symbolic/concrete destination
                else:
                    for src_opt in src:
                        implication_opts = [And(send(chunk_id, src_opt, d), start(chunk_id, d) == step) for d in dst]
                        s.add(Implies(start(chunk_id, src_opt) == step-1, Xor(*implication_opts)))
                src = dst
                frontier = frontier[1:] + op.next

