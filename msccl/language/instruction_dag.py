# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
import heapq
import time
from typing import DefaultDict

from msccl.language.ir import *
from msccl.language.passes import *

def remove_op(op):
    for p in op.prev:
        p.next.remove(op)
        p.next += op.next

    for n in op.next:
        n.prev.remove(op)
        n.prev =  op.prev.union(n.prev)

    op.inst = Instruction.delete

def same_tb(op1, op2):
    return op1.tb == op2.tb and op1.channel == op2.channel

# TODO: Temporary until runtime supports strided copy. Only fuse instructions with count == 1
# Fusion with count > 1 can result in deadlocks and is disabled. 
def same_count(op1, op2):
    return op1.cnt() == 1 and op1.cnt() == op2.cnt()

# A chain of fused instructions
# send -> rrs -> rrcs -> rcs -> r 
class Chain:
    def __init__(self, send_op):
        assert send_op.inst == Instruction.send
        self.ops = [send_op, send_op.recv_match]
        self.connections = defaultdict(list)
        self.connections[send_op.rank].append((-1, send_op.recv_match.rank))

    # Try to fuse in another send
    # Make certain we don't run into a situation where we have a->b->c and want to add x->b->c
    # since this can't map to our runtime constraints since it require two threadblocks on b send to c.
    # Return whether it was successful or not
    def can_add(self, send_op):
        # send(start)---recv(mid),send(mid) ---- recv(end)
        start = self.ops[-2].rank
        mid = self.ops[-1].rank
        assert mid == send_op.rank, f"Cannot fuse a ops that starts at {mid} with instruction that starts at {send_op.rank}"
        end = send_op.recv_match.rank 
        mid_connections = self.connections[mid]
        for c_start, c_end in mid_connections: 
            if c_start == start and c_end != end and c_end != -1 or c_end == end and c_start != start and c_start != -1:
                return False
        return True

    def add(self, op):
        self.ops[-1] = op
        self.ops.append(op.recv_match)
        self.connections[op.rank].append((op.send_match.rank, op.recv_match.rank))

    def length(self):
        return len(self.ops)

    def connection_set(self):
        connections = set()
        for op in self.ops:
            if op.is_send():
                connections.add((op.rank, op.recv_match.rank))
        return connections


class InstructionDAG:
    def __init__(self, num_ranks, buffers, protocol):
        self.num_ranks = num_ranks
        self.buffers = buffers
        self.protocol = protocol
        # State for the instruction DAG
        self.operations = {} # slot -> operations
        self.last_writer = {} # slot -> last writing op
        self.last_readers = defaultdict(list) # slot -> list of last reading ops
        # State for the MSCCL-IR
        self.tbs = [] 
        for _ in range(num_ranks):
            self.tbs.append({}) 
        self.tb_mapping = {}
        self.num_channels = [1] * num_ranks
        self.sends = [] # list of all sends
        self.chains = [] # list of all fused instruction chains
        self.num_instrs = 0
        self.ordered_instrs = None

    # InstructionDAG helper - identifies the dependencies for a write-type operation (recv, copy, rrc, reduce)
    def _write(self, rank, buffer, index, size, op, read=False):
        prev_ops = set()
        for i in range(index, index+size):
            slot = (rank, buffer, i)
            if read:
                assert slot in self.last_writer, f"Destination slot has never been written before a reduce {op}"

            # First write to this slot
            if slot not in self.operations:
                self.operations[slot] = op

            # If there are active readers - these are the previous operations
            # Else the previous operation is the last write (if there is one)
            readers = self.last_readers[slot]
            if len(readers) > 0:
                prev_ops.update(readers)
            elif slot in self.last_writer:
                prev_ops.add(self.last_writer[slot])
  
            # Set the last_writer to this op, and clear all readers
            self.last_writer[slot] = op
            self.last_readers[slot] = []

        # Update the next pointer of the previous ops
        for prev_op in prev_ops:
            prev_op.next.add(op)
            op.prev.add(prev_op)

    # InstructionDAG helper - identifies the dependencies for read-type operations (send, copy, reduce)
    def _read(self, rank, buffer, index, size, op):
        prev_ops = set()
        for i in range(index, index+size):
            slot = (rank, buffer, i)
            assert slot in self.last_writer, f"Slot has never been written before a read-type {op}"
            # The previous operation for a reader is the last write to the slot
            writer = self.last_writer[slot]
            prev_ops.add(writer)
            self.last_readers[slot].append(op)
            
        # Update the next pointer of the previous ops
        for prev_op in prev_ops:
            prev_op.next.add(op)
            op.prev.add(prev_op)

    # InstructionDAG - builds the roots of the DAG
    def add_start(self, ref):
        slot = (ref.rank, ref.buffer, ref.index)
        op = Op(Instruction.start, ref.rank, ref, ref, next=set(), prev=set())
        self.operations[slot] = op
        self.last_writer[slot] = op

    # InstructionDAG - adds a copy node
    def add_copy(self, rank, send_ref, recv_ref, tb, ch):
        self.num_instrs += 1
        op = Op(Instruction.copy, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel=ch)
        dstbuffer = recv_ref.buffer
        dstindex = recv_ref.index
        srcbuffer = send_ref.buffer
        srcindex = send_ref.index
        size = recv_ref.size
        # Sending part of copy [Read]
        self._read(rank, srcbuffer, srcindex, size, op)
        # Receiving part of copy [Write]
        self._write(rank, dstbuffer, dstindex, size, op)
        return op

    # InstructionDAG - adds a redduce node
    def add_reduce(self, rank, send_ref, recv_ref, tb, ch):
        self.num_instrs += 1
        op = Op(Instruction.reduce, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel=ch)
        dstbuffer = recv_ref.buffer
        dstindex = recv_ref.index
        srcbuffer = send_ref.buffer
        srcindex = send_ref.index
        size = recv_ref.size
        prev_ops = []
        # Sending part of reduce
        self._read(rank, srcbuffer, srcindex, size, op)
        # Reduce part of copy
        self._write(rank, dstbuffer, dstindex, size, op, read=True)
        return op

    # InstructionDAG - adds a send node
    def add_send(self, rank, send_ref, recv_ref, tb, ch):
        self.num_instrs += 1
        op = Op(Instruction.send, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel=ch)
        buffer = send_ref.buffer
        index = send_ref.index
        size = send_ref.size
        self._read(rank, buffer, index, size, op)
        self.sends.append(op)
        return op

    # InstructionDAG - adds a recv node
    def add_recv(self, rank, send_ref, recv_ref, tb, ch, send_op):
        self.num_instrs += 1
        op = Op(Instruction.recv, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel=ch)
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size
        self._write(rank, buffer, index, size, op)
        op.send_match = send_op
        return op

    # InstructionDAG - adds a rrc node
    def add_recv_reduce_copy(self, rank, send_ref, recv_ref, tb, ch, send_op):
        self.num_instrs += 1
        op = Op(Instruction.recv_reduce_copy, rank, send_ref, recv_ref, next=set(), prev=set(), tb=tb, channel=ch)
        buffer = recv_ref.buffer
        index = recv_ref.index
        size = recv_ref.size
        self._write(rank, buffer, index, size, op, read=True)
        op.send_match = send_op
        return op

    def convert_set_list(self):
        ops = []
        for slot, op in self.operations.items():
            if op.inst == Instruction.start:
                op.next = list(op.next)
                for o in op.next:
                    ops.append(o)
            elif op.inst != Instruction.copy:
                ops.append(op)

            visited = set()
            while len(ops) > 0:
                op = ops[0]
                if op not in visited:
                    visited.add(op)
                    op.next = list(op.next)
                    ops = ops[1:] + op.next
                else:
                    ops = ops[1:]
                    
    def optimize(self):
        self._instr_fusion()
        
    # Completes metadata for chunk_steps (number of steps from a start op) and priority (number of steps to the last op)
    def _complete_metadata(self):
        def dfs(op, cs):
            if op.chunk_step < cs+1:
                op.chunk_step = cs+1

                if len(op.next) == 0 and op.recv_match is None:
                    op.priority = 0
                else:
                    for o in op.next:
                        dfs(o, op.chunk_step)
                    # Priority = +1 of the highest priority child
                    if len(op.next) > 0:
                        highest_next_priority = max([x.priority+1 for x in op.next])
                        op.priority = max(highest_next_priority, op.priority)
                    if op.is_send():
                        dfs(op.recv_match, op.chunk_step)
                        op.priority = max(op.priority, op.recv_match.priority+1)

        for chunk, op in self.operations.items():
            if op.inst == Instruction.start:
                dfs(op,-2) # Start instructions should start at -1
    
    # Given the set of operations that operate over a particular slot (rank, buffer, idx) fixed
    # Try and replace operations with pipelined ops like receive copy send (rcs)
    # or receive reduce send (rrs) and receive reduce copy send (rrcs)
    # Rules:
    # recv-copy-send 
    # recv(src, sbuf, si, _, _, _ ) send(_, _, _, dst, dbuf, di) -> recv_copy_send(src, sbuf, si, dst, dbuf, di)
    # Given the set of operations that operate over a particular slot (rank, buffer, idx) fixed
    # Try and replace operations with pipelined ops like receive copy send (rcs)
    # or receive reduce send (rrs) and receive reduce copy send (rrcs)
    # Rules:
    # recv-copy-send 
    # recv(src, sbuf, si, _, _, _ ) send(_, _, _, dst, dbuf, di) -> recv_copy_send(src, sbuf, si, dst, dbuf, di)
    # recv-reduce-send - A rrc followed by a send that gets overwritten
    # rrc(src, sbuf, si, ...) send(_, _, _, dst, dbuf, di) recv(_, _, _, dst, dbuf, di) 
    # recv-reduce-copy-send - A rrc followed by a send that does not get overwritten
    # rrc(src, sbuf, si, ...) send(_, _, _, dst, dbuf, di)

    def _instr_fusion(self):
        start = time.time()
        def dfs(send_op, chain):
            op = send_op.recv_match
            for next_op in op.next:
                if next_op.inst == Instruction.send and same_tb(op, next_op) and same_count(op, next_op) and chain.can_add(next_op):
                    if op.inst == Instruction.recv:
                        op.inst = Instruction.recv_copy_send
                    else: # op.inst == Instruction.recv_reduce_copy:
                        nnext_op = next_op.next[0] if len(next_op.next) > 0 else None
                        if nnext_op and nnext_op.inst is Instruction.recv and same_count(op, nnext_op):
                            op.inst = Instruction.recv_reduce_send   
                        else:
                            op.inst = Instruction.recv_reduce_copy_send
    
                    op.fused_dst = next_op.dst 
                    next_op.recv_match.send_match = op
                    op.recv_match = next_op.recv_match
                    remove_op(next_op) # Remove the send that was fused into the previous instruction
                    chain.add(op)
                    self.num_instrs -= 1
                    return dfs(op, chain)
            return chain

        for send in self.sends:
            if send.inst == Instruction.send:
                chain = Chain(send)
                dfs(send, chain)
                if chain.length() > 2:
                    self.chains.append(chain)

    def lower_pt1(self, instances):
        self.lower_buffers(instances)
    
    def lower_pt2(self, instances, interleaved):
        self.replicate(instances, interleaved)
        return self.lower_tbs()


    def infer_dependencies(self, instances=False):
        tbs = self.instanced_tbs if instances else self.tbs
        for rank_tbs in tbs:
            for tb in rank_tbs.values():
                for op in tb.ops:
                    depends = {}
                    for dep_op in list(op.prev):
                        if dep_op.inst != Instruction.start and op.rank == dep_op.rank:
                            tb = dep_op.tb
                            if tb not in depends or dep_op.step > depends[tb].step:
                                depends[tb] = dep_op
                    op.depends = list(depends.values())

    # Convert local scratch buffers to index into one global scratch buffer
    def lower_chunk(self, chunk):
        if chunk.buffer is not Buffer.input and chunk.buffer is not Buffer.output:
            buffer = self.buffers[chunk.rank][chunk.buffer].get_buffer()
            index = self.buffers[chunk.rank][chunk.buffer].get_global_index(chunk.index)
            return ChunkRef(chunk.rank, buffer, index, chunk.size)
        return chunk

    # Assigns each scratch buffer an offset into the global scratch buffer
    def lower_buffers(self, instances):
        for rank_buffers in self.buffers:
            offset = 0
            for key, buf in rank_buffers.items():
                if key is not Buffer.input and key is not Buffer.output:
                    buf.set_offset(offset)
                    offset += buf.instance_size() * instances

    # Preprocess the threadblocks for lowering into xml
    def lower_tbs(self):
        gpus = []
        for rank, rank_tbs in enumerate(self.instanced_tbs):
            lowered_tbs = {}
            for tbid, tb in rank_tbs.items():
                for op in tb.ops:
                    op.src = self.lower_chunk(op.src)
                    op.dst = self.lower_chunk(op.dst)
                lowered_tbs[tbid] = tb
            gpus.append(Gpu(rank, list(lowered_tbs.values())))
        return gpus


    # Automatically replicates the algorithm instance number of times
    # interleaved sets the replication policy
    # if True chunks are split as: ChunkA ChunkB -> ChunkA0 ChunkA1 .. ChunkB0 ChunkB1 ...
    # if false chunks are divided as ChunkA0 ChunkB0 ChunkA1 ChunkB1 ...
    # For collectives were chunks are designated for a particular GPU (e.g. AllToAll) 
    # only interleaved replication will be correct
    # Interleaved policy only supports single count sends/receives from the input/output buffer
    # (multicount ops are fine between scratch)
    def replicate(self, instances, interleaved):
        if instances == 1:
            self.instanced_tbs = self.tbs
            self.infer_dependencies()
            return 
        
        self.instanced_tbs = []
       
        for _ in range(self.num_ranks):
            self.instanced_tbs.append({})

        def is_scratch(buffer):
            return buffer != Buffer.input and buffer != Buffer.output

        def get_new_index(rank, buffer, index, size, i):
            # Scratch buffers always use batched
            if is_scratch(buffer):
                buf_instance_len = self.buffers[rank][buffer].instance_size()
                return buf_instance_len * i + index
            # If this is operating on the input/output buffer then replication strategy can be either interleaved or batched
            # This is to fit with the semantics of certain collectives
            elif interleaved:
                return  index * instances + i * size
            else:
                return  len(self.buffers[rank][buffer]) * i + index

        def get_instance_ref(ref, i):
            iindex = get_new_index(ref.rank, ref.buffer, ref.index, ref.size, i)
            iref = ChunkRef(ref.rank, ref.buffer, iindex, ref.size)
            return iref

        def add_instance_op(parent_op, i):
            ichan = max_channels * i + parent_op.channel
            itbid = parent_op.tb * instances + i
            isrc = get_instance_ref(parent_op.src, i)
            idst = get_instance_ref(parent_op.dst, i)
            rank = parent_op.rank

            op = Op(parent_op.inst, rank, isrc, idst, next=set(), prev=set(), step=parent_op.step, tb=itbid, channel=ichan)
            dstbuffer = idst.buffer
            dstindex = idst.index
            srcbuffer = isrc.buffer
            srcindex = isrc.index
            size = idst.size
            inst_read = op.is_reduce()

            if op.is_fused(): # RRS RCS RRCS
                self._write(rank, dstbuffer, dstindex, size, op, read=inst_read)
                # self._read(rank, srcbuffer, srcindex, size, op)
            elif op.is_local():
                # Sending part of fused instruction [Read]
                self._read(rank, srcbuffer, srcindex, size, op)
                # Receiving part of fused instruction [Write]
                self._write(rank, dstbuffer, dstindex, size, op, read=inst_read)
            elif op.is_send(): # S
                self._read(rank, srcbuffer, srcindex, size, op)
            else: # RRC, R
                self._write(rank, dstbuffer, dstindex, size, op, read=inst_read)
            return op


        max_channels = max(self.num_channels)
        # Generate all instanced threadblocks
        for i in range(instances):
            for rank, rank_tbs in enumerate(self.tbs):
                for tbid, tb in rank_tbs.items():
                    instance_channel = max_channels * i + tb.channel
                    itb = Threadblock(instance_channel, tb.send, tb.recv)
                    itbid = tbid * instances + i
                    self.instanced_tbs[rank][itbid] = itb

        # Redo dependency analysis
        # 1. Build the instanced Rank DAG but with the threadblock/channel assignment of the base Rank DAG
        # Clear prior state for building the DAG
        self.operations = {} # slot -> operations
        self.last_writer = {} # slot -> last writing op
        self.last_readers = defaultdict(list) # slot -> list of last reading ops
        # Initialize starting instanced chunks
        for rank, rank_buffers in enumerate(self.buffers):
            for bufname, buffer in rank_buffers.items():
                for index in range(len(buffer)):
                    for i in range(instances):
                        ref = ChunkRef(rank, bufname, index*instances+i, 1)
                        self.add_start(ref)

        # Walk through the topological sort and build the instanced Rank DAG
        # Add instanced ops to their proper threadblocks
        for op in self.ordered_instrs:
            for i in range(instances):
                iop = add_instance_op(op, i)
                self.instanced_tbs[op.rank][iop.tb].ops.append(iop)

        self.convert_set_list()
        self.infer_dependencies(instances=True)