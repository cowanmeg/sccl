# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from collections import deque
from msccl.language.ir import *

buf_size = 4194304 # 4MB
def num_sends(count, chunk_size, slots):
    # A single chunk is tiled to fit into the buffer. 
    chunk_size = max(buf_size, chunk_size) 
    multi_count_split = buf_size // chunk_size * count
    trigger_point = slots * multi_count_split
    print(f"Send of {count} will trigger a deadlock at {trigger_point}B given {slots} slots")

# Simulate the program to catch deadlocks caused by fixed remote buffer sizes.
# Assume a sends of count=1 are guaranteed to fit into the remote buffer due to tiling
# At some size ?? a multi-count send will serially broken up into sends
class SimulatedThreadblock:
    def __init__(self, rank, tbid, tb, protocol):
        self.rank = rank
        self.tbid = tbid
        self.tb = tb
        self.protocol = protocol
        self.remote_buffer = RemoteBuffer(protocol)
        self.step = -1
        self.chunks_remaining = 0

    def instr_done(self):
        # Finished when there are no chunks remaining 
        return self.chunks_remaining == 0 

    def next_inst(self):
        if  self.instr_done() and self.step < len(self.tb)-1:
            self.step += 1
            if self.tb[self.step].inst is Instruction.nop:
                self.chunks_remaining = 1
            else:
                self.chunks_remaining = self.tb[self.step].src.size
        return self.tb[self.step]

    # Simulates part of an instruction
    # Returns whether or not progress was made, whether or not instruction finished
    def simulate(self, recv_tb):
        op = self.tb[self.step]
        if op.is_recv() and op.is_send(): # RRCS, RRS, RCS
            writeable_slots = recv_tb.remote_buffer.open_slots()
            chunks_read = self.remote_buffer.read(min(self.chunks_remaining, writeable_slots))
            chunks_processed = recv_tb.remote_buffer.write(chunks_read)
        else:
            if op.is_recv(): # RRC, R
                chunks_processed = self.remote_buffer.read(self.chunks_remaining)
            elif op.is_send(): # S
                chunks_processed = recv_tb.remote_buffer.write(self.chunks_remaining)
            else: # Local op
                chunks_processed = self.chunks_remaining
        self.chunks_remaining -= chunks_processed
        return chunks_processed > 0, self.chunks_remaining == 0

    def finished(self):
        # Finished when the last instruction of the threadblock is done
        return self.step == len(self.tb)-1 and self.instr_done()

class RemoteBuffer:
    def __init__(self, protocol, remotebuffer_mb=4):
        if protocol == 'Simple':
            self.slots = 2
        else: # LL and LL128
            self.slots = 8
        self.chunks = deque(maxlen=self.slots)

    def used_slots(self):
        return len(self.chunks)

    def open_slots(self):
        return self.slots - len(self.chunks)

    # Returns the number of chunks written
    def write(self, count):
        c = 0
        while (self.slots - len(self.chunks)) > 0 and c < count:
            self.chunks.append(count)
            c += 1
        return c

    # Returns the number of chunks read
    def read(self, count):
        c = 0
        while len(self.chunks) >  0 and c < count:
            self.chunks.pop()
            c += 1
        return c
    

# Check that there are no cyclic dependencies within a rank
# This check is a subset of the check_deadlock program to narrow
# if deadlock is caused by explicit cross-threadblock dependencies inserted by the compiler
def check_dependency_cycles(tbs):
    for rank, rank_tbs in enumerate(tbs):
        for tbid, tb in rank_tbs.items():
            for op in tb.ops:
                deps = op.depends
                chain = [op]
                # DFS to check for cycles
                while len(deps) > 0:
                    dep = deps[0]
                    if dep in chain:
                        print(f"Cyclic dependency in rank {rank} threadblock {tbid} at {op}")
                        for op in chain:
                            print("  ", op)
                        sys.exit(1)
                    next_depends = dep.depends
                    if len(next_depends) > 0:
                        chain.append(dep)
                    else:
                        chain = [op]
                    deps = next_depends + deps[1:]

# Checks there are no deadlocks across the program
# We assume the runtime tiles execution of the program so that the maximum program transfer unit
# max(chunk_size * cnt) can fit into a connection's recv_buffer
# We conservatively model the program as follows: an instruction can execute if all its explicit dependcies have executed and 
# if it is a send that the recv_buffer on the receiving threadblock is empty
# i.e. we model the recv_buffer as only able to hold chunks for one pending recv regardless if two recvs
# could fit in the buffer. 
def check_deadlock(instr_dag, protocol):
    # Program finishes when all threadblocks execute their instructions
    def finished(tbs):
        for rank, rank_tbs in enumerate(tbs):
            for tb in rank_tbs.values():
                if not tb.finished():
                    return False
        return True

    # Create remote buffers for all threadblocks (for easy indexing assume everyone has a recv peer)
    tbs = []
    for rank, rtbs in enumerate(instr_dag.tbs):
        rank_tbs = {}
        for tbid, tb in rtbs.items():
            rank_tbs[tbid] = (SimulatedThreadblock(rank, tbid, tb.ops, protocol))
        tbs.append(rank_tbs)
    
    executed = set()
    progress = True
    while progress and not finished(tbs):
        progress = False
        for rank, rank_tbs in enumerate(tbs):
            for tbid, tb in rank_tbs.items():
                tb_blocked = False
                while not tb.finished() and not tb_blocked:
                    op = tb.next_inst()
                    op_ready =  len(op.depends) == 0 or set(op.depends).issubset(executed)
                    if op_ready:
                        if op.is_send():
                            recv_tb = tbs[op.recv_match.rank][op.recv_match.tb]
                        else:
                            recv_tb = None
                        tb_progress, instr_finished = tb.simulate(recv_tb)
                        progress = progress or tb_progress
                        tb_blocked = not tb_progress
                        if instr_finished:
                            executed.add(op)
                    else:
                        tb_blocked = True

    if not finished(tbs):
        print("ERROR DEADLOCK!")
        for rank, rank_tbs in enumerate(tbs):
            for tb in rank_tbs.values():
                if not tb.finished():
                    print(f"Rank {rank} Threadblock step:{tb.step} Chunks remaining:{tb.chunks_remaining}")
                    print(f"TB op {tb.tb[tb.step]}")


# Creates a dependency between a send and recv.
# Add these edges to check if the program is valid for blocking sends
def add_blocking_send_edges(tbs):
    for rank, rank_tbs in enumerate(tbs):
        for tbid, tb in rank_tbs.items():
            for op_step, op in enumerate(tb.ops):
                # If a send is blocking then it doesn't return until its recv happens
                if op.is_send():
                    op.depends.append(op.recv_match)


# Check there are no ordering violations between threadblocks across ranks
def check_threadblock_ordering(instr_dag):
    for rank in range(instr_dag.num_ranks):
        for tb in instr_dag.tbs[rank].values():
            prev_steps = {} # tbid -> step of last recv from tbid
            # Check that sends and their corresponding receives between two threadblocks
            # happen in the same order.
            for op_step, op in enumerate(tb.ops):
                if op.is_send():
                    match = op.recv_match
                    assert match.is_recv, "Bug in SCCLang: Send has no matching receive"
                    if op.is_fused():
                        assert op.fused_dst.rank == match.rank, f"Bug in SCCLang: Sends don't match receives"
                    else:
                        assert op.dst.rank == match.rank, f"Bug in SCCLang: Sends don't match receives"

                    other_tbid = match.tb
                    if other_tbid in prev_steps:
                        if match.step <= prev_steps[other_tbid].step:
                            print("Offending Steps", match.step, prev_steps[other_tbid].step)
                            sender = op.rank
                            receiver = match.rank
                            print(f"Sending tb Rank:{sender}")
                            for op in tb.ops:
                                print(f'{op.step}: Recv step: {op.recv_match.step if op.is_send() else -1} {op} priority:{(op.chunk_step, op.priority, op.dst.index)}')
                            print(f"Receiving tb Rank:{receiver}")
                            for op in instr_dag.tbs[match.rank][other_tbid].ops:
                                print(f'{op.step}: {op} priority:{(op.chunk_step, op.priority, op.dst.index)}')
                            # assert match.step >  prev_steps[other_tbid].step, f"Rank {sender} sends op1 then op2 but {receiver} receives op2 then op1"
                        
                    prev_steps[other_tbid] = match
