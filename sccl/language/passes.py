# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from sccl.language.ir import *

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
def check_deadlock(tbs, instr_dag):
    def finished(tbs):
        fin = True
        for rank, rank_tbs in enumerate(tbs):
            for tbid, tb in rank_tbs.items():
                if tb.step < len(tb.ops):
                    return False
        return True

    executed = set()
    progress = True
    while progress and not finished(tbs):
        progress = False
        for rank, rank_tbs in enumerate(tbs):
            for tbid, tb in rank_tbs.items():
                tb_blocked = False
                while tb.step < len(tb.ops) and not tb_blocked:
                    op = tb.ops[tb.step]
                    op_ready =  len(op.depends) == 0 or set(op.depends).issubset(executed)

                    if op_ready and op.is_recv() and op.is_send():
                        recv_tb = instr_dag.tbs[op.recv_match.rank][op.recv_match.tb]
                        # Simulate the receive happened
                        tb.buf_full = False 

                        if recv_tb.buf_full:
                            # If the send can't happen - the instruction blocks
                            tb_blocked = True
                        else:
                            # Simulate the send portion
                            recv_tb.buf_full = True
                            progress = True
                            executed.add(op)
                            tb.step += 1

                    elif op_ready and op.is_recv():
                        # Simulate emptying the buffer
                        tb.buf_full = False
                        progress = True
                        executed.add(op)
                        tb.step += 1
                    elif op_ready and op.is_send():
                        recv_tb = instr_dag.tbs[op.recv_match.rank][op.recv_match.tb]
                        if recv_tb.buf_full:
                            tb_blocked = True
                        else:
                            # Simulate the recv_buffer being filled
                            recv_tb.buf_full = True
                            progress = True
                            executed.add(op)
                            tb.step += 1
                    else:
                        tb_blocked = True
    
    if not finished(tbs):
        print("ERROR!!!! Deadlock from blocking sends")

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
                            print("Sending tb")
                            for op in tb.ops:
                                print(f'{op.step}: Recv step: {op.recv_match.step if op.is_send() else -1} {op} priority:{(op.chunk_step, op.priority, op.dst.index)}')
                            print("Receiving tb")
                            for op in instr_dag.tbs[match.rank][other_tbid].ops:
                                print(f'{op.step}: {op} priority:{(op.chunk_step, op.priority, op.dst.index)}')
                            assert match.step >  prev_steps[other_tbid].step, f"Rank {op.rank} sends op1 then op2 but {match.rank} receives op2 then op1"
                        
                    prev_steps[other_tbid] = match

