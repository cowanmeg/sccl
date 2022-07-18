# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
import heapq

from numpy import insert

from sccl.language.ir import *
from sccl.language.rank_dag import *

def _verify_tb_op_compatible(tb, op):
    s = op.send_peer() if op.is_send() else -1
    r = op.recv_peer() if op.is_recv() else -1
        
    sends_ok = tb.send == s or s == -1 or tb.send == -1
    recvs_ok = tb.recv == r or r == -1 or tb.recv == -1
    channel_ok = tb.channel == op.channel or tb.channel == -1 or op.channel == -1
    return sends_ok and recvs_ok and channel_ok

# Manual threadblock, channel assignment
def manual_assign_tbs(instr_dag):
    instrs = topo_sort_instrs(instr_dag)
    for op in instrs:
        rank = op.rank
        tbid = op.tb
        if tbid not in instr_dag.tbs[rank]:
            instr_dag.tbs[rank][tbid] = Threadblock()
        tb = instr_dag.tbs[rank][tbid]
        if _verify_tb_op_compatible(tb, op):
            tb.ops.append(op)
            tb.channel = op.channel if op.channel != -1 else 0
            tb.send = op.send_peer() if op.is_send() else tb.send
            tb.recv = op.recv_peer() if op.is_recv() else tb.recv
            op.step = len(tb.ops)-1
            instr_dag.num_channels[rank] = max(op.channel+1, instr_dag.num_channels[rank] )
        else:
            raise Exception(f"Illegal threadblock assignment. Trying to add {op} to threadblock {tbid}\n" \
                f"Threadblock {tbid} send:{tb.send} recv:{tb.recv} channel:{tb.channel}\n" \
                f"Operation send:{op.dst.rank if op.is_send() else -1} recv:{op.dst.rank if op.is_recv() else -1} channel:{op.channel}")

def _get_tb_options(mapping, send, recv, channel, num_tbs):
    options = []
    for tbid, tb in mapping.items():
        tb_s = tb.send
        tb_r = tb.recv
        tb_c = tb.channel
        sender_ok = send == -1 or tb_s == send
        receiver_ok = recv == -1 or tb_r == recv
        channel_ok = channel == -1 or channel == tb_c
        # For correctness - if one of the peer's channels is already allocated we must use it.
        if channel_ok and ((tb_s == send and send != -1) or (tb_r == recv and recv != -1)):
            return [tbid]
        if sender_ok and receiver_ok and channel_ok:
             options.append(tbid)
    return options

def auto_assign_tbs(instr_dag):
    instrs = topo_sort_instrs(instr_dag)
    channel_assignment(instrs, instr_dag)
    rank_tbids = [0] * instr_dag.num_ranks
    current_tb_step = []
    for rank_tbs in instr_dag.tbs:
        current_tb_step.append({})

    for op in instrs:
        rank = op.rank
        s = op.send_peer()
        r = op.recv_peer()

        channel = 0 if op.channel == -1 else op.channel

        # Get all possible TBs this can be mapped to
        tb_options = _get_tb_options(instr_dag.tbs[rank], s, r, channel, rank_tbids[rank])
        if len(tb_options) == 0: # If there are no options, create a new threadblock
            tbid = rank_tbids[rank]
            instr_dag.tbs[rank][tbid] = Threadblock(send=s, recv=r, channel=channel)
            rank_tbids[rank] += 1
        else: 
            tbid = tb_options[0]
            for tbid_opt in tb_options:
                if current_tb_step[rank][tbid_opt] < current_tb_step[rank][tbid] and _verify_tb_op_compatible(instr_dag.tbs[rank][tbid], op):
                    tbid = tbid_opt
        
        tb = instr_dag.tbs[rank][tbid]
        assert _verify_tb_op_compatible(tb, op), f"{op.rank} Failing: Operations uses channel {op.channel}, send:{s} recv:{r} {op}\n" \
                f"Threadblock uses send:{tb.send} recv:{tb.recv} channel:{tb.channel}"

        instr_dag.num_channels[rank] = max(instr_dag.num_channels[rank], channel+1)

        tb.ops.append(op)
        tb.send = s if op.is_send() else tb.send
        tb.recv = r if op.is_recv() else tb.recv
        
        op.step = len(tb.ops)-1
        op.tb = tbid
        current_tb_step[rank][tbid] = op.chunk_step

def priority(op):
    return ((op.chunk_step, -op.priority, op.dst.index))

# Topologically orders instructions so that (1): Sends occur before their receives
# (2): Dependent instructions occur before 
def topo_sort_instrs(instr_dag):
    insert_buffer_dependencies(instr_dag)

    visited = set()
    ops = []
    ordered = []
    for slot, op in instr_dag.operations.items():
        if op.inst == Instruction.start:
            visited.add(op)
            for o in op.next:
                if (o.inst == Instruction.send or o.inst == Instruction.copy) and all([x in visited for x in o.prev]):
                    heapq.heappush(ops, (priority(o), o))

    while len(ops) > 0:
        _, op = heapq.heappop(ops)
        if op not in visited:
            rmatch = op.recv_match
            ordered.append(op)
            visited.add(op)
            
            # Add a matching receive if one exists and its dependencies are satisfied
            if rmatch is not None and all([x in visited for x in rmatch.prev]): 
                heapq.heappush(ops, (priority(rmatch), rmatch))
            # Add other operation that have dependencies satisfied
            for o in op.next:
                if all([x in visited for x in o.prev]):
                    heapq.heappush(ops, (priority(o), o))

    instr_dag.ordered_instrs = ordered
    return ordered

def channel_assignment(instrs, instr_dag):
    def all_channels():
        return set([x for x in range(32)])    # First handle flows - if an instruction at Rx is fused Rw->Rx->Ry and takes c
    # Then flow Rw->Rx->Rz must be ib a different channel c' where c!=c'
    rank2sendch = [defaultdict(all_channels) for _ in range(instr_dag.num_ranks)]
    rank2recvch = [defaultdict(all_channels) for _ in range(instr_dag.num_ranks)]

    # DFS through the InstructionDAG identifying flows
    def valid_send_ch(sender, receiver, ch):
        return ch in rank2sendch[sender][receiver]
    def valid_recv_ch(sender, receiver, ch):
        return ch in rank2recvch[receiver][sender]

    # Returns a channel this flow can be scheduled on, else -1 
    def is_matching_flow(flow):
        # Exact match
        if flow in flows:
            ch = flow_channels[flows.index(flow)]
            return flow_channels[flows.index(flow)]
        # Check if this flow is a subset of an existing flow
        # for existing_flow in flows:
        #     if flow.issubset(existing_flow):
        #         return flow_channels[flows.index(existing_flow)]
        # No match
        return -1

    def reserve_channel(sender, receiver, ch):
        if ch in rank2sendch[sender][receiver]:
            rank2sendch[sender][receiver].remove(ch)
        if ch in rank2recvch[receiver][sender]:
            rank2recvch[receiver][sender].remove(ch)
    flows = []
    flow_channels = []

    def create_flow(f):
        flow = set()
        for i in range(1, len(f)):
            flow.add((f[i-1], f[i]))
        return flow
        
    def dfs(op, channels, f):
        if op.is_local():
            op.channel = 0
        elif op.is_send():
            match = op.recv_match
            sender = op.rank
            receiver = match.rank
            # Available channels
            channels = rank2sendch[sender][receiver].intersection(rank2recvch[receiver][sender]).intersection(channels)
            f.append(op.rank)
            # If not a fused op use the first possible channel (send, recv/rrc)
            if not match.is_fused():
                f.append(match.rank)
                flow = create_flow(f)
                # If the user has already manually scheduled this onto a channel, respect it
                if op.channel != -1:
                    ch = op.channel
                else:
                    ch = is_matching_flow(flow)
                    if ch == -1: # No flow matched - use the smallest available channel
                        ch = min(channels)
                        flows.append(flow)
                        flow_channels.append(ch)

                op.channel = ch
                match.channel = ch
                reserve_channel(sender, receiver, ch)
            else:
                dfs(match, channels, f)
                ch = match.channel
                op.channel = ch
                reserve_channel(sender, receiver, ch)

    # Assign channels to flows
    for slot, op in instr_dag.operations.items():
        if op.inst == Instruction.start:
            for o in op.next:
                if o.inst == Instruction.send and o.recv_match.is_fused():
                    dfs(op, all_channels(), [])

    # Iterate through and make certain the sends and receives between a pair of GPUs is consistent
    # Shift a (s,r) pair to another channel if the ordering isn't consistent
    repeat = True
    while repeat:
        repeat = False
        pending_recv = defaultdict(list)  # (sender, receiver, ch) -> pending receive
        for op in instrs:
            rank = op.rank
            channel = 0 if op.channel == -1 else op.channel
            if op.is_send():
                dst = op.dst.rank
                pending_recv[(rank, dst, channel)].append(op.recv_match)
            
            if op.is_recv():
                src = op.src.rank
                pr = pending_recv[(src, rank, channel)]
                if op in pr:
                    if pr[0] is op:
                        del pr[0]
                    else:
                        repeat = True
                        op.channel += 1
                        op.send_match.channel += 1
                        print(f"+=1  to {op.channel}")
                        pr.remove(op)

# Inserts extra edges in the DAG to ensure sends aren't blocked by buffer space.
def insert_buffer_dependencies(instr_dag):
    slots = 2 if instr_dag.protocol == 'Simple' else 8
    connections = defaultdict(list) # A connection is uniquely identified by (rank, recv_peer, channel)

    visited = set()
    def bfs(frontier):
        i = 0
        while(i < len(frontier)):
            op = frontier[i]
            if op not in visited:
                visited.add(op)
                if op.is_send():
                    rank = op.rank
                    recv_peer = op.recv_peer()
                    channel = op.channel
                    instrs = connections[(rank, recv_peer, channel)]
                    heapq.heappush(instrs, (priority(op), op))
                frontier += op.next
            i += 1
    
    frontier = []
    for op in instr_dag.operations.values():
        if op.inst == Instruction.start:
            frontier.append(op)
    bfs(frontier)
    for c, instrs in connections.items():
        for i in range(slots, len(instrs)):
            _, inst = instrs[i]
            _, prev_inst = instrs[i-slots]
            inst.prev.add(prev_inst.recv_match)
            prev_inst.recv_match.next.append(inst)

