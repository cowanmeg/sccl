# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
import heapq
from msccl.language.ir import *
from msccl.language.instruction_dag import *

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
            instr_dag.num_channels[rank] = max(op.channel+1, instr_dag.num_channels[rank])
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
        send_match = tb_s == send and send != -1
        recv_match = tb_r == recv and recv != -1
        channel_ok = channel == tb_c
        local = send == -1 and recv == -1
        tb_local = tb_s == -1 and tb_r == -1

        # For correctness - if one of the peer's channels is already allocated we must use it.
        if channel_ok and (send_match or recv_match):
            options.append(tbid)
        elif channel_ok and tb_local and local:
            options.append(tbid)
        # TODO: Uncomment out if copies should be dispersed.
        # if sender_ok and receiver_ok and channel_ok and (send != -1 or recv != -1):
        #      options.append(tbid)

    if len(options) > 1:
        print("send:", send, "recv:", recv, "chan", channel)
        print(options)
        for tbid in options:
            print(mapping[tbid])
        print(" ")
    return options

def _create_threadblock(instr_dag, rank_tbids, rank, s, r, channel):
    tbid = rank_tbids[rank]
    tb = Threadblock(send=s, recv=r, channel=channel)
    instr_dag.tbs[rank][tbid] = tb 
    rank_tbids[rank] += 1
    tb.send = s
    tb.recv = r
    return tbid

def auto_assign_tbs(instr_dag):
    channel_assignment(instr_dag)
    instrs = topo_sort_instrs(instr_dag)
    rank_tbids = [0] * instr_dag.num_ranks
    current_tb_step = []
    for rank_tbs in instr_dag.tbs:
        current_tb_step.append({})
    send_con_map = []
    recv_con_map = []
    for _ in range(instr_dag.num_ranks):
        send_con_map.append(defaultdict(lambda: -1))
        recv_con_map.append(defaultdict(lambda: -1))
    
    # Create all fused instruction threadblocks
    for op in instrs:
        if op.is_fused():
            rank = op.rank
            s = op.send_peer()
            r = op.recv_peer()
            channel = op.channel
            
            # Check that one connection has been assigned but not the other - this is a bug
            if send_con_map[rank][s, channel] != recv_con_map[rank][r, channel]:
                print("Error: This should not happen....")
            # Assign the connections to a threadblock if they haven't already
            elif send_con_map[rank][s, channel] == -1:
                tbid = _create_threadblock(instr_dag, rank_tbids, rank, s, r, channel)
                current_tb_step[rank][tbid] = 0
                send_con_map[rank][s, channel] = tbid
                recv_con_map[rank][r, channel]= tbid

    for op in instrs:
        rank = op.rank
        s = op.send_peer()
        r = op.recv_peer()

        channel = 0 if op.channel == -1 else op.channel

        # Get all possible TBs this can be mapped to
        tb_options = _get_tb_options(instr_dag.tbs[rank], s, r, channel, rank_tbids[rank])

        if len(tb_options) == 0: # If there are no options, create a new threadblock
            tbid = _create_threadblock(instr_dag, rank_tbids, rank, s, r, channel)
            current_tb_step[rank][tbid] = 0
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
        
        op.step = len(tb.ops)-1
        op.tb = tbid
        current_tb_step[rank][tbid] = op.chunk_step

def debug_print(instr_dag):
    for rank in range(instr_dag.num_ranks):
        print(f"Rank={rank}")
        for tbid, tb in instr_dag.tbs[rank].items():
            print(f"  TB={tbid} send={tb.send} recv={tb.recv} ch={tb.channel}")
            for op in tb.ops:
                print(priority(op), op)

def priority(op):
    return ((op.chunk_step, -op.priority, op.dst.index))

# Topologically orders instructions so that 
# (1): Sends occur before their receives
# (2): Instruction dependencies are respected
def topo_sort_instrs(instr_dag):
    insert_connection_dependencies(instr_dag)
    visited = set()
    ops = []
    ordered = []
    for op in instr_dag.operations.values():
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
                    if not o.is_recv() or o.send_match in visited:
                        heapq.heappush(ops, (priority(o), o))

    instr_dag.ordered_instrs = ordered
    # This error sometimes shows up with manual tb assignment or instruction fusion introducing a circular dependency
    assert instr_dag.num_instrs == len(ordered), \
        f'Error TB assignment: Instructions in DAG: {instr_dag.num_instrs} vs. instructions scheduled {len(ordered)}'
    return ordered

# TODO: Merge flow channel assignment with fusion
def channel_assignment(instr_dag):
    flows = []
    flow_channels = []

    def all_channels():
        return set([x for x in range(32)])
    rank2sendch = [defaultdict(all_channels) for _ in range(instr_dag.num_ranks)]
    rank2recvch = [defaultdict(all_channels) for _ in range(instr_dag.num_ranks)]

    # Returns a channel this flow can be scheduled on, else -1 
    def is_matching_flow(flow):
        # Exact match
        if flow in flows:
            ch = flow_channels[flows.index(flow)]
            return flow_channels[flows.index(flow)]
        # Check if this flow is a subset of an existing flow
        # TODO: Why is this causing issues?
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

    def assign_flow_channel(chain):
        flow = chain.connection_set()
        user_ch = chain.ops[0].channel
        ch = is_matching_flow(flow)
        if ch == -1 or user_ch != ch: # No flow matched - use the smallest available channel
            possible_channels = all_channels()
            for i in range(0, len(chain.ops)-1):
                op = chain.ops[i]
                sender = op.rank
                receiver = op.send_peer()
                possible_channels = rank2sendch[sender][receiver].intersection(rank2recvch[receiver][sender]).intersection(possible_channels)
            # If the program specified a channel try to respect it
            # Might not be possible due to valid tb-channel constraints
            if user_ch in possible_channels:
                ch = user_ch
            else:
                ch = min(possible_channels)
            flows.append(flow)
            flow_channels.append(ch)

        for op in chain.ops:
            if op.is_send():
                reserve_channel(op.rank, op.send_peer(), ch)
            op.channel = ch

    # Assign channels to flows
    for chain in instr_dag.chains:
        assign_flow_channel(chain)

    # Assign all remaining (send, recv) to channel 0
    for send in instr_dag.sends:
        if send.inst == Instruction.send and not send.recv_match.is_fused() and send.channel == -1:
            send.channel = 0 
            send.recv_match.channel = 0

# Inserts extra edges in the DAG to ensure 
# 1. Remote buffer slots aren't blocking
# 2. Chunks sent over a channel are received in the same order
def insert_connection_dependencies(instr_dag):
    slots = 2 if instr_dag.protocol == 'Simple' else 8
    connections = defaultdict(list) # A connection is uniquely identified by (send_peer, recv_peer, channel)

    visited = set()
    def iterate(frontier):
        while len(frontier) > 0:
            _, op = heapq.heappop(frontier)
            if op not in visited:
                rmatch = op.recv_match
                visited.add(op)

                if op.is_send():
                    rank = op.rank
                    recv_peer = op.recv_match.rank
                    channel = op.channel
                    instrs = connections[(rank, recv_peer, channel)]
                    instrs.append(op)
                
                # Add a matching receive if one exists and its dependencies are satisfied
                if rmatch is not None and all([x in visited for x in rmatch.prev]): 
                    heapq.heappush(frontier, (priority(rmatch), rmatch))
                # Add other operation that have dependencies satisfied
                for o in op.next:
                    if all([x in visited for x in o.prev]):
                        if not o.is_recv() or o.send_match in visited:
                            heapq.heappush(frontier, (priority(o), o))
    frontier = []
    for op in instr_dag.operations.values():
        if op.inst == Instruction.start:
            heapq.heappush(frontier, (priority(op), op))     
    iterate(frontier)

    for instrs in connections.values():
        # Remote buffer constraint. Across a connection only slot number of sends are allowed to be buffered
        # before the receiver reads
        for i in range(slots, len(instrs)):
            send = instrs[i]
            buffer_slot_recv = instrs[i-slots].recv_match
            # Send cannot be scheduled until buffer_slot_recv is scheduled or else the buffer will be full.
            send.prev.add(buffer_slot_recv)
            buffer_slot_recv.next.append(send)

        # In-order constraint. Across a connection: send a, send b --> recv a, recv b
        for i in range(0, len(instrs)-1):
            send0 = instrs[i]
            send1 = instrs[i+1]
            recv0 = send0.recv_match
            recv1 = send1.recv_match

            recv0.next.append(recv1)
            recv1.prev.add(recv0)
    _detect_cycle(instr_dag)
    

def _detect_cycle(instr_dag):
    def deep_copy(s):
        c = list()
        for i in s:
            c.append(i)
        return c

    def dfs(op, ops):
        if op in ops:
            print("CYCLE", op)
            for o in ops:
                print(o)
            sys.exit()
        else:
            ops.append(op)

        for o in op.next:
            dfs(o, deep_copy(ops))

    for chunk, op in instr_dag.operations.items():
        if op.inst == Instruction.start:
            dfs(op, list())
