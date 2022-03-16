# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
import heapq
import threading, queue
from z3 import *

from sccl.language.ir import *
from sccl.language.rank_dag import *


def _verify_tb_op_compatible(tb, op):
    s = op.dst.rank if op.is_send() else -1
    r = op.src.rank if op.is_recv() else -1
        
    sends_ok = tb.send == s or s == -1 or tb.send == -1
    recvs_ok = tb.recv == r or r == -1 or tb.recv == -1
    channel_ok = tb.channel == op.channel or tb.channel == -1 or op.channel == -1
    return sends_ok and recvs_ok and channel_ok

# Manual threadblock, channel assignment
def manual_assign_tbs(rank_dag):
    ops = []
    for slot, op in rank_dag.operations.items():
        if op.inst == Instruction.start:
            for o in op.next:
                if o.inst == Instruction.send or o.inst == Instruction.copy:
                    heapq.heappush(ops, o)

    visited = set()
    while len(ops) > 0:
        op = heapq.heappop(ops)
        if op not in visited:
            visited.add(op)
            rank = op.rank
            tbid = op.tb
            if tbid not in rank_dag.tbs[rank]:
                rank_dag.tbs[rank][tbid] = Threadblock()
            tb = rank_dag.tbs[rank][tbid]
            if _verify_tb_op_compatible(tb, op):
                tb.ops.append(op)
                tb.channel = op.channel if op.channel != -1 else 0
                tb.send = op.dst.rank if op.is_send() else tb.send
                tb.recv = op.src.rank if op.is_recv() else tb.recv
                op.step = len(tb.ops)-1
                rank_dag.num_channels[rank] = max(op.channel+1, rank_dag.num_channels[rank] )
            else:
                raise Exception(f"Illegal threadblock assignment. Trying to add {op} to threadblock {tbid}\n" \
                    f"Threadblock {tbid} send:{tb.send} recv:{tb.recv} channel:{tb.channel}\n" \
                    f"Operation send:{op.dst.rank if op.is_send() else -1} recv:{op.dst.rank if op.is_recv() else -1} channel:{op.channel}")
            
            for o in op.next:
                heapq.heappush(ops, o)
            for o in op.match:
                heapq.heappush(ops, o)
# @dataclass
# class C:
#     ch: int

#     def __hash__(self):
#         return hash(self.ch)

# def channel_assignment(rank_dag):
#     # This function solves a coloring problem to ops to a minimal set of channels
#     ctx = Context()

#     # Channel variable -> Each send-type op gets assigned a channel variable
#     def chan(idx):
#         return Int(f'chan_{idx}', ctx=ctx)
#     max_channels = Int('max_channels', ctx=ctx)
#     constraints = []

#     # Fused instruction "flows" must be on the same channel
#     ops = []
#     for slot, op in rank_dag.operations.items():
#         if op.inst == Instruction.start:
#             for o in op.next:
#                 if o.inst == Instruction.send or o.inst == Instruction.copy:
#                     ops.append(o)

#     # Find all "flows" in the program
#     # Find all unique edges
#     i = 0
#     visited = set()
#     chains = []
#     op_ids = []
#     edges = defaultdict(set)
#     while i < len(ops):
#         op = ops[i]
#         if (op.inst == Instruction.send or op.inst == Instruction.copy) and op not in visited:
#             print("Start", op.rank, op.inst)
#             visited.add(op)
#             rank = op.rank
#             chain = [rank]
#             op_id = [op.num]
#             match = op.match
#             while match is not None:
#                 print("  ", match.rank, match.inst)
#                 visited.add(match)
#                 chain.append(match.rank)
#                 op_id.append(match.num)
#                 match = match.match
#             if len(chain) > 2:
#                 chains.append(chain)
#                 op_ids.append(op_id)
#             ops += op.next
#         i += 1

#     # Rotate all flows to begin at the smallest index
#     # Remove duplicates
#     flows = []
#     flow_ids = []
#     counts = []
#     for (chain, op_id) in zip(chains, op_ids):
#         idx = 0
#         min_val = chain[0]
#         for x, val in enumerate(chain):
#             if val < min_val:
#                 idx = x
#                 min_val = val
#         flow = chain[idx:] + chain[0:idx]
#         flow_id = op_id[idx:] + op_id[0:idx]
#         duplicate = False
#         for i, f in enumerate(flows):
#             if f == flow:
#                 duplicate = True
#                 counts[i] += 1
#         if not duplicate:
#             flows.append(flow)
#             flow_ids.append(flow_id)
#             counts.append(1)
#     # TODO: only fuse the top-k flows
#     for f, fid, count in zip(flows, flow_ids, counts):
#         print(f, count)

#     # All flows must have the same channel
#     # Ex a -> b -> c must all be on the same channel
#     for flow in flow_ids:
#         for i in range(0, 1):
#             f1 = flow[i]
#             constraints.append(chan(f1) >= 0)
#             constraints.append(chan(f1) < max_channels)
#             for j in range(1, len(flow)):
#                 f2 = flow[j]
#                 # print(f"{f1}=={f2}")
#                 constraints.append(chan(f1) == chan(f2))
                
#     # If a flow uses one side of the same edge it needs to be on a different channel
#     # Ex a -> b -> c and a -> b -> d can't use the same channel because b can't send to c and d on the same channel
#     # However a -> b can share the same channel as the flow 1 
#     # TODO: This is a very inefficient way to do this....
#     def chain_collision(chain1, chain2):
#         for i1 in range(0, len(chain1)-2):
#                 for i2 in range(0, len(chain2)-2):
#                     if chain1[i1] == chain2[i2] and chain1[i1+1] == chain2[i2+1]  and chain1[i1+2] != chain2[i2+2]:
#                         return True
#         return False

#     for i in range(len(flows)):
#         if counts[i] > 1:
#             for j in range(i+1, len(flows)):
#                 if counts[j] > 1:
#                     chain1 = flows[i]
#                     chain2 = flows[j]
#                     if chain_collision(chain1, chain2):
#                         print(f'{i} and {j} {flow_ids[i][0]} != {flow_ids[j][0]}')
#                         constraints.append(chan(flow_ids[i][0]) != chan(flow_ids[j][0]))

#     opt = Optimize(ctx=ctx)
#     opt.add(constraints)
#     opt.minimize(max_channels)
    
#     t = threading.Thread(target=opt.check)
#     t.start()
#     t.join(1)
#     main_ctx().interrupt()
#     t.join()

#     try:
#         model = opt.model()
#     except Z3Exception:
#         # TODO: This alternate process does not guarantee that channels are contiguous
#         s = Solver(ctx=ctx)
#         s.add(constraints)
#         s.check()
#         model = s.model()
            
#     print(f'Using up to {model[max_channels].as_long()} channels')
#     print(f'Using {model[max_channels]} channels')

    # Group the operations by which channels they use
    # ops_by_channel = defaultdict(list)
    # for idx, op_set in enumerate(op_sets):
    #     ops = ops_by_channel[model[chan(idx)].as_long()]
    #     ops.extend(op_set)

    # return ops_by_channel
    
# def channel_assignment(rank_dag):
#     rank_chan = {}
#     rank_send_chan = defaultdict(set)
#     rank_recv_chan = defaultdict(set)

#     ops = []
#     for slot, op in rank_dag.operations.items():
#         if op.inst == Instruction.start:
#             for o in op.next:
#                 if o.inst == Instruction.send or o.inst == Instruction.copy:
#                     ops.append(o)

#     i = 0
#     visited = set()
#     while i < len(ops):
#         op = ops[i]
#         if (op.inst == Instruction.send or op.inst == Instruction.copy) and op not in visited:
#             visited.add(op)
#             rank = op.rank
#             chain = [rank]
#             match = op.match
#             while match is not None:
#                 chain.append(match.rank)
#                 match = match.match
#             if len(chain) > 2:
#                 print(chain)
#             ops += op.next
#         i += 1

#     i = 0
#     visited = set()
#     while i < len(ops):
#         op = ops[i]
#         if (op.inst == Instruction.send or op.inst == Instruction.copy) and op not in visited:
#             visited.add(op)
#             rank = op.rank
#             send_peer = op.send_peer()
#             recv_peer = op.recv_peer()
#             if op.channel == -1:
#             # Check if an existing channel is already there
#                 if (rank, send_peer, recv_peer) in rank_chan:
#                     channel = rank_chan[(rank, send_peer, recv_peer)]
#                 # Brute force find the first open channel
#                 else:
#                     channel = C(0)
#                     while channel in rank_send_chan[(rank, send_peer)] or channel in rank_recv_chan[(rank, recv_peer)]:
#                         print(rank, send_peer, recv_peer)
#                         print("collide",(rank, send_peer), rank_send_chan[(rank, send_peer)])
#                         print("collide", (rank, recv_peer), rank_recv_chan[(rank, recv_peer)])
#                         channel.ch += 1
#                     if send_peer != -1:
#                         rank_send_chan[(rank, send_peer)].add(channel)
#                         print("add rank_send_chan", (rank, send_peer))
#                     if recv_peer != -1:
#                         rank_recv_chan[(rank, recv_peer)].add(channel)
#                         print("add rank_recv_chan", (rank, recv_peer))
#                     rank_chan[(rank, send_peer, recv_peer)] = channel
#                     print(rank, send_peer, recv_peer, "gets", channel)
#                 op.channel = channel
#                 match = op.match
#                 while match is not None:
#                     match.channel = channel
#                     rank = match.rank
#                     send_peer = match.send_peer()
#                     recv_peer = match.recv_peer()
#                     print("match", rank, send_peer, recv_peer)
#                     if (rank, send_peer, recv_peer) in rank_chan:
#                         cc = rank_chan[(rank, send_peer, recv_peer)]
#                         assert channel == cc, f"{rank}, {send_peer}, {recv_peer}, {cc}, {channel}"
#                     # Brute force find the first open channel
#                     else:
#                         channel = C(0)
#                         while channel in rank_send_chan[(rank, send_peer)] or channel in rank_recv_chan[(rank, recv_peer)]:
#                             print(rank, send_peer, recv_peer)
#                             print("collide",(rank, send_peer), rank_send_chan[(rank, send_peer)])
#                             print("collide", (rank, recv_peer), rank_recv_chan[(rank, recv_peer)])
#                             channel.ch += 1
#                         if send_peer != -1:
#                             rank_send_chan[(rank, send_peer)].add(channel)
#                             print("add rank_send_chan", (rank, send_peer))
#                         if recv_peer != -1:
#                             rank_recv_chan[(rank, recv_peer)].add(channel)
#                             print("add rank_recv_chan", (rank, recv_peer))
#                         rank_chan[(rank, send_peer, recv_peer)] = channel
#                         print(rank, send_peer, recv_peer, "gets", channel)
#                     match = match.match
#                 ops += op.next
#         i += 1

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

def auto_assign_tbs(rank_dag):
    # channel_assignment(rank_dag)
    rank_tbids = [0] * rank_dag.num_ranks
    current_tb_step = []
    for rank_tbs in rank_dag.tbs:
        current_tb_step.append({})

    ops = []
    for slot, op in rank_dag.operations.items():
        if op.inst == Instruction.start:
            for o in op.next:
                if o.inst == Instruction.send or o.inst == Instruction.copy:
                    heapq.heappush(ops, ((o.chunk_step, o.priority, o.dst.index), o))

    visited = set()
    while len(ops) > 0:
        _, op = heapq.heappop(ops)
        if op not in visited:
            visited.add(op)
            rank = op.rank
            s = op.send_peer()
            r = op.recv_peer()
            channel = 0 if op.channel == -1 else op.channel
            if op.channel == -1:
                print(op.channel, op)
            # Get all possible TBs this can be mapped to
            tb_options = _get_tb_options(rank_dag.tbs[rank], s, r, channel, rank_tbids[rank])
            if len(tb_options) == 0: # If there are no options, create a new threadblock
                tbid = rank_tbids[rank]
                
                rank_dag.tbs[rank][tbid] = Threadblock(send=s, recv=r, channel=channel)
                # rank_tb_assignments[rank][(s,r,channel)] = tbid
                rank_tbids[rank] += 1
            else: 
                tbid = tb_options[0]
                for tbid_opt in tb_options:
                    if current_tb_step[rank][tbid_opt] < current_tb_step[rank][tbid] and _verify_tb_op_compatible(rank_dag.tbs[rank][tbid], op):
                        tbid = tbid_opt
                # if op.chunk_step < current_tb_step[rank][tbid]:
                #     tbid = rank_tbids[rank]
                #     rank_dag.tbs[rank][tbid] = Threadblock(send=s, recv=r, channel=channel)
                #     rank_tbids[rank] += 1

            tb = rank_dag.tbs[rank][tbid]
            assert _verify_tb_op_compatible(tb, op), f"Failing: Rank: {rank} Operations uses channel {op.channel}, send:{s} recv:{r} {op}\n" \
                    f"Threadblock uses send:{tb.send} recv:{tb.recv} channel:{tb.channel}"

            rank_dag.num_channels[rank] = max(rank_dag.num_channels[rank], channel+1)

            tb.ops.append(op)
            tb.send = op.dst.rank if op.is_send() else tb.send
            tb.recv = op.src.rank if op.is_recv() else tb.recv
            
            op.step = len(tb.ops)-1
            op.tb = tbid
            current_tb_step[rank][tbid] = op.chunk_step

            # For sends, add corresponding receive operation
            match = op.match
            if match is not None:
                heapq.heappush(ops, ((match.chunk_step, match.priority), match))
            # Next operations that happen on this slot
            for o in op.next:
                heapq.heappush(ops, ((o.chunk_step, o.priority), o))
            