# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from lxml import etree as ET
from collections import defaultdict
from dataclasses import dataclass, field, replace
import math
import threading, queue
from enum import Enum
from z3 import *
from sccl.language.ir import *
from sccl.language import *
import sccl.language.collectives as lang_collectives

@dataclass
class CopyOp:
    src_buf: Buffer
    src_off: int
    dst_buf: Buffer
    dst_off: int
    cnt: int

# Poor hack
is_reduce = False 

def _analyze_liveness(gpus, algorithm):
    # Initialize liveness intervals for buffers on each GPU
    input_livenesses = {rank: [[(-1,-1)] for _ in range(gpu.input_chunks)] for rank, gpu in gpus.items()}
    output_livenesses = {rank: [[(math.inf,math.inf)] for _ in range(gpu.output_chunks)] for rank, gpu in gpus.items()}
    scratch_livenesses = {rank: [[(math.inf,-1)] for addr, idx in gpu.scratch.items()] for rank, gpu in gpus.items()}

    # For copies reserve the index in the output buffer from the very beginning
    for rank, gpu in gpus.items():
        for copy in gpu.copies:
            output_livenesses[rank][copy.output_offset] = [(-1,math.inf)]

    def update_liveness(rank, addr, step_idx):
        gpu = gpus[rank]
        # Find the relevant buffer and livenesses for the address
        if addr in gpu.inputs:
            buffer = gpu.inputs
            liveness = input_livenesses[rank]
        elif addr in gpu.outputs:
            buffer = gpu.outputs
            liveness = output_livenesses[rank]
        elif addr in gpu.scratch:
            buffer = gpu.scratch
            liveness = scratch_livenesses[rank]
        else:
            raise RuntimeError(f'Address {addr} not found in any buffer of rank {rank}.')
        
        # Expand the interval to include the step
        idx = buffer[addr]
        start, end = liveness[idx][0]
        liveness[idx][0] = (min(start, step_idx), max(end, step_idx))

    # For each step of the algorithm, update liveness intervals for all buffers
    for step_idx, step in enumerate(algorithm.steps):
        if len(step.sends[0]) == 5:
            for addr, src, dst, _, _ in step.sends:
                update_liveness(src, addr, step_idx)
                update_liveness(dst, addr, step_idx)
        elif len(step.sends[0]) == 6:
            for addr, src, dst, _, _, _ in step.sends:
                update_liveness(src, addr, step_idx)
                update_liveness(dst, addr, step_idx)
        else:
            for addr, src, dst in step.sends:
                update_liveness(src, addr, step_idx)
                update_liveness(dst, addr, step_idx)
    
    return (input_livenesses, output_livenesses, scratch_livenesses)

def _remap_scratch_into_input_output(liveness, gpus, logging):
    '''
    This function solves and applies a static mapping for scratch buffer indices to input/output buffers that minimizes
    scratch buffer usage for each GPU. The solving is done per GPU using the Z3 SMT solver.
    '''
    input_livenesses, output_livenesses, scratch_livenesses = liveness

    if logging:
        print('Remapping scratch into input/output...')

    def conflict(b1, b2):
        # Check if any of the intervals in lists b1 and b2 overlap
        return any(s1 <= e2 and s2 <= e1 for s1, e1 in b1 for s2, e2 in b2)

    print('Optimizing scratch mapping on all GPUs: ', end='', flush=True)
    # Handle each GPU separately
    for rank, gpu in gpus.items():
        ctx = Context()
        s = Solver(ctx=ctx)

        def remap(idx):
            # Choose for each scratch index a new index in one of the buffers
            # The index space has the input buffer from 0 to input_chunks-1,
            # the output buffer from input_chunks to output_chunks-1,
            # and the scratch buffer for any indices past that.
            return Int(f'{idx}_remap', ctx=ctx)

        # This variable limits the maximum index, in effect the size of the scratch buffer
        idx_end = Int(f'idx_end', ctx=ctx)

        for scratch_idx, scratch_liveness in enumerate(scratch_livenesses[rank]):
            # Block any input indices that conflict with the scratch index
            for input_idx, liveness in enumerate(input_livenesses[rank]):
                if conflict(scratch_liveness, liveness):
                    s.add(remap(scratch_idx) != input_idx)
            # Block any output indices that conflict with the scratch index
            for output_idx, liveness in enumerate(output_livenesses[rank]):
                if conflict(scratch_liveness, liveness):
                    s.add(remap(scratch_idx) != output_idx + gpu.input_chunks)
            # Block remapping conflicting scratch indices to the same input/output indices
            for other_idx, liveness in enumerate(scratch_livenesses[rank]):
                if other_idx != scratch_idx and conflict(liveness, scratch_liveness):
                    s.add(remap(scratch_idx) != remap(other_idx))
            # Require all indices to fit in the allowed buffer space
            s.add(remap(scratch_idx) >= 0)
            s.add(remap(scratch_idx) < idx_end)

        no_memory = gpu.input_chunks + gpu.output_chunks

        q = queue.Queue()
        def optimize(q):
            # Iterate the memory limit down to find a mapping that minimizes scratch usage
            for memory in range(no_memory + gpu.scratch_size(), no_memory - 1, -1):
                if s.check(idx_end == memory) == sat:
                    # Remember the model for the best solution
                    try:
                        m = s.model()
                        new_idxs = {addr: m[remap(old_idx)].as_long() for addr, old_idx in gpu.scratch.items()}
                        q.put(new_idxs)
                    except Z3Exception:
                        # This can happen when the solver is interrupted
                        return
                else:
                    return
        t = threading.Thread(target=optimize, args=(q,))
        t.start()
        t.join(1)
        ctx.interrupt()

        new_idxs = None
        while not q.empty():
            new_idxs = q.get()
        
        if new_idxs != None:
            print('.', end='', flush=True)
            # Apply the model to remap the scratch indices
            new_scratch = {}
            new_scratch_livenesses = [[] for addr, idx in gpu.scratch.items()]
            for addr, old_idx in gpu.scratch.items():
                new_idx = new_idxs[addr]
                # Figure out which buffer the index is in
                if new_idx < gpu.input_chunks:
                    tgt_buffer = gpu.inputs
                    tgt_idx = new_idx
                    tgt_liveness = input_livenesses[rank][tgt_idx]
                elif new_idx < gpu.input_chunks + gpu.output_chunks:
                    tgt_buffer = gpu.outputs
                    tgt_idx = new_idx - gpu.input_chunks
                    tgt_liveness = output_livenesses[rank][tgt_idx]
                else:
                    tgt_buffer = new_scratch
                    tgt_idx = new_idx - gpu.input_chunks - gpu.output_chunks
                    tgt_liveness = new_scratch_livenesses[tgt_idx]

                # Check that the remapping doesn't conflict with any existing mappings
                liveness = scratch_livenesses[rank][old_idx]
                assert not conflict(tgt_liveness, liveness)
                tgt_liveness.extend(liveness)

                # Remap the scratch index to the new index in the target buffer
                tgt_buffer[addr] = tgt_idx
            gpu.scratch = new_scratch
        else:
            print('x', end='', flush=True)
    else:
        print()

    if logging:
        max_scratch_overhead = max(gpu.scratch_size() / (gpu.input_chunks + gpu.output_chunks) for gpu in gpus.values())
        print(f'Maximum scratch overhead is {max_scratch_overhead * 100:.0f}%')

def _allocate_channels_max_concurrency(op_sets, logging):
    # This function solves a coloring problem to ops to a minimal set of channels
    ctx = Context()

    def chan(idx):
        return Int(f'chan_{idx}', ctx=ctx)
    max_channels = Int('max_channels', ctx=ctx)

    constraints = []

    # Add basic constraints and find conflicting sets of operations
    conflict_groups = defaultdict(set)
    for idx, op_set in enumerate(op_sets):
        for op in op_set:
            # Two operations conflict if they use the same src-dst edge on the same step
            conflict_groups[(op.gpu, op.is_send, op.peer, op.step)].add(idx)
        constraints.append(chan(idx) >= 0)
        constraints.append(chan(idx) < max_channels)

    # Require channels within the conflict groups to be disjoint
    for grp in conflict_groups.values():
        constraints.append(Distinct([chan(idx) for idx in grp]))

    opt = Optimize(ctx=ctx)
    opt.add(constraints)
    opt.minimize(max_channels)
    
    t = threading.Thread(target=opt.check)
    t.start()
    t.join(1)
    main_ctx().interrupt()
    t.join()

    try:
        model = opt.model()
    except Z3Exception:
        # TODO: This altenate process does not guarantee that channels are contiguous
        s = Solver(ctx=ctx)
        s.add(constraints)
        s.check()
        model = s.model()
            
    if logging:
        print(f'Using up to {model[max_channels].as_long()} channels')

    # Group the operations by which channels they use
    ops_by_channel = defaultdict(list)
    for idx, op_set in enumerate(op_sets):
        ops = ops_by_channel[model[chan(idx)].as_long()]
        ops.extend(op_set)

    return ops_by_channel

def _is_relay_link(topology, src, dst):
    if "copies" in topology.name:
        num_copies = topology.name.split(",")[1].strip(")")
        copies = int(num_copies[7:])
    else:
        copies = 1
    num_local = len(topology.links) // copies
    if src // num_local != dst // num_local:
        return True
    return False

def _allocate_channels_match_topology(op_sets, topology, instances, scale_remote, logging):
    if len(topology.switches) > 0 and logging:
        print('Warning: Switches in the topology are ignored for the channel policy MatchTopology.')
    # print(topology)
    ops_by_channel = defaultdict(list)
    next_channel = defaultdict(lambda: 0)
    for op_set in op_sets:
        send = op_set[0]
        # assert send.op_type == 's'
        assert send.inst == Instruction.send
        # src = send.gpu
        # dst = send.peer
        src = send.rank
        dst = send.send_peer()
        ops_by_channel[next_channel[(src,dst)]].extend(op_set)
        link = topology.link(src,dst) * instances    
        global is_reduce
        if is_reduce and ("DGX1" in topology.name or "DGX2RFix" in topology.name):
            if link == 0:
                print(f"link {src}->{dst} was 0. Making it {topology.link(dst,src)}")
                topology.links[dst][src] = topology.links[src][dst]
                link = topology.link(src,dst)
                # topology.link(src,dst) = topology.link(dst,src)
                assert link > 0
        else:
            assert link > 0, 'Encountered send on non-existent link'
        if _is_relay_link(topology, src, dst):
            link = link * scale_remote
        next_channel[(src,dst)] = (next_channel[(src,dst)] + 1) % link

    return ops_by_channel

class ChannelPolicy(Enum):
    One = 'One'
    MaxConcurrency = 'MaxConcurrency'
    MatchTopology = 'MatchTopology'

    def __str__(self):
        return self.value

def ncclize(algorithm, remap_scratch = None, channel_policy=ChannelPolicy.MatchTopology, pretty_print = True, old_format=False, use_scratch=False, merge_contiguous=True, instances=1, scale_remote=1, combine_contig=False, add_time_deps=False, aid_IB_contig=False, prefix="", logging=False):
    '''
    Generate the XML format used by the NCCL SCCL backend.

    Sends are split into send/recv operations and grouped by the rank executing them. Within each rank operations are
    grouped under <threadblock/> tags, which handle 1) a single peer, 2) a single type of operation, and 3) at most one
    operation per each step of the algorithm. Additional threadblocks are created as necessary to meet these
    constraints.

    Each send operation is mapped from the abstract addresses used by the synthesized algorithm to offsets into three
    named buffers, "input", "output" and "scratch", based on whether the address appears in a particular rank's
    precondition, postcondition or neither. For addresses that would be in both the input and output buffers <copy/>
    tags are created to mark an initial transfer to the output buffer and only the output buffer mapping is kept.
    '''

    if algorithm.is_pipelined():
        raise ValueError('Pipelining is not supported.')

    if remap_scratch is None:
        if algorithm.instance.extra_memory != None:
            remap_scratch = True
            if logging:
                print('Turning scratch remapping on to honor the memory limit set in the instance.')
        else:
            remap_scratch = False

    # Create GPUs, their address to buffer mappings and possible copies
    gpus = {}
    for rank in algorithm.ranks():
        outputs = {}
        if rank in algorithm.output_map:
            outputs.update({ addr: idx for idx, addr in enumerate(sorted(algorithm.output_map[rank])) })
        inputs = {}
        copies = []
        if rank in algorithm.input_map:
            for idx, addr in enumerate(sorted(algorithm.input_map[rank])):
                if addr in outputs:
                    # copies.append(_Copy(idx, outputs[addr]))
                    # src_ref = ChunkRef(rank, Buffer.input, idx, 1)
                    # dst_ref = ChunkRef(rank, Buffer.output, outputs[addr], 1)
                    # copies.append(Op(Instruction.copy, rank, src_ref, dst_ref))
                    op = CopyOp(Buffer.input, idx, Buffer.output, outputs[addr], 1)
                    copies.append(op)
                    
                else:
                    inputs[addr] = idx
        gpus[rank] = Gpu(rank, [], copies, [], inputs, outputs, len(inputs), len(outputs))

        # gpus[rank] = _Gpu(copies, inputs, outputs, len(inputs) + len(copies), len(outputs))
        # tb = Threadblock(channel=0, ops=copies)
        # gpus[rank] = Gpu(rank, [], [], [], inputs, outputs, len(inputs), len(outputs))
        # gpus[rank].threadblocks.append(tb)
# 
    # Create scratch buffer mappings if necessary
    def allocate_scratch(gpu, addr):
        if not (addr in gpu.inputs or addr in gpu.outputs or addr in gpu.scratch):
            offset = len(gpu.scratch)
            gpu.scratch[addr] = offset

    # longest_relay = [0]*len(algorithm.steps)
    # for i, step in enumerate(algorithm.steps):
    #     if len(step.sends[0]) == 5:
    #         for addr, src, dst, _, _ in step.sends:
    #             if _is_relay_link(algorithm.topology,src,dst):
    #                 longest_relay[i] += 1

    if aid_IB_contig:
        # first add scratch for relay sends only
        for step in algorithm.steps:
        # for s, cnt1 in sorted(list(enumerate(longest_relay)), key=lambda x:x[1], reverse=True):
            # step = algorithm.steps[s]
            if len(step.sends[0]) == 5:
                for addr, src, dst, _, _ in step.sends:
                    if _is_relay_link(algorithm.topology,src,dst):
                        allocate_scratch(gpus[src], addr)
                        allocate_scratch(gpus[dst], addr)
            elif len(step.sends[0]) == 6:
                for addr, src, dst, _, _, _ in step.sends:
                    if _is_relay_link(algorithm.topology,src,dst):
                        allocate_scratch(gpus[src], addr)
                        allocate_scratch(gpus[dst], addr)
            else:
                for addr, src, dst in step.sends:
                    if _is_relay_link(algorithm.topology,src,dst):
                        allocate_scratch(gpus[src], addr)
                        allocate_scratch(gpus[dst], addr)

    # next add for remaining steps
    for step in algorithm.steps:
        if len(step.sends[0]) == 5:
            for addr, src, dst, _, _ in step.sends:
                allocate_scratch(gpus[src], addr)
                allocate_scratch(gpus[dst], addr)
        elif len(step.sends[0]) == 6:
            for addr, src, dst, _, _, _ in step.sends:
                allocate_scratch(gpus[src], addr)
                allocate_scratch(gpus[dst], addr)
        else:
            for addr, src, dst in step.sends:
                allocate_scratch(gpus[src], addr)
                allocate_scratch(gpus[dst], addr)

    # Analyze liveness of indices in buffers and remap scratch into input/output as possible
    if remap_scratch:
        liveness = _analyze_liveness(gpus, algorithm)
        _remap_scratch_into_input_output(liveness, gpus, logging)

    # Sort scratch mappings in an attemp to make more of them contiguous (this is of course a heuristic).
    # for gpu in gpus.values():
    #     gpu.scratch = { addr: idx for idx, addr in enumerate(sorted(gpu.scratch)) }

    def get_buffer_and_offset(gpu, addr):
        # Map an address to one of the named buffers
        # if addr in gpu.scratch:
        #     return 's', gpu.scratch[addr]
        if addr in gpu.inputs:
            return Buffer.input, gpu.inputs[addr]
        elif addr in gpu.outputs:
            return Buffer.output, gpu.outputs[addr]
        elif addr in gpu.scratch:
            return Buffer.scratch, gpu.scratch[addr]
        else:
            raise RuntimeError('Address is not mapped to a buffer')

    def make_intervals(src, dst, addrs_set):
        if len(addrs_set) == 0:
            return

        buffs_and_offs = []
        for addr in addrs_set:
            srcbuff, srcoff = get_buffer_and_offset(gpus[src], addr)
            dstbuff, dstoff = get_buffer_and_offset(gpus[dst], addr)
            buffs_and_offs.append((srcbuff, srcoff, dstbuff, dstoff))
        
        if merge_contiguous:
            # Sort sends by both buffers and offsets and merge sends into larger intervals when both the source and
            # destination are contiguous.
            buffs_and_offs.sort()
            start = prev = buffs_and_offs[0]

            def make_interval(a,b):
                cnt = b[1] - a[1] + 1
                assert cnt == b[3] - a[3] + 1, 'Source and destination count mismatch'
                return (a[0], a[1], a[2], a[3], cnt)
        
            for x in buffs_and_offs[1:]:
                if x[0] == prev[0] and x[1] == prev[1] + 1 and x[2] == prev[2] and x[3] == prev[3] + 1:
                    # Merge into previous interval if buffers match and the new offsets are at the end of the interval
                    prev = x
                else:
                    # Yield the previous interval and start a new one
                    yield make_interval(start, prev)
                    start = prev = x
            # Yield the last interval
            yield make_interval(start, prev)
        else:
            # Just yield size 1 intervals if merging is disabled
            for srcbuff, srcoff, dstbuff, dstoff in buffs_and_offs:
                yield (srcbuff, srcoff, dstbuff, dstoff, 1)    

    # Turn all steps of the algorithm into operations
    op_sets = []
    # Track the latest op that wrote to each buffer index
    writers = defaultdict(list)
    # Track all the reads since the last write to each buffer index
    readers = defaultdict(list)
    relays = defaultdict(list)
    s_relays = defaultdict(list)
    for step_idx, step in enumerate(algorithm.steps):
        # print("step:", step_idx)
        new_writers = defaultdict(list)
        new_readers = defaultdict(list)

        # Group sent addresses by edge
        grouped_sends = defaultdict(set)
        if len(step.sends[0]) == 5:
            for addr, src, dst, t, l in step.sends:
                if combine_contig:
                    grouped_sends[(src,dst)].add(addr)
                else:
                    grouped_sends[(src,dst,t,l)].add(addr)
        elif len(step.sends[0]) == 6:
            for addr, src, dst, t, l, redop in step.sends:
                if combine_contig:
                    grouped_sends[(src,dst)].add(addr)
                else:
                    grouped_sends[(src,dst,t,l,redop)].add(addr)
        else:
            for addr, src, dst in step.sends:
                grouped_sends[(src,dst)].add(addr)

        # Combine sends into intervals and create multiple instances if necessary
        sends = []
        if combine_contig or len(step.sends[0])<5:
            for (src, dst), addrs in grouped_sends.items():
                for src_buf, src_off, dst_buf, dst_off, cnt in make_intervals(src, dst, addrs):
                    for i in range(instances):
                        new_src_off = src_off * instances + i * cnt
                        new_dst_off = dst_off * instances + i * cnt
                        send = (src, dst, src_buf, new_src_off, dst_buf, new_dst_off, cnt)
                        sends.append(send)
        elif len(step.sends[0])==6:
            for (src,dst,t,l,redop) in sorted(grouped_sends, key=lambda x: x[2]):
                addrs = grouped_sends[(src,dst,t,l,redop)]
                for src_buf, src_off, dst_buf, dst_off, cnt in make_intervals(src, dst, addrs):
                    for i in range(instances):
                        new_src_off = src_off * instances + i * cnt
                        new_dst_off = dst_off * instances + i * cnt
                        send = (src, dst, src_buf, new_src_off, dst_buf, new_dst_off, cnt, redop)
                        sends.append(send)
        else:
            for (src, dst,t,l) in sorted(grouped_sends, key=lambda x: x[2]):
                addrs = grouped_sends[(src,dst,t,l)]
                for src_buf, src_off, dst_buf, dst_off, cnt in make_intervals(src, dst, addrs):
                    for i in range(instances):
                        new_src_off = src_off * instances + i * cnt
                        new_dst_off = dst_off * instances + i * cnt
                        send = (src, dst, src_buf, new_src_off, dst_buf, new_dst_off, cnt)
                        sends.append(send)
        # print("sends", sends)
        # Perform dependency tracking and create _Op instances
        global is_reduce
        for send in sends:
            redop = None
            if len(send) == 7:
                src, dst, src_buf, src_off, dst_buf, dst_off, cnt = send
            else:
                src, dst, src_buf, src_off, dst_buf, dst_off, cnt, redop = send
            # read_keys = [(src,src_buf,src_off+i) for i in range(cnt)]
            # # A send must wait for the previous recv (if any) to finish
            # send_depends = list(set(d for k in read_keys for d in writers[k]))

            # write_keys = [(dst,dst_buf,dst_off+i) for i in range(cnt)]
            # # A receive must wait for both the previous recv and any previous sends to finish
            # recv_depends = list(set(d for deps in (readers, writers) for k in write_keys for d in deps[k]))
            # print(send_depends)
            # print(recv_depends)
            # TODO: What is this for?
            if add_time_deps:
                if _is_relay_link(algorithm.topology, src, dst):
                    if dst in relays:
                        d, src_last = relays[dst]
                        if src_last != src:
                            recv_depends.append(d)
                    if src in s_relays:
                        d1, dst_last = s_relays[src]
                        if dst_last != dst:
                            send_depends.append(d1)

            # send_op = _Op(src, dst, step_idx, True, 's', src_buf, src_off, dst_buf, dst_off, cnt, send_depends)
            # src_ref = ChunkRef(src, src_buf, src_off, cnt)
            # dst_ref = ChunkRef(dst, dst_buf, dst_off, cnt)
            # send_op = Op(Instruction.send, src, src_ref, dst_ref, send_depends, step=step_idx)
            # if redop is None:
            #     # recv_op = _Op(dst, src, step_idx, False, 'r', src_buf, src_off, dst_buf, dst_off, cnt, recv_depends)
            #     recv_op = Op(Instruction.recv, dst, src_ref, dst_ref, recv_depends, step=step_idx)
            # else:
            #     assert redop == 'rrc'
            #     is_reduce = True
            #     recv_op = Op(Instruction.recv_reduce_copy, dst, src_ref, dst_ref, recv_depends, step=step_idx)
            #     # recv_op = _Op(dst, src, step_idx, False, redop, src_buf, src_off, dst_buf, dst_off, cnt, recv_depends)
            # # Record the send and receive as a set of operations that must happen on the same channel
            # # if src_off == 0 or src_off == 1:
            # op_sets.append([send_op, recv_op])
            # # print(send_op, recv_op)
            if redop is None:
                op_type = 'send'
            else:
                op_type = 'reduce'
            op = (op_type, src, dst, src_buf, src_off, dst_buf, dst_off, cnt, 0)
            op_sets.append(op)

            if add_time_deps:
                if _is_relay_link(algorithm.topology, src, dst):
                    relays[dst] = (recv_op,src)
                    s_relays[src] = (send_op,dst)
            # Mark writers and readers to be added for the next step
            # for k in write_keys:
            #     new_writers[k].append(recv_op)
            # for k in read_keys:
            #     new_readers[k].append(send_op)
        # Writes cut the dependency to both previous writes and reads
        for key, deps in new_writers.items():
            if key in new_readers:
                gpu, buf, off = key
                if "phasewise" in prefix:
                    print("key", key)
                    print("readers", new_readers[key])
                    print("writers", new_writers[key])
                    dep_send_op = new_readers[key][0]
                    old_recv_op = new_writers[key][0]
                    assert old_recv_op.op_type == 'rrc'
                    deplist = old_recv_op.depends
                    deplist.append(dep_send_op)
                    new_src_ref = ChunkRef(old_recv_op.src.rank, old_recv_op.src.buffer, old_recv_op.src.index, old_recv_op.src.size)
                    new_dst_ref = ChunkRef(old_recv_op.dst.rank, old_recv_op.dst.buffer, old_recv_op.dst.index, old_recv_op.dst.size)
                    new_recv_op = Op(old_recv_op.inst, old_recv.op.rank, new_src_ref, new_dst_ref, deplist)
                    # TODO: how is this different?
                    # new_recv_op = _Op(old_recv_op.gpu, old_recv_op.peer, old_recv_op.step, False, old_recv_op.op_type, old_recv_op.src_buffer, old_recv_op.src_offset, old_recv_op.dst_buffer, old_recv_op.dst_offset, old_recv_op.cnt, deplist)
                    for i, op_set in enumerate(op_sets):
                        if op_set[1] == old_recv_op:
                            op_sets[i][1] = new_recv_op
                    new_writers[key][0] = new_recv_op
                    print(f'Encountered receive and send on the same buffer index on step {step_idx + 1} (gpu={gpu}, buf={buf}, off={off})')
                    # print('but added deps')
                else:
                    raise RuntimeError(f'Encountered receive and send on the same buffer index on step {step_idx + 1} (gpu={gpu}, buf={buf}, off={off})\nAre you running a phasewise algo? Add prefix="_phasewise"')
            writers[key] = deps
            readers[key] = []
        # Reads get added to any previous reads
        for key, deps in new_readers.items():
            readers[key].extend(deps)

    # Fixup everything to match the instanced sends when multiple instances are generated
    if instances > 1:
        for rank, gpu in gpus.items():
            # Create instances copies of the copies.
            new_copies = []
            for copy in gpu.copies:
                for i in range(instances):
                    # new_copy = _Copy(copy.input_offset * instances + i, copy.output_offset * instances + i)
                    src_ref = ChunkRef(rank, Buffer.input, copy.input_offset * instances + i, 1)
                    dst_ref = ChunkRef(rank, Buffer.output, copy.output_offset * instances + i, 1)
                    new_copy = Copy(rank, src_ref, dst_ref)
                    new_copies.append(new_copy)
            gpu.copies = new_copies

            # Multiply the other metadata with instances
            def expand_mappings(mappings):
                return { addr * instances + i: idx * instances + i for addr, idx in mappings.items() for i in range(instances) }
            gpu.inputs = expand_mappings(gpu.inputs)
            gpu.outputs = expand_mappings(gpu.outputs)
            gpu.input_chunks *= instances
            gpu.output_chunks *= instances
            gpu.scratch = expand_mappings(gpu.scratch)

    # Allocate channels and group operations by channel
    ops_by_channel = {0: [op for op_set in op_sets for op in op_set]}
    # if channel_policy == ChannelPolicy.One:
    #     ops_by_channel = {0: [op for op_set in op_sets for op in op_set]}
    # elif channel_policy == ChannelPolicy.MaxConcurrency:
    #     ops_by_channel = _allocate_channels_max_concurrency(op_sets, logging)
    # elif channel_policy == ChannelPolicy.MatchTopology:
    #     ops_by_channel = _allocate_channels_match_topology(op_sets, algorithm.topology, instances, scale_remote, logging)
    # else:
    #     assert False, 'Unhandled channel policy'

    # Group by which operations need to be in the same threadblock
    # tb_groups = defaultdict(list)
    # for chan, chan_ops in ops_by_channel.items():
    #     for op in chan_ops:
    #         tb_groups[(op.rank, op.is_send(), op.peer(), chan)].append(op)

    # tbs_by_gpu_chan = defaultdict(lambda: defaultdict(list))
    # # For each group find or create a threadblock to add them to
    # for key, grp in tb_groups.items():
    #     rank, is_send, peer, chan = key
    #     make_none = False
    #     # # uncomment to only create IB transfers
    #     # if rank//16 == peer//16:
    #     #     make_none = True
    #     #     continue
    #     tbs = tbs_by_gpu_chan[rank][chan]
    #     for tb in tbs:
    #         tb_peer = tb.send if is_send else tb.recv
    #         # An existing threadblock can be reused if:
    #         # - Either the relevant peer is not set yet or the peer is the same
    #         # - No operations already in the threadblock execute in the same step
    #         if tb_peer == -1 or tb_peer == peer:
    #             if all(not any(op1.step == op2.step for op2 in grp) for op1 in tb.ops):
    #                 break
    #     else:
    #         # No existing threadblock was suitable, so create a new one
    #         # tb = _Threadblock(chan)
    #         tb = Threadblock(channel=chan)
    #         tbs.append(tb)
    #     # Ensure the peer is set correctly
    #     if is_send:
    #         assert tb.send == -1 or tb.send == peer
    #         tb.send = peer
    #     else:
    #         assert tb.recv == -1 or tb.recv == peer
    #         tb.recv = peer
    #     # tb.steps.extend(grp)
    #     tb.ops.extend(grp)

    # for rank, tb_by_chan in tbs_by_gpu_chan.items():
    #     for _, tbs in tb_by_chan.items():
    #         for tb in tbs:
    #             gpus[rank].threadblocks.append(tb)

    protocol='Simple'
    inplace = False
    chunks = algorithm.collective.num_chunks
    co_name = algorithm.collective.runtime_name
    num_ranks = algorithm.topology.num_nodes()
    if co_name == 'allreduce':
        collective = lang_collectives.AllReduce(num_ranks, chunks, inplace)
    elif co_name == 'allgather':
        collective = lang_collectives.AllGather(num_ranks, chunks // num_ranks, inplace)
    elif co_name == 'alltoall':
        collective = lang_collectives.AllToAll(num_ranks, chunks // num_ranks, inplace)
    elif co_name == 'reduce_scatter':
        collective = lang_collectives.ReduceScatter(num_ranks, chunks, inplace)
    # program = Program('name', 'allreduce', inplace, protocol, gpus.values())
    # return ir_to_xml(program, old_format, True, pretty_print)
    instr_fusion = True
    program = SCCLProgram(algorithm.name, algorithm.topology, collective, 1, instr_fusion=instr_fusion)
    with program:
        for rank, gpu in gpus.items():
            for copy_op in gpu.precopies:
                chunk(rank, copy_op.src_buf, copy_op.src_off, copy_op.cnt).send(rank, copy_op.dst_buf, copy_op.dst_off)

        # for step_idx, sends in enumerate(op_sets):
        #     # print(step_idx)
        for op_type, src, dst, src_buf, src_off, dst_buf, dst_off, cnt, chan in op_sets:
            # print("  ", src, chan, src_buf, src_off, dst, dst_buf, dst_off, cnt)
            if op_type == 'send':
                chunk(src, src_buf, src_off, cnt).send(dst, dst_buf, dst_off, ch=chan)
            else:
                chunk(src, src_buf, src_off, cnt).reduce(dst, dst_buf, dst_off, ch=chan)

        for rank, gpu in gpus.items():
            for copy_op in gpu.postcopies:
                chunk(rank, copy_op.src_buf, copy_op.src_off, copy_op.cnt).send(rank, copy_op.dst_buf, copy_op.dst_off)

        # Add any copies from input to output that weren't already added
        for rank, gpu in gpus.items():
            for addr in gpu.inputs:
                if addr in gpu.outputs:
                    chunk(rank, Buffer.input, gpu.inputs[addr]).send(rank, Buffer.output, gpu.outputs[addr])
                    del gpu.outputs[addr]
                    
    return ir_to_xml(program.lower())
