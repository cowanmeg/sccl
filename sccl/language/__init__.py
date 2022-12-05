# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bdb import effective
import math
from dataclasses import dataclass
from enum import Enum
from sccl.language.ir import *
from sccl.language.passes import *
from sccl.language.tb_assignment import *
from sccl.language.chunk import *
from sccl.language.buffer import *
from sccl.language.instruction_dag import *
from sccl.language.schedule import *
from sccl.language.device import *
import sccl.collectives as collectives

max_count = 128

_parallel_id = 0
_current_program = None
def _curr():
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program

class SCCLProgram:
    def __init__(self, name, topo, collective, instances, protocol='Simple', \
            threadblock_policy=ThreadblockPolicy.auto, interleaved_replication=True,
            instr_fusion=True, check_xml=True, dependence_nop=False, device=None):
        self.name = name
        self.topo = topo
        self.collective = collective       
        self.num_ranks = topo.num_nodes()
        self.instances = instances
        self.protocol = protocol
        self.threadblock_policy = threadblock_policy
        self.interleaved_replication = interleaved_replication
        self.instr_fusion = instr_fusion
        self.check_xml = check_xml
        self.dependence_nop = dependence_nop
        self.device = Generic if device is None else device
        assert protocol == 'Simple' or protocol == 'LL' or protocol == 'LL128', \
            f'Given protocol: {protocol}. Must be either Simple, LL, LL128'
        self.run_opt = True # Runs optimization passes
        # Initialize the input buffers
        self.buffers = collective.init_buffers()
        self.instr_dag = InstructionDAG(self.num_ranks, self.buffers, self.protocol)
        self.trace = []

    def __enter__(self):
        global _current_program
        if _current_program != None:
            raise RuntimeError("There is already a SCCL Program in context")
        _current_program = self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _current_program
        if _current_program != self:
            raise RuntimeError("This program is not currently in context")
        _current_program = None

    # Tracks a send operation on the buffers
    def apply_send(self, src, src_buffer, src_index, dst, dst_buffer, dst_index, size):
        src_buffer, src_index = self.collective.get_buffer_index(src, src_buffer, src_index)
        dst_buffer, dst_index = self.collective.get_buffer_index(dst, dst_buffer, dst_index)
        sb = self.buffers[src][src_buffer]
        db = self.buffers[dst][dst_buffer]
        for i in range(size):
            db[dst_index + i] = sb[src_index + i]

    # Tracks a reduce operation on the buffers
    def apply_reduce(self, src, src_buffer, src_index, dst, dst_buffer, dst_index, size):
        src_buffer, src_index = self.collective.get_buffer_index(src, src_buffer, src_index)
        dst_buffer, dst_index = self.collective.get_buffer_index(dst, dst_buffer, dst_index)
        sb = self.buffers[src][src_buffer]
        db = self.buffers[dst][dst_buffer]
        for i in range(size):
            reduce_chunk = db[dst_index + i]
            sent_chunk = sb[src_index + i]
            db[dst_index + i] = reduce_chunk.reduce(dst, sent_chunk)

    def get_ref(self, rank, buffer, index, size):
        buffer, index = self.collective.get_buffer_index(rank, buffer, index)
        return Ref(rank, buffer, index, size, self)

    def get_chunks(self, rank, buffer, index, size=1):
        chunks = [None] * size
        for i in range(0, size):
            if self.buffers[rank][buffer] and index+i < len(self.buffers[rank][buffer]):
                chunks[i] = self.buffers[rank][buffer][index+i]
            else:
                 chunks[i] = None
        return chunks

    def check_buffer_exists(self, rank, name):
        if name not in self.buffers[rank]:
            self.buffers[rank][name] = BufferSlice(Buffer.scratch, name)

    def apply_parallelize(self):
        # Parallelize fragments of code by replicating instructions
        # TODO: This code is very messy....clean up

        # Determine the new chunk
        replicate = 1
        effective_chunk_size = 1
        for op in self.trace:
            if type(op) is ScheduleOp and op.op is ScheduleType.parallel_enter:
                replicate *= op.factor
            elif type(op) is ScheduleOp and op.op is ScheduleType.parallel_exit:
                replicate //= op.factor
            elif replicate > 1:
                if op.src.size * effective_chunk_size % replicate != 0:
                    effective_chunk_size = (op.src.size * replicate * effective_chunk_size) // math.gcd(op.src.size * replicate, effective_chunk_size)

        # Gather TB and channel data
        stb_assignment = defaultdict(set)
        rtb_assignment = defaultdict(set)
        ch_assignment = defaultdict(set)
            
        for op in self.trace:
            if type(op) is not ScheduleOp and (op.inst == ChunkInstruction.copy or op.inst == ChunkInstruction.reduce):
                sender = op.src.rank
                receiver = op.dst.rank
                if sender != receiver:
                    ch_assignment[(sender, receiver)].add(op.ch)
                if op.sendtb != -1:
                    stb_assignment[op.src.rank].add(op.sendtb)
                if op.recvtb != -1:
                    rtb_assignment[op.dst.rank].add(op.recvtb)

        ch2rep = defaultdict(dict)
        sendtb2rep = defaultdict(dict)
        recvtb2rep = defaultdict(dict)

        self.scheduled_trace = []
        replicate = 1
        for op in self.trace:
            if type(op) is ScheduleOp and op.op is ScheduleType.parallel_enter:
                replicate *= op.factor
            elif type(op) is ScheduleOp and op.op is ScheduleType.parallel_exit:
                replicate //= op.factor
            elif replicate > 1:
                sender = op.src.rank
                receiver = op.dst.rank
                if sender == receiver:
                    replicated_base_ch = 0 # Technically no channel needed for local op tb
                elif op.ch not in ch2rep[(sender, receiver)]:
                    replicated_base_ch = max(ch_assignment[(sender, receiver)]) + 1
                    ch2rep[(sender,receiver)][op.ch] = replicated_base_ch
                else:
                    replicated_base_ch = ch2rep[(sender,receiver)][op.ch]

                if op.sendtb not in sendtb2rep[sender]:
                    replicated_base_sendtb = max(stb_assignment[sender]) + 1
                    sendtb2rep[sender][op.sendtb] = replicated_base_sendtb
                else:
                    replicated_base_sendtb = sendtb2rep[sender][op.sendtb] 

                if op.recvtb not in recvtb2rep[receiver]:
                    replicated_base_recvtb = max(rtb_assignment[receiver]) + 1
                    recvtb2rep[receiver][op.recvtb] = replicated_base_recvtb
                else:
                    replicated_base_recvtb = recvtb2rep[receiver][op.recvtb]

                for i in range(replicate):
                    iop = op.copy()
                    src = op.src
                    dst = op.dst
                    isrc = Ref(src.rank, src.buffer, src.index*effective_chunk_size+i, src.size*effective_chunk_size//replicate, src.prog)
                    idst = Ref(dst.rank, dst.buffer, dst.index*effective_chunk_size+i, dst.size*effective_chunk_size//replicate, dst.prog)
                    iop.src = isrc
                    iop.dst = idst
                    if op.ch != -1:
                        iop.ch = replicated_base_ch + i if i > 0 else op.ch
                        if sender != receiver:
                            ch_assignment[(sender, receiver)].add(iop.ch)
                    if op.sendtb != -1:
                        iop.sendtb = replicated_base_sendtb + i
                        stb_assignment[sender].add(iop.sendtb)
                    if op.recvtb != -1:
                        iop.recvtb = replicated_base_recvtb + i
                        rtb_assignment[receiver].add(iop.recvtb)
                    self.scheduled_trace.append(iop)
            else:
                op.src.size *= effective_chunk_size
                op.src.index *= effective_chunk_size
                op.dst.size *= effective_chunk_size
                op.dst.index *= effective_chunk_size
                self.scheduled_trace.append(op)
        return effective_chunk_size

    def lower_instr_dag(self, effective_chunk_size):
        # Initialize chunks
        for r in range(self.num_ranks):
            for index, chunk in enumerate(self.buffers[r][Buffer.input]):
                buffer, index = self.collective.get_buffer_index(r, Buffer.input, index)
                for i in range(effective_chunk_size):
                    ref = self.get_ref(r, buffer, index*effective_chunk_size + i, 1)
                    self.instr_dag.add_start(ref)

        # Build chunk dag from the scheduled trace of ops
        for op in self.scheduled_trace:
            sender = op.src.rank
            receiver = op.dst.rank
            if op.inst == ChunkInstruction.copy:
                if sender != receiver:
                    sop = self.instr_dag.add_send(sender, op.src, op.dst, op.sendtb, op.ch)
                    rop = self.instr_dag.add_recv(receiver, op.src, op.dst, op.recvtb, op.ch, sop)
                    sop.recv_match = rop
                else:
                    self.instr_dag.add_copy(sender, op.src, op.dst, op.sendtb, op.ch)
            elif op.inst == ChunkInstruction.reduce:
                if sender != receiver:
                    sop = self.instr_dag.add_send(sender, op.src, op.dst, op.sendtb, op.ch)
                    rop = self.instr_dag.add_recv_reduce_copy(receiver, op.src, op.dst, op.recvtb, op.ch, sop)
                    sop.recv_match = rop
                else:
                    self.instr_dag.add_reduce(sender, op.src, op.dst, op.sendtb, op.ch)



    # Checks that all chunks that should be on each rank
    # are present in the output buffer.
    def prgm_check(self):
        return self.collective.check(self)

    def ir_check(self):
        # Check generated SCCL-IR for correctness - no circular dependencies, sends and receives are ordered
        check_dependency_cycles(self.instr_dag.tbs)
        check_threadblock_ordering(self.instr_dag)
        check_deadlock(self.instr_dag, self.protocol)


    def get_maxcount(self):
        maxcount = 1
        for op in self.scheduled_trace:
            if type(op) is ChunkOp:
                maxcount = max(maxcount, op.src.size)
        return maxcount

    # Lower program to XML
    def lower(self):
        effective_chunk_size = self.apply_parallelize()
        self.lower_instr_dag(effective_chunk_size)
        self.instr_dag.convert_set_list() # Pre-emptively convert sets to lists
        if self.instr_fusion:
            self.instr_dag.optimize()
        self.instr_dag._complete_metadata()
        if self.threadblock_policy == ThreadblockPolicy.manual:
            manual_assign_tbs(self.instr_dag)
        else:
            auto_assign_tbs(self.instr_dag)
        self.instr_dag.lower_pt1(self.instances)
        gpu_prgms = self.instr_dag.lower_pt2(self.instances, self.interleaved_replication)
        return Program(self.name, self.collective.name, self.collective.inplace, self.protocol, self.get_maxcount(), gpu_prgms)  

    def generate_xml(self, fname):
        ir_to_xml(self.lower(), self.device, fname=fname, dependence_nop=self.dependence_nop)
    
    def print_chunk_dag(self):
        visualize_chunk_dag(self.chunk_dag.chunk_paths)

    def print_instr_dags(self, rank):
        if rank == 0:
            for r in range(len(self.ranks)):
                visualize_instr_dag(self.instr_dags[r].operations)
        else:
            visualize_instr_dag(self.instr_dags[rank].operations)

def Print():
    _curr().print_chunk_dag()

def chunk(rank, buffer, index, size=1):
    return _curr().get_ref(rank, buffer, index, size)

def create_scratch(rank, name):
    return _curr().create_scratch(rank, name)

def XML(fname=sys.stdout):
    _curr().generate_xml(fname)

def Check():
    return _curr().prgm_check()

def CheckIR():
    return _curr().ir_check()

def InstrDAG():
    return _curr().instr_dag

class parallelize():
    def __init__(self, instances):
        global _parallel_id
        self.instances = instances
        self.id = _parallel_id
        _parallel_id += 1

    def __enter__(self):
        _curr().trace.append(ScheduleOp(ScheduleType.parallel_enter, self.instances))

    def __exit__(self, exc_type, exc_value, exc_traceback):
        _curr().trace.append(ScheduleOp(ScheduleType.parallel_exit, self.instances))

@dataclass
class Ref(ChunkRef):
    prog: SCCLProgram

    def __repr__(self):
        return f'Ref(Buffer:{self.buffer}, Index:{self.index}, Size:{self.size}, Rank:{self.rank})'

    def _end(self):
        return self.index + self.size

    def _get_chunk(self, index):
        return self.prog.buffers[self.rank][self.buffer][index]

    def split(self, num):
        assert (self.size % num == 0), f'Trying to split a chunk of {self.size} elements into {num} parts'
        chunks = [None] * num
        size = self.size // num
        for i in range(num):
            index = self.index + i * size
            chunks[i] = self.prog.get_ref(self.rank, self.buffer, index, size)
        return chunks

    def group(self, other):
        assert (self.rank == other.rank), f'Trying to concatenate chunks on ranks {self.rank} and {other.rank}'
        assert (self.buffer == other.buffer), f'Trying to concatenate chunks in {self.buffer} and {other.buffer}'
        if self.index < other.index:
            first = self
            second = other
        else:
            first = other
            second = self

        end = max(first._end(), second._end())
        return Ref(self.rank, self.buffer, first.index, end - first.index, self.prog)
        
    # Copies the chunk(s) referenced by this chunkref onto Rank dst at location (buffer, index)
    def copy(self, dst, buffer=None, index=-1, sendtb=-1, recvtb=-1, ch=-1):
        self.prog.check_buffer_exists(dst, buffer)

        # If index is not specified assume it is going to the same place in the next gpu
        if index == -1 and buffer == None:
            index = self.index
            buffer = self.buffer
        elif index == -1 and buffer is not Buffer.input and buffer is not Buffer.output:
            index = self.prog.buffers[dst][buffer].instance_size()

        # Some inplace collectives have custom logic for buffers and index (ReduceScatter, AllGather)
        buffer, index = self.prog.collective.get_buffer_index(self.rank, buffer, index)

        # Direct send
        assert (self.prog.topo.link(self.rank, dst) or dst == self.rank), f'No link from {self.rank} to {dst}'
        dst_chunkref = self.prog.get_ref(dst, buffer, index, self.size)

        # Check if we are copying the chunk to the same index (easy mistake when we are using inplace)
        if dst_chunkref == self:
            return

        # chunks = self.prog.get_chunks(self.rank, self.buffer, self.index, self.size)
        # overwritten_chunks = self.prog.get_chunks(dst, buffer, index, self.size)
        
        self.prog.apply_send(self.rank, self.buffer, self.index, dst, buffer, index, self.size)
        # Break up the copy to fit into the max_count
        if self.size > max_count:
            count = 0
            while count < self.size:
                c = min(max_count, self.size-count)
                srcref = self.prog.get_ref(self.rank, self.buffer, self.index + count, c)
                dstref = self.prog.get_ref(dst, buffer, index + count, c)
                chunkop = ChunkOp(ChunkInstruction.copy, srcref, dstref, sendtb, recvtb, ch)
                self.prog.trace.append(chunkop)
                count += c
        else:
            chunkop = ChunkOp(ChunkInstruction.copy, self, dst_chunkref, sendtb, recvtb, ch)
            self.prog.trace.append(chunkop)

        return dst_chunkref

    # Reduces the chunk(s) referenced by other_chunkref into the chunk(s) referenced by this chunkref
    def reduce(self, other_chunkref, sendtb=-1, recvtb=-1, ch=-1):
        # Receive reduce copy
        dst = self.rank
        src = other_chunkref.rank
        assert (self.prog.topo.link(src, dst) or src == dst), f'No link from {src} to {dst}'
        self.prog.apply_reduce(src, other_chunkref.buffer, other_chunkref.index, dst, self.buffer, self.index, self.size)
        # Break up reduce to fit into the max_count
        if self.size > max_count:
            c = min(max_count, self.size-count)
            srcref = self.prog.get_ref(src, other_chunkref.buffer, other_chunkref.index + count, c)
            dstref = self.prog.get_ref(dst, self.buffer, self.index + count, c)
            chunkop = ChunkOp(ChunkInstruction.reduce, srcref, dstref, sendtb, recvtb, ch)
            self.prog.trace.append(chunkop)
            count += c
        else:
            chunkop = ChunkOp(ChunkInstruction.reduce, other_chunkref, self, sendtb, recvtb, ch)
            self.prog.trace.append(chunkop)
        return self

    def get_origin_index(self, index=0):
        return self._get_chunk(index + self.index).origin_index

    def get_origin_rank(self, index=0):
        return self._get_chunk(index + self.index).origin_rank

    def get_dst_index(self, index=0):
        return self._get_chunk(index + self.index).dst_index

    def get_dst_rank(self, index=0):
        return self._get_chunk(index + self.index).dst_rank

    def print_chunk_info(self, index=0):
        print(self._get_chunk(index + self.index)) 

@dataclass
class ChunkOp():
    inst: ChunkInstruction
    src: Ref # Ref Chunk acted on
    dst: Ref # Ref Chunk created
    sendtb: int = -1
    recvtb: int = -1
    ch: int = -1

    def copy(self):
        return ChunkOp(self.inst, self.src, self.dst, self.sendtb, self.recvtb, self.ch)