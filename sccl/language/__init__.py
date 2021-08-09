# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from dataclasses import dataclass
from enum import Enum
from sccl.language.ir import *

_current_program = None
def _curr():
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program

class SCCLProgram:
    def __init__(self, name, collective, topo):
        self.name = name
        self.collective = collective
        self.topo = topo
        # Initialize the chunks on each rank according to the precondition
        self.ranks = []
        # TODO: Clean this up - not using the collective
        num_chunks = topo.num_nodes()
        for r in collective.ranks():
            input_chunks = [None] * num_chunks
            output_chunks = [None] * num_chunks
            scratch_chunks = [None] * num_chunks * num_chunks
            # for c in collective.chunks():
            #     if collective.precondition(r, c):
            #         input_chunks[c] = Ref(Buffer.input, c, 1, self, r, [])
            chunks = {Buffer.input : input_chunks, 
                      Buffer.output : output_chunks, 
                      Buffer.scratch : scratch_chunks}
            self.ranks.append(Process(self, r, chunks))

    def rank(self, rank):
        return self.ranks[rank]

    # Checks that all chunks that should be on each rank
    # are present in the output buffer.
    def check(self):
        correct = True
        for r in self.collective.ranks():
            output_chunks = self.ranks[r].chunks[Buffer.output]
            for c in self.collective.chunks():
                if self.collective.postcondition(r, c) and output_chunks[c] is None:
                    print(f'Rank {r} chunk {c} is missing')
                    correct = False
        return correct

    def lower(self):
        gpu_prgms = [rank.lower() for rank in self.ranks]
        return Program(self.name, gpu_prgms)

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

def Rank(index):
    return _curr().rank(index)

def XML():
   print(ir_to_xml(_curr().lower()))

def Check():
    return _curr().check()


class Process:
    def __init__(self, prog, rank, chunks):
        self.prog = prog
        self.rank = rank
        self.chunks = chunks
        self.tbs = {}
    
    def input(self, index):
        c = Ref(Buffer.input, index, 1, self.prog, self.rank, [])
        self.chunks[Buffer.input][index] = c
        return c

    def _add_send(self, tbid, step, ch, op):
        # print(f'Send {op.dst.index} from {op.src.rank} to {op.dst.rank} {tbid} {step}')
        assert(op.inst == Instruction.send)
        sendto = op.dst.rank
        if tbid not in self.tbs:
            self.tbs[tbid] = Threadblock(ch, send=sendto, ops={step: op})
        else:
            tb = self.tbs[tbid]
            assert (tb.send == -1 or tb.send == sendto), \
                f'Rank {self.rank}: Threadblock {tbid} is already set to send to {tb.send}, trying to send to {sendto}'
            tb.send = sendto
            assert step not in tb.ops, f'Step {step} already in rank {self.rank} tbid {tbid}'
            tb.ops[step] = op

    def _add_recv(self, tbid, step, ch, op):
        assert(op.inst == Instruction.recv)
        recvd_chunk = op.dst
        self.chunks[recvd_chunk.buffer][recvd_chunk.index] = recvd_chunk
        # print(f"{self.rank} adds chunk to index {recvd_chunk.index}")
        receivefrom = op.src.rank
        if tbid not in self.tbs:
            self.tbs[tbid] = Threadblock(ch, recv=receivefrom, ops={step: op})
        else:
            tb = self.tbs[tbid]
            assert (tb.recv == -1 or tb.recv == receivefrom), \
                   f'Rank {self.rank}: Threadblock {tbid} is already set to receive from {tb.recv}, trying to receive from {receivefrom}'
            tb.recv = receivefrom
            assert step not in tb.ops, f'Step {step} in rank {self.rank} tbid {tbid}'
            tb.ops[step] = op

    def _add_copy(self, tbid, step, ch, op):
        assert(op.inst == Instruction.copy)
        self.chunks[op.dst.buffer][op.dst.index] = op.dst
        if tbid not in self.tbs:
            self.tbs[tbid] = Threadblock(ch, ops={step: op})
        else:
            tb = self.tbs[tbid]
            tb.ops[step] = op

    def lower(self):
        for tb in self.tbs.values():
            tb.ops = [v for k,v in sorted(tb.ops.items())]
        return Gpu(self.rank, self.tbs.values())

@dataclass
class Ref(ChunkRef):
    prog: SCCLProgram
    rank: int
    creator: list

    def _get_ref(self, dst, buffer, index):
        index = self.index if index == -1 else index
        return Ref(buffer, index, self.size, self.prog, dst, [])

    def split(self, num):
        assert (self.size % num == 0), 'Trying to split a chunk of {self.size} elements into {num} parts'
        chunks = [None] * num
        size = self.size // num
        for i in range(num):
            index = self.index + i * size
            chunks[i] = Ref(self.buffer, index, size, self.prog, self.rank, self.creator)
        return chunks

    def concatenate(self, other):
        assert (self.rank == other.rank), f'Trying to concatenate chunks on ranks {self.rank} and {other.rank}'
        assert (self.buffer == other.buffer), f'Trying to concatenate chunks in {self.buffer} and {other.buffer}'
        if self.index < other.index:
            first = self
            second = other
        else:
            first = other
            second = self
        # TODO: Check somewhere that all chunks are valid before sending
        # Merge the creators
        return  Ref(self.buffer, first.index, first.size + second.size, self.prog, self.rank, self.creator + other.creator)
        

    def send(self, dst, buffer=Buffer.output, index=-1, step=-1, sendtb=-1, recvtb=-1, ch=0):
        # Local copy
        if dst == self.rank:
            return self._copy(buffer, index, step, sendtb, ch)
        # Direct send
        sendtb = dst if sendtb == -1 else sendtb
        recvtb = self.rank if recvtb == -1 else recvtb
        dstchunk = self._get_ref(dst, buffer, index)
        self.prog.ranks[self.rank]._add_send(sendtb, step, ch, Op(Instruction.send, self, dstchunk, self.creator))
        receiveInstr = Op(Instruction.recv, self, dstchunk, [])
        self.prog.ranks[dst]._add_recv(recvtb, step, ch, receiveInstr)
        dstchunk.creator.append(receiveInstr)
        return dstchunk
    
    # def wait(self, steps):
    #     # TODO: fix this - I don't think we need this anymore?
    #     future = Ref(self.prog, self.proc, )
    #     self.prog._add_op(TransferOp(OpCode.Wait, self, future))
    #     return future

    def _copy(self, buffer=Buffer.output, index=-1, step=-1, tb=-1, ch=0):
        dstchunk = self._get_ref(self.rank, buffer, index)
        self.prog.ranks[self.rank]._add_copy(tb, step, ch, Op(Instruction.copy, self, dstchunk, self.creator))
        return dstchunk

    def reduce(self, other):
        # TODO: do something
        return self