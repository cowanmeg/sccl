# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.language import *

def const_func(x):
    def f(chunk): return x
    return f

def alternate(x, offset=0):
    def f(chunk): return chunk % x + offset
    return f


def ring_reduce_scatter(size, rank_offset=0, rank_step=1, local_chunk_size=1, chunk_offset=0, chunk_stride=1, 
    sendtbf=const_func(-1), recvtbf=const_func(-1), chanf=const_func(-1)):
    for ch in range(0, size):
        index = ch * chunk_stride * local_chunk_size + chunk_offset
        for step in range(0, size-1):
            other = chunk(((step+1+ch) % size)*rank_step +rank_offset, Buffer.output, index, local_chunk_size)
            c = chunk(((step+2+ch) % size)*rank_step+rank_offset, Buffer.output, index, local_chunk_size)
            c.reduce(other, sendtb=sendtbf(index), recvtb=recvtbf(index), ch=chanf(index))

def ring_all_gather(size, rank_offset=0, rank_step=1, local_chunk_size=1, chunk_offset=0, chunk_stride=1, 
    sendtbf=const_func(-1), recvtbf=const_func(-1), chanf=const_func(-1)):
    for ch in range(0, size):
        index = ch * chunk_stride * local_chunk_size + chunk_offset
        for step in range(0, size-1):
            c = chunk(((step+ch) % size)*rank_step + rank_offset, Buffer.output, index, local_chunk_size)
            c.copy(((step+1+ch) % size)*rank_step + rank_offset, Buffer.output, index, sendtb=sendtbf(index), recvtb=recvtbf(index), ch=chanf(index))
