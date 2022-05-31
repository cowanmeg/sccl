# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

# Blue Connect style AllReduce https://proceedings.mlsys.org/paper/2019/file/9b8619251a19057cff70779273e95aa6-Paper.pdf
# Assumes only two-level switches

def ring_reduce_scatter(size, rank_offset=0, rank_step=1, local_chunk_size=1, chunk_offset=0, chunk_stride=1, chan=-1):
    for ch in range(0, size):
        index = ch * chunk_stride * local_chunk_size + chunk_offset
        for step in range(0, size-1, rank_step):
            other = chunk(((step+1+ch) % size)*rank_step +rank_offset, Buffer.input, index, local_chunk_size)
            c = chunk(((step+2+ch) % size)*rank_step+rank_offset, Buffer.input, index, local_chunk_size)
            c.reduce(other, ch=chan)

def ring_all_gather(size, rank_offset=0, rank_step=1, local_chunk_size=1, chunk_offset=0, chunk_stride=1, chan=-1):
    for ch in range(0, size):
        index = ch * chunk_stride * local_chunk_size + chunk_offset
        for step in range(0, size-1):
            c = chunk(((step+ch) % size)*rank_step + rank_offset, Buffer.input, index, local_chunk_size)
            c.copy(((step+1+ch) % size)*rank_step + rank_offset, Buffer.input, index, ch=chan)

def blueconnect_allreduce(num_local_gpus, num_nodes, instances, protocol):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_gpus, True)


    with SCCLProgram("allreduce_hierarchical", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False):

        # Reduce Scatter within each node
        local_chunk_size = num_nodes
        for n in range(num_nodes):
            ring_reduce_scatter(num_local_gpus, rank_offset=n * num_local_gpus, local_chunk_size=num_nodes, chan=0)

        # Cross node Reduce-Scatter
        for g in range(num_local_gpus):
            ring_reduce_scatter(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, chan=1)

        # Cross node All-gather
        for g in range(num_local_gpus):
            ring_all_gather(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, chan=2)


        # All gather within each node
        for n in range(num_nodes):
            ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, local_chunk_size=num_nodes, chan=2)

        XML()
        Check()

def blueconnect_allreduce_v2(num_local_gpus, num_nodes, instances, protocol):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_gpus, True)


    with SCCLProgram("blueconnect", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False):

        # Reduce Scatter within each node
        local_chunk_size = num_nodes
        for n in range(num_nodes):
            ring_reduce_scatter(num_local_gpus, rank_offset=n * num_local_gpus, chunk_stride=2, chan=0)
        for n in range(num_nodes):
            ring_reduce_scatter(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=1, chunk_stride=2, chan=1)

        # Cross node Reduce-Scatter
        for g in range(num_local_gpus):
            ring_reduce_scatter(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, chan=2)

        # Cross node All-gather
        for g in range(num_local_gpus):
            ring_all_gather(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, chan=3)


        # All gather within each node
        for n in range(num_nodes):
            ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, chunk_stride=2, chan=0)
        for n in range(num_nodes):
            ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=1, chunk_stride=2, chan=1)

        XML()
        Check()

# https://arxiv.org/pdf/1807.11205.pdf
def jia_allreduce(num_local_gpus, num_nodes, instances, protocol):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_nodes, True)


    with SCCLProgram("allreduce_hierarchical", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False):

        local_chunk_size = num_nodes
        # Reduce within each node onto local gpu 0 (master learner)
        for n in range(num_nodes):
            for g in range(1, num_local_gpus):
                c = chunk(num_local_gpus*n+g, Buffer.input, 0, local_chunk_size)
                chunk(num_local_gpus*n, Buffer.input, 0, local_chunk_size).reduce(c)

        # AllReduce among master learners
        # Cross node Reduce-Scatter
        ring_reduce_scatter(num_nodes, rank_step=num_local_gpus)
        # Cross node All-gather
        ring_all_gather(num_nodes, rank_step=num_local_gpus)

        # # Broadcast within each node
        for n in range(num_nodes):
            # chunk on the master learner
            c = chunk(num_local_gpus*n, Buffer.input, 0, local_chunk_size)
            for g in range(1, num_local_gpus):
                c.copy(num_local_gpus*n+g, Buffer.input, 0)

        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL128', 'LL'], help='Protocol')
args = parser.parse_args()

blueconnect_allreduce_v2(8, args.num_nodes, args.instances, args.protocol)
