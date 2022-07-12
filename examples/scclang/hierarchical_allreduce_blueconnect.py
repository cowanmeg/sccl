# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from concurrent.futures import thread

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

# Blue Connect style AllReduce https://proceedings.mlsys.org/paper/2019/file/9b8619251a19057cff70779273e95aa6-Paper.pdf
# Assumes only two-level switches

def const_func(x):
    def f(index): return x
    return f

def alternate(x, offset=0):
    def f(index): return (index % x) + offset
    return f


def ring_reduce_scatter(size, rank_offset=0, rank_step=1, local_chunk_size=1, chunk_offset=0, chunk_stride=1, sendtbf=const_func(-1), recvtbf=const_func(-1), chanf=const_func(-1)):
    for ch in range(0, size):
        index = ch * chunk_stride * local_chunk_size + chunk_offset
        for step in range(0, size-1):
            other = chunk(((step+1+ch) % size)*rank_step +rank_offset, Buffer.input, index, local_chunk_size)
            c = chunk(((step+2+ch) % size)*rank_step+rank_offset, Buffer.input, index, local_chunk_size)
            c.reduce(other, sendtb=sendtbf(index), recvtb=recvtbf(index), ch=chanf(index))

def ring_all_gather(size, rank_offset=0, rank_step=1, local_chunk_size=1, chunk_offset=0, chunk_stride=1, sendtbf=const_func(-1), recvtbf=const_func(-1), chanf=const_func(-1)):
    for ch in range(0, size):
        index = ch * chunk_stride * local_chunk_size + chunk_offset
        for step in range(0, size-1):
            c = chunk(((step+ch) % size)*rank_step + rank_offset, Buffer.input, index, local_chunk_size)
            c.copy(((step+1+ch) % size)*rank_step + rank_offset, Buffer.input, index, sendtb=sendtbf(index), recvtb=recvtbf(index), ch=chanf(index))


def blueconnect_allreduce_v2(num_local_gpus, num_nodes, instances, protocol, schedule):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_gpus, True)


    with SCCLProgram("allreduce_hierarchical_v2", topology, collective, instances, protocol=protocol, 
        interleaved_replication=True, instr_fusion=True):

        local_chunk_size = num_nodes
        if schedule == 'auto':
            for n in range(num_nodes):
                ring_reduce_scatter(num_local_gpus, rank_offset=n * num_local_gpus, local_chunk_size=num_nodes)

            # Cross node Reduce-Scatter
            for g in range(num_local_gpus):
                ring_reduce_scatter(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes)

            # Cross node All-gather
            for g in range(num_local_gpus):
                ring_all_gather(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes)


            # All gather within each node
            for n in range(num_nodes):
                ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, local_chunk_size=num_nodes)


        else:
            for n in range(num_nodes):
                ring_reduce_scatter(num_local_gpus, rank_offset=n * num_local_gpus, local_chunk_size=num_nodes, chan=0)

            # Cross node Reduce-Scatter
            for g in range(num_local_gpus):
                ring_reduce_scatter(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, chan=g%2+2)

            # Cross node All-gather
            for g in range(num_local_gpus):
                ring_all_gather(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, chan=g%2+2)


            # # All gather within each node
            for n in range(num_nodes):
                ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, local_chunk_size=num_nodes, chan=1)

        XML()
        Check()

def blueconnect_allreduce_v1(num_local_gpus, num_nodes, instances, protocol, schedule):
    def fuse_critical_path(instr_dag):
        def critical_chunk(rank):
            n = rank // num_local_gpus
            g = rank % num_local_gpus
            return ((g-1) % num_local_gpus) * num_nodes + ((n-1) % num_nodes)
        
        for slot, op in instr_dag.operations.items():
            rank, _, index = slot
            critical_index = critical_chunk(rank)
            print(slot, op.priority, critical_index)
            # Maximally fuse the critical path.
            # TODO: May fold-over itself?
            if index == critical_index:
                op = op.next[0]
                print("Critical OP", op, index, 0)
                f = {}
                while op is not None:
                    op.channel = index
                    op.tb = index
                    if op.recv_match:
                        if op.rank in f and f[op.rank] != op.recv_match.rank:
                            print(op.rank, "collides", f[op.rank], op.recv_match.rank)
                        else:
                            f[op.rank] = op.recv_match.rank
                            print(op.rank, "sends to", f[op.rank])
                    print("  ", op.recv_match)
                    op = op.recv_match
            # For every other chunk, fuse as much as possible
            else:
                op = op.next[0]
                parent_op = op
                print("", op, index, index+1)

                while op is not None:
                    op.channel = index
                    op.tb = index
                    print("  ", op.recv_match)
                    op = op.recv_match
                op = parent_op
                while op is not None:
                    # if len(op.next) > 1:
                    #     print("FFFFFF")
                    op.channel = index
                    op.tb = index
                    op = op.next[0] if len(op.next) > 0 else None
                

    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_gpus, True)

    threadblock_policy = ThreadblockPolicy.auto if schedule == 'auto' else ThreadblockPolicy.manual
    with SCCLProgram("blueconnect", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False, threadblock_policy=threadblock_policy, DAG_preprocess_func=fuse_critical_path):

        local_chunk_size = num_nodes
        if schedule == 'distribute':
            channels = num_gpus
            for n in range(num_nodes):
                for offset in range(num_nodes):
                    ring_reduce_scatter(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=offset, 
                        chunk_stride=num_nodes, chanf=alternate(channels), sendtbf=alternate(channels), recvtbf=alternate(channels))

            # Cross node Reduce-Scatter
            for g in range(num_local_gpus):
                ring_reduce_scatter(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, 
                    chanf=alternate(num_nodes), sendtbf=alternate(num_nodes, channels), recvtbf=alternate(num_nodes, channels))

            # Cross node All-gather
            for g in range(num_local_gpus):
                ring_all_gather(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, 
                    chanf=alternate(num_nodes), sendtbf=alternate(num_nodes, channels), recvtbf=alternate(num_nodes, channels))


            # All gather within each node
            for n in range(num_nodes):
                for offset in range(num_nodes):
                    ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=offset, 
                        chunk_stride=num_nodes, chanf=alternate(channels), sendtbf=alternate(channels), recvtbf=alternate(channels))
           
        elif schedule == 'const':
            # Reduce Scatter within each node
            for n in range(num_nodes):
                for offset in range(num_nodes):
                    ring_reduce_scatter(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=offset, chunk_stride=num_nodes, 
                    chanf=const_func(offset), sendtbf=const_func(offset), recvtbf=const_func(offset))

            # Cross node Reduce-Scatter
            for g in range(num_local_gpus):
                ring_reduce_scatter(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, 
                chanf=const_func(g%2), sendtbf=const_func(g+num_nodes*2), recvtbf=const_func(g+num_nodes*2))

            # Cross node All-gather
            for g in range(num_local_gpus):
                ring_all_gather(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, 
                    chanf=const_func(g%2), sendtbf=const_func(g+num_nodes*2), recvtbf=const_func(g+num_nodes*2))


            # All gather within each node
            # For large sizes, keep the allgather in separate channel/tb for better tiling overlap. 
            for n in range(num_nodes):
                for offset in range(num_nodes):
                    ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=offset, chunk_stride=num_nodes, 
                    chanf=const_func(offset+num_nodes), sendtbf=const_func(offset+num_nodes), recvtbf=const_func(offset+num_nodes))

        else: # auto
            # Reduce Scatter within each node
            for n in range(num_nodes):
                for offset in range(num_nodes):
                    ring_reduce_scatter(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=offset, chunk_stride=num_nodes)

            # Cross node Reduce-Scatter
            for g in range(num_local_gpus):
                ring_reduce_scatter(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes)

            # Cross node All-gather
            for g in range(num_local_gpus):
                ring_all_gather(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes)


            # All gather within each node
            for n in range(num_nodes):
                for offset in range(num_nodes):
                    ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=offset,  chunk_stride=num_nodes)

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
parser.add_argument('num_gpus', type=int, help='number of gpus per node')
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--version',type=str, default='v1', choices=['v1', 'v2'], help='v1=count 1 v2 = count2')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL128', 'LL'], help='Protocol')
parser.add_argument('--schedule', type=str, default='const', choices=['distribute', 'const', 'auto'], help='Scheduling')

args = parser.parse_args()

if args.version == 'v1':
    blueconnect_allreduce_v1(args.num_gpus, args.num_nodes, args.instances, args.protocol, args.schedule)
elif args.version == 'v2':
    blueconnect_allreduce_v2(args.num_gpus, args.num_nodes, args.instances, args.protocol, args.schedule)