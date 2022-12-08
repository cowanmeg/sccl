# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce
from ring import *

# Blue Connect style AllReduce https://proceedings.mlsys.org/paper/2019/file/9b8619251a19057cff70779273e95aa6-Paper.pdf
# Assumes only two-level switches

def hierarchical_allreduce(num_local_gpus, num_nodes, instances, protocol, schedule, device, fname):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_gpus, True)

    threadblock_policy = ThreadblockPolicy.auto if schedule == 'auto' else ThreadblockPolicy.manual
    with MSCCLProgram("hierarchical_allreduce", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False, threadblock_policy=threadblock_policy, device=device):

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
                    chanf=alternate(channels), sendtbf=alternate(channels, channels), recvtbf=alternate(channels, channels))

            # Cross node All-gather
            for g in range(num_local_gpus):
                ring_all_gather(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g*num_nodes, 
                    chanf=alternate(channels), sendtbf=alternate(channels, channels), recvtbf=alternate(channels, channels))


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
            for n in range(num_nodes):
                for offset in range(num_nodes):
                    ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, chunk_offset=offset, 
                        chunk_stride=num_nodes, chanf=const_func(offset+num_nodes), sendtbf=const_func(offset+num_nodes), recvtbf=const_func(offset+num_nodes))

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

        XML(fname)
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help='number of gpus per node')
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL128', 'LL'], help='Protocol')
parser.add_argument('--schedule', type=str, default='const', choices=['distribute', 'const', 'auto'], help='Scheduling')
parser.add_argument('--device', type=str, default='None', choices=['A100', 'V100', 'None'], help='Target device')
parser.add_argument('--output', type=str, default=sys.stdout, help='File name to save xml. Default: print to stdout')
args = parser.parse_args()

device = get_device(args.device)
hierarchical_allreduce(args.num_gpus, args.num_nodes, args.instances, args.protocol, args.schedule, device, args.output)
