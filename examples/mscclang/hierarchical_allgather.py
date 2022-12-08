# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllGather
from ring import *

def hierarchical_allgather(num_local_gpus, num_nodes, instances, protocol, intra_ch, device, fname):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    inplace = False
    collective = AllGather(num_gpus, 1, inplace)

    with MSCCLProgram(f"hierarchical_allgather_{num_nodes}nodes_{intra_ch}ch_{instances}in", topology, 
        collective, instances, protocol=protocol, threadblock_policy=ThreadblockPolicy.manual, interleaved_replication=True, device=device):

        if not inplace:
            for g in range(num_gpus):
                chunk(g, Buffer.input, 0).copy(g, Buffer.output, g)

        # Cross node All-gather 
        # Each (n, g) gpu N chunks at [g, g+G, g+G*2, ... g+G*(N-1)]
        for g in range(num_local_gpus):
            ring_all_gather(num_nodes, rank_offset=g, rank_step=num_local_gpus, chunk_offset=g, chunk_stride=num_local_gpus, chanf=const_func(g%2)
                , sendtbf=const_func(num_nodes*intra_ch+g%2), recvtbf=const_func(num_nodes*intra_ch+g%2)
                )

        # All gather within each node
        for n in range(num_nodes):
            # Each node needs to run N local rings since there are N scattered chunks after the allreduce
            for offset in range(num_nodes): 
                ring_all_gather(num_local_gpus, rank_offset=n * num_local_gpus, 
                    chunk_offset=offset*num_local_gpus, chanf=alternate(intra_ch, offset*intra_ch)
                    , sendtbf=alternate(intra_ch, offset*intra_ch), recvtbf=alternate(intra_ch, offset*intra_ch)
                    )

        XML(fname)
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help='number of gpus per node')
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('channels', type=int, help='number of channels per intra_node ring')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL128', 'LL'], help='Protocol')
parser.add_argument('--device', type=str, default='None', choices=['A100', 'V100', 'None'], help='Target device')
parser.add_argument('--output', type=str, default=sys.stdout, help='File name to save xml. Default: print to stdout')
args = parser.parse_args()

device = get_device(args.device)

hierarchical_allgather(args.num_gpus, args.num_nodes, args.instances, args.protocol, args.channels, device, args.output)

