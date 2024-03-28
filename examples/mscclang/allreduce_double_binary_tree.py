# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def double_binary_tree_allreduce(num_local_gpus, num_nodes, instances, protocol):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    num_chunks = 2 * num_local_gpus # TODO: should be nics per node * 2, this assumes each local gpu has a nic.
    collective = AllReduce(num_gpus, num_chunks, True)

    def rank(n, r):
        return n * num_local_gpus + r

    with MSCCLProgram("double_binary_tree_allreduce", topology, collective, instances, protocol=protocol, 
         instr_fusion=False, old_format=False):

        # Within a node form a chain to each NIC
        # Post: Each GPU (n, g) has chunks (g*2, g*2+1) that are reduced within a node
        for n in range(num_nodes):
            for dst in range(num_local_gpus):
                c = dst * 2
                for step in range(num_local_gpus-1):
                    start = rank(n, (dst+step+1)%num_local_gpus)
                    stop = rank(n, (dst+step+2)%num_local_gpus)
                    chunk(stop, Buffer.input, c, 2).reduce(chunk(start, Buffer.input, c, 2))

        # Double binary tree between NICs
        for dst in range(num_local_gpus):
            for tree in range(0, 2):
                # print(f"Tree {tree}")
                c = dst*2 + tree
                # Reduce up tree
                for step in range(math.floor(math.log2(num_nodes))):
                    distance = 2 ** (step+1)
                    start = (2 ** step) - tree
                    # print(f"Step {step} starts at {start}")
                    for i in range(0, num_nodes//distance):
                        leaf = (2 ** step) - tree + i * distance
                        root = (2**(step+1) + distance*(i//2) - tree) % num_nodes
                        # print(f"{leaf}->{root}")
                        chunk(rank(root, dst), Buffer.input, c).reduce((chunk(rank(leaf, dst), Buffer.input, c)))

                # Reduce down tree
                for step in range(math.floor(math.log2(num_nodes))-1, -1, -1):
                    distance = 2 ** (step+1)
                    start = (2 ** step) - tree
                    # print(f"Step {step} starts at {start}")
                    for i in range(0, num_nodes//distance):
                        leaf = (2 ** step) - tree + i * distance
                        root = (2**(step+1) + distance*(i//2) - tree) % num_nodes
                        # print(f"{root}->{leaf}")
                        chunk(rank(root, dst), Buffer.input, c).copy(rank(leaf, dst), Buffer.input, c)

        for n in range(num_nodes):
            for dst in range(num_local_gpus):
                c = dst * 2
                for step in range(num_local_gpus-1):
                    start = rank(n, (dst+step)%num_local_gpus)
                    stop = rank(n, (dst+step+1)%num_local_gpus)
                    chunk(start, Buffer.input, c, 2).copy(stop, Buffer.input, c, 2)
                #     print(f"{start}->{stop}")
                # print("")

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help='number of gpus per node')
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL128', 'LL'], help='Protocol')
args = parser.parse_args()
double_binary_tree_allreduce(args.num_gpus, args.num_nodes, args.instances, args.protocol)