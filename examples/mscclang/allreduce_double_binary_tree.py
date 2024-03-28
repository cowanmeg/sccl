# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce


# See https://github.com/NVIDIA/nccl/issues/545
def double_binary_tree_allreduce(num_local_gpus, num_nodes, instances, protocol):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    num_chunks = 2 * num_local_gpus # TODO: should be nics per node * 2, this assumes each local gpu has a nic.
    collective = AllReduce(num_gpus, num_chunks, True)

    def rank(n, g):
        return (n % num_nodes) * num_local_gpus + (g % num_local_gpus)

    with MSCCLProgram("double_binary_tree_allreduce", topology, collective, instances, protocol=protocol, 
         instr_fusion=False, old_format=False):

        # Within a node form a chain to each NIC [3 -> 2 -> 1 -> 0]
        # Post: Each GPU (n, g) has chunks (g*2, g*2+1) that are reduced within a node
        for n in range(num_nodes):
            for dst in range(num_local_gpus):
                c = dst * 2
                for step in range(num_local_gpus-1, 0, -1):
                    start = rank(n, dst+step)
                    stop = rank(n, dst+step-1)
                    print(f'{step} {start}->{stop}')
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
                        child = (2 ** step) - tree + i * distance
                        parent = 2**(step+1) + distance*(i//2) - tree
                        # print(f"{child}->{parent}")
                        chunk(rank(parent, dst), Buffer.input, c).reduce((chunk(rank(child, dst), Buffer.input, c)))

                # copy down tree
                for step in range(math.floor(math.log2(num_nodes))-1, -1, -1):
                    distance = 2 ** (step+1)
                    start = (2 ** step) - tree
                    # print(f"Step {step} starts at {start}")
                    for i in range(0, num_nodes//distance):
                        child = (2 ** step) - tree + i * distance
                        parent = 2**(step+1) + distance*(i//2) - tree
                        # print(f"{parent}->{child}")
                        chunk(rank(parent, dst), Buffer.input, c).copy(rank(child, dst), Buffer.input, c)

        for n in range(num_nodes):
            for dst in range(num_local_gpus):
                c = dst * 2
                for step in range(num_local_gpus-1):
                    start = rank(n, dst+step)
                    stop = rank(n, dst+step+1)
                    chunk(start, Buffer.input, c, 2).copy(stop, Buffer.input, c, 2)

        XML()
        Check()

def double_binary_tree_allreduce_split(num_local_gpus, num_nodes, instances, protocol):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    num_chunks = 2 * num_local_gpus # TODO: should be nics per node * 2, this assumes each local gpu has a nic.
    collective = AllReduce(num_gpus, num_chunks, True)

    def rank(n, g):
        return (n%num_nodes) * num_local_gpus + g % num_local_gpus

    with MSCCLProgram("double_binary_tree_allreduce_split", topology, collective, instances, protocol=protocol, 
         instr_fusion=False, old_format=False):

        # Within a node form a chain to each NIC gpu (g) for leaf gpus
        # For non-leaf gpus form a chain up to g+1
        # Post: Each GPU (n, g) has chunks (g*2, g*2+1) that are reduced within a node
        for n in range(num_nodes):
            print("Node", n)
            for dst in range(num_local_gpus):
                for tree in range(2):
                    c = dst * 2 + tree
                    e = 1 if n % 2 == tree else 0
                    print("Chunk", c, "Tree", tree)
                    for step in range(num_local_gpus-1, e, -1):
                        start = rank(n, dst+step)
                        stop = rank(n, dst+step-1)
                        print(f'{step} {start}->{stop}')
                        chunk(stop, Buffer.input, c).reduce(chunk(start, Buffer.input, c))
                    print(" ")

        # Double binary tree between NICs
        for dst in range(num_local_gpus):
            for tree in range(0, 2):
                c = dst*2 + tree
                print(f"Tree {tree} for {c}")
                # Reduce up tree
                for step in range(math.floor(math.log2(num_nodes))):
                    distance = 2 ** (step+1)
                    start = (2 ** step) - tree
                    # print(f"Step {step} starts at {start}")
                    for i in range(0, num_nodes//distance):
                        child = (2 ** step) - tree + i * distance
                        parent = (2**(step+1) + distance*(i//2) - tree) % num_nodes
                        print(f"{child},{dst}->{parent},{dst+1}")
                        chunk(rank(parent, dst+1), Buffer.input, c).reduce((chunk(rank(child, dst), Buffer.input, c)))
                    # Finish off the chain
                    for i in range(0, num_nodes//distance, distance):
                        parent = (2**(step+1) + distance*(i//2) - tree) % num_nodes
                        print(f"{parent},{dst+1}->{parent},{dst}")
                        chunk(rank(parent, dst), Buffer.input, c).reduce((chunk(rank(parent, dst+1), Buffer.input, c)))
                print("down")

                # Copy down tree
                for step in range(math.floor(math.log2(num_nodes))-1, -1, -1):
                    distance = 2 ** (step+1)
                    start = (2 ** step) - tree
                    # print(f"Step {step} starts at {start}")
                    for i in range(0, num_nodes//distance, 2):
                        parent = (2**(step+1) + distance*(i//2) - tree) % num_nodes
                        print(f"{parent},{dst}->{parent},{dst+1}")
                        chunk(rank(parent, dst), Buffer.input, c).copy(rank(parent, dst+1), Buffer.input, c)

                    for i in range(0, num_nodes//distance):
                        child = (2 ** step) - tree + i * distance
                        parent = (2**(step+1) + distance*(i//2) - tree) % num_nodes
                        print(f"{parent},{dst+1}->{child},{dst}")
                        chunk(rank(parent, dst+1), Buffer.input, c).copy(rank(child, dst), Buffer.input, c)

        for n in range(num_nodes):
            for dst in range(num_local_gpus):
                for tree in range(2):
                    c = dst * 2 + tree
                    start = 1 if n % 2 == tree else 0
                    for step in range(start, num_local_gpus-1):
                        start = rank(n, (dst+step)%num_local_gpus)
                        stop = rank(n, (dst+step+1)%num_local_gpus)
                        chunk(start, Buffer.input, c).copy(stop, Buffer.input, c)
                        print(f"{start}->{stop}")
                    print("")

        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help='number of gpus per node')
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL128', 'LL'], help='Protocol')
args = parser.parse_args()
double_binary_tree_allreduce_split(args.num_gpus, args.num_nodes, args.instances, args.protocol)