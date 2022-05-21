# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

def allreduce(num_local_gpus, num_nodes, instances, protocol):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_gpus, True)

    def rank(n, g):
        return (n % num_nodes) * num_local_gpus + (g % num_local_gpus)

    with SCCLProgram("allreduce_multinode_a100", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False):

        # Ring Reduce Scatter within each node
        local_chunk_size = num_nodes
        for n in range(num_nodes):
            for ch in range(0, num_local_gpus):
                index = ch * local_chunk_size
                for step in range(0, num_local_gpus-1):
                    other = chunk(rank(n, ch+step+1), Buffer.input, index, local_chunk_size)
                    c = chunk(rank(n, ch+step+2), Buffer.input, index, local_chunk_size)
                    c.reduce(other, ch=0)

        # Ring across nodes
        for g in range(0, num_local_gpus):
            for n in range(0, num_nodes):
                index = g * local_chunk_size + n
                # Reduce ring
                for step in range(0, num_nodes-1):
                    other = chunk(rank(n+step+1, g), Buffer.input, index)
                    c = chunk(rank(n+step+2, g), Buffer.input, index)
                    c.reduce(other)
                # Gather ring
                for step in range(0, num_nodes-1):
                    c = c.copy(rank(n+step+1, g), Buffer.input, index)


        # Ring All gather within each node
        for n in range(num_nodes):
            for ch in range(0, num_local_gpus):
                index = ch * local_chunk_size
                for step in range(0, num_local_gpus-1):
                    c = chunk(rank(n, ch+step), Buffer.input, index, local_chunk_size)
                    c = c.copy(rank(n, ch+step+1), Buffer.input, index, ch=1)
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL128', 'LL'], help='Protocol')
args = parser.parse_args()

allreduce(8, args.num_nodes, args.instances, args.protocol)
