# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllReduce

def allreduce_manual_schedule(num_local_gpus, num_nodes, instances, protocol):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_local_gpus, True)

    def rank(n, g):
        return n * num_local_gpus + (g % num_local_gpus)

    with SCCLProgram("allreduce_2node_a100", topology, collective, instances, protocol=protocol):

        # Ring Reduce Scatter within each node
        for n in range(num_nodes):
            for ch in range(0, num_local_gpus):
                for step in range(0, num_local_gpus-1):
                    c1 = chunk(rank(n, ch+step+1), Buffer.input, ch)
                    c = chunk(rank(n, ch+step+2), Buffer.input, ch)
                    c.reduce(c1, ch=0)
        
        # Exchange across IBs
        for ch in range(0, num_local_gpus):
            chunk(rank(0, ch), Buffer.input, ch).rexchange(rank(1, ch), Buffer.input, ch, ch=ch%2 + 2)

        # Ring All gather within each node
        for n in range(num_nodes):
            for ch in range(0, num_local_gpus):
                for step in range(0, num_local_gpus-1):
                    chunk(rank(n, ch+step), Buffer.input, ch).copy(rank(n, ch+step+1), Buffer.input, ch, ch=1)
        XML()
        Check()

def allreduce_default_schedule(num_local_gpus, num_nodes, instances, protocol):
    num_gpus = num_local_gpus * num_nodes
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_local_gpus, True)

    def rank(n, g):
        return n * num_local_gpus + (g % num_local_gpus)

    with SCCLProgram("allreduce_2node_a100", topology, collective, instances, protocol=protocol):

        # Ring Reduce Scatter within each node
        for n in range(num_nodes):
            for ch in range(0, num_local_gpus):
                for step in range(0, num_local_gpus-1):
                    c1 = chunk(rank(n, ch+step+1), Buffer.input, ch)
                    c = chunk(rank(n, ch+step+2), Buffer.input, ch)
                    c.reduce(c1)
        
        # Exchange across IBs
        for ch in range(0, num_local_gpus):
            chunk(rank(0, ch), Buffer.input, ch).rexchange(rank(1, ch), Buffer.input, ch)

        # Ring All gather within each node
        for n in range(num_nodes):
            for ch in range(0, num_local_gpus):
                for step in range(0, num_local_gpus-1):
                    chunk(rank(n, ch+step), Buffer.input, ch).copy(rank(n, ch+step+1), Buffer.input, ch)
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help='number of gpus per node')
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL128', 'LL'], help='Protocol')
parser.add_argument('--schedule', type=str, default='auto', choices=['auto', 'manual'], help='Different schedules')
args = parser.parse_args()

assert args.num_nodes == 2, "Only works for 2 nodes right now"

if args.schedule == 'manual':
    allreduce_manual_schedule(args.num_gpus, args.num_nodes, args.instances, args.protocol)
else:
    allreduce_default_schedule(args.num_gpus, args.num_nodes, args.instances, args.protocol)
