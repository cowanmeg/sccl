# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.collectives import *
from sccl.language.collectives import AllReduce


def allreduce_ring(size, instances):
    topology = fully_connected(size)
    collective = AllReduce(size, size*instances, inplace=True, name="allreduce")
    with SCCLProgram("allreduce_ring_inplace", topology, collective, size*instances):
        for r in range(size):
            for ch in range(instances):
                index = ch + r*instances
                c = Rank(r).input(index)
                next = (r + 1) % size
                # Chunk travels around the ring being reduced
                while next != r:
                    c = c.reduce(next, buffer=Buffer.input, index=ch + r*instances, ch=ch)
                    next = (next + 1) % size
                
                # Send the fully reduced chunk around the ring
                while next != (r - 1) % size:
                    c = c.send(next, buffer=Buffer.input, index=ch + r*instances, ch=ch)
                    next = (next + 1) % size

        Check()
        XML()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help='number of instances')
args = parser.parse_args()

allreduce_ring(args.num_gpus, args.instances)
