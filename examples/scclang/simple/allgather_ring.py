# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllGather

def allgather_ring(size):
    topology = fully_connected(size)
    collective = AllGather(size, 1, False)
    with SCCLProgram("allgather_ring", topology, collective, 1):
        # Loop over each chunk's root
        for r in range(size):
            # Get the chunk at rank r, input[r]
            c = Rank(r).input(0)
            # Copy chunk to the output buffer
            c = c.send(r, sendtb=0, buffer=Buffer.output, index=r)

            next = (r + 1) % size
            while next != r:
                # For each rank in the ring, send the chunk to the next rank
                c = c.send(next, sendtb=1, recvtb=2, buffer=Buffer.output, index=r)
                next = (next + 1) % size
        XML()
        Check()

# def wait():
#     with SCCLProgram("scheduling", line(2)):
#         Rank(0).input(0).send(2).send(3).send(4).output()
#         # Here wait(1) is used to avoid the two send(3)s happening at the same time
#         Rank(1).input(1).send(2).wait(1).send(3).output()


allgather_ring(16)