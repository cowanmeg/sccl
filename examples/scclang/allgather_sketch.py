from sccl.language import *
from sccl.topologies import *
from sccl.language.sketch import *
from sccl.language.collectives import AllGather

# A very simple sketch of a ring all reduce algorithm where the last send of each ring 
# has an unspecified destination gpu


def allgather_ring_1concrete(size):
    topology = fully_connected(size)
    collective = AllGather(size, 1, False, "allgather")

    with SCCLSketch("allgather_ring", topology, collective, 1) as sketch:
        # Loop over each chunk's root
        for r in range(size):
            # Get the chunk at rank r, input[r]
            c = Rank(r).input(0)
            next = (r + 1) % size
            c = c.send(next, buffer=Buffer.output, index=r)
        sketch.synthesize()

def allgather_ring_last_step_missing(size):
    topology = fully_connected(size)
    collective = AllGather(size, 1, False, "allgather")

    with SCCLSketch("allgather_ring", topology, collective, 1) as sketch:
        # Loop over each chunk's root
        for r in range(size):
            # Get the chunk at rank r, input[r]
            c = Rank(r).input(0)


            next = (r + 1) % size
            while next != (r-1) % size:
                c = c.send(next, buffer=Buffer.output, index=r)
                next = (next + 1) % size
            # Removed the last step of the ring

        sketch.synthesize()

def allgather_ring_last_2steps_hole(size):
    topology = fully_connected(size)
    collective = AllGather(size, 1, False, "allgather")

    with SCCLSketch("allgather_ring", topology, collective, 1) as sketch:
        # Loop over each chunk's root
        for r in range(size):
            # Get the chunk at rank r, input[r]
            c = Rank(r).input(0)


            next = (r + 1) % size
            hops = size - 1
            while next != (r-1) % size and hops > 2:
                c = c.send(next, buffer=Buffer.output, index=r)
                next = (next + 1) % size
                hops -= 1
            # Removed the last 2 steps of the ring replaced with (next, nex+1)
            options = [(next + 1) % size, next]
            c = c.send(options, buffer=Buffer.output, index=r)
            options2 = [(next + 1) % size, next, 0]
            c.send(options2, buffer=Buffer.output, index=r)

        sketch.synthesize()

# allgather_ring_1concrete(8)
# allgather_ring_last_step_missing(8)
allgather_ring_last_2steps_hole(8)