import argparse

from sccl.language import *
from sccl.topologies import *
from sccl.language.collectives import AllToAll


def alltoall_hierarchical(num_nodes, gpus_per_node):
    num_ranks = num_nodes * gpus_per_node

    for n1 in range(num_nodes):
        for r in range(1,num_nodes):
            n2 = (n1 + r) % num_nodes
            # print(f"r {r} n1 {n1} n2 {n2}")

            # Gather all local chunks for the node neighbor
            for g1 in range(gpus_per_node):
                rank1 = n1 * gpus_per_node + g1

                for g2 in range(gpus_per_node):
                    rank2 = n1 * gpus_per_node + g2
                    # chunk to send: g2 on n2
                    index = n2 * gpus_per_node + g2 
                    c = chunk(rank1, Buffer.input, index)
                    c = c.send(rank2, f'send_{n2}')

        for r in range(1,num_nodes):
            n2 = (n1 + r) % num_nodes
            # IB send
            for g1 in range(gpus_per_node):
                rank = n1 * gpus_per_node + g1
                ib_peer = n2 * gpus_per_node + g1
                c = chunk(rank, f'send_{n2}', 0, 8)
                c = c.send(ib_peer, Buffer.output, c.get_dst_index(), ch=((n1+n2) % 8)*2+(rank%2)+2)

        
    # Handle local chunks within a node
    for rank in range(num_ranks):
        for g in range(gpus_per_node):
            index = (rank // gpus_per_node) * gpus_per_node + g
            c = chunk(rank, Buffer.input, index)
            c.send(c.get_dst_rank(), Buffer.output, c.get_dst_index())


