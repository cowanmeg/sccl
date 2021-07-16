from sccl.topologies import topology
from sccl.path_encoding import *

class Heuristic:
    def __init__(self, buddies):
        self.buddies = buddies

    def add_constraints(self, collective, topology, s):
        # Enforce only 1 send/receive step between buddies
        for src, dst in self.buddies:
            for c1 in collective.chunks():
                for c2 in range(c1+1, collective.num_chunks):
                    s.add(Implies(And(send(c1, src, dst), send(c2, src, dst)),
                                    start(c1, dst) == start(c2, dst)))