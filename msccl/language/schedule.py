# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
import heapq
import functools
from typing import DefaultDict

from msccl.language.ir import *


class ScheduleType(Enum):
    parallel_enter = 'parallel_enter'
    parallel_exit = 'parallel_exit'

    def __str__(self):
        return self.value

@dataclass
class ScheduleOp:
    op: ScheduleType
    factor: int = 0


class Connections:
    
    def __init__(self, num_ranks):
        self.send_connections = []
        self.recv_connections = []
        for _ in range(self.num_ranks):
            self.send_connections.append(defaultdict(list))
            self.recv_connections.append(defaultdict(list))

    def create(self, instrs):
        for inst in instrs:
            if inst.is_send():
                self.send_connections[inst.rank][inst.send_peer()].append(inst)
            elif inst.is_recv():
                self.recv_connections[inst.rank][inst.recv_peer()].append(inst)

    def num_sends(self, sender, receiver):
        return len(self.send_connections[sender][receiver])


class Schedule:
    def __init__(self, num_ranks, connections):
        self.connections = connections
        # Channel and threadblock assignments for instructions that use a connection
        self.num_channels = []
        self.channel_assignment = []
        self.tb_assignment = []
        for rank in range(num_ranks):
            for peer in self.connections[rank].keys():
                self.num_channels[peer] = 1
            self.channel_assignment.append(defaultdict(list))

    def split_connection_blocked(self, rank, peer, num_channels):
        num_sends = self.connection.num_sends(rank, peer)
        if num_channels > num_sends:
            return False

        self.num_channels[rank][peer] = num_channels
        connection_channels = [0] * num_sends
        each_channel = ceil(num_sends / num_channels)
        for i in range(num_sends, step=each_channel):
            connection_channels[i:i+each_channel] = i

        # each_channel = ceil(num_sends / num_channels)
        # for i in range(num_sends, step=each_channel):
        #     block = sends[i:i+each_channel]
        #     for send in block:
        #         send.channel = i
        #         send.recv_match.channel = i
        return True

    def split_connection_interleaved(self, rank, peer, num_channels):
        num_sends = self.connection.num_sends(rank, peer)
        if num_channels > num_sends:
            return False

        connection_channels = [0] * num_sends
        each_channel = ceil(num_sends / num_channels)
        for i in range(num_sends):
            connection_channels[i] = i % num_channels


    def get_connections_by_channel(self, rank, channel):
        scons = defaultdict(list)
        for peer, insts in self.send_connections[rank].items():
            for inst in insts:
                if inst.channel == channel:
                    scons[peer].append(inst)

        rcons = defaultdict(list)
        for peer, insts in self.recv_connections[rank].items():
            for inst in insts:
                if inst.channel == channel:
                    rcons[peer].append(inst)

        return scons, rcons

    def merge_threadblocks(self, scon, rcon, tb):
        for i in scon:
            i.tb = tb
        for i in rcon:
            i.tb = tb
    