# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .topology import Topology

from fractions import Fraction
import subprocess

def dgx1_1KB_AB(remote_alpha, remote_beta, remote_total):
    assert remote_alpha is not None
    assert remote_beta is not None
    assert remote_total is not None
    links = [
        #0  1  2  3  4  5  6  7
        [0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 0]
    ]
    # mult-factor is 100
    invbws = [
        [0, 52, 75, 75, 52, 0, 0, 0],
        [52, 0, 75, 52, 0, 75, 0, 0],
        [75, 75, 0, 52, 0, 0, 52, 0],
        [75, 52, 52, 0, 0, 0, 0, 75],
        [52, 0, 0, 0, 0, 52, 75, 75],
        [0, 75, 0, 0, 52, 0, 75, 52],
        [0, 0, 52, 0, 75, 75, 0, 52],
        [0, 0, 0, 75, 75, 52, 52, 0]
    ]

    # We do not fix remote bw in this topology because it will change
    # depending on if the node is fully connected or distributed relay
    # for same connected, multiply beta by 8 to share link alpha=260, beta=82, total=342
    # for fully connected, multiply beta by 64, alpha=260, beta=656, total=916
    remote_invbw = remote_total
    return Topology('DGX1_1KB_AB', links, invbws=invbws, remote_invbw=remote_invbw, remote_alpha=remote_alpha, remote_beta=remote_beta)

# DGX1 config assuming 1MB data transfer size
def dgx1():
    # (0 1 2 3) (4 5 6 7) are two sockets
    # 0 1 3 2 is the high bandwidth chain in socket 1
    # 4 5 7 6 is the high bandwidth chain in socket 2
    # 0 4 and 2 6 are high bandwidth intersocket links

    # links = [
    #     #0  1  2  3  4  5  6  7
    #     [0, 2, 1, 1, 2, 0, 0, 0],
    #     [2, 0, 1, 2, 0, 1, 0, 0],
    #     [1, 1, 0, 2, 0, 0, 2, 0],
    #     [1, 2, 2, 0, 0, 0, 0, 1],
    #     [2, 0, 0, 0, 0, 2, 1, 1],
    #     [0, 1, 0, 0, 2, 0, 1, 2],
    #     [0, 0, 2, 0, 1, 1, 0, 2],
    #     [0, 0, 0, 1, 1, 2, 2, 0]
    # ]

    # Link connection matrix
    links = [
        #0  1  2  3  4  5  6  7
        [0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 0]
    ]

    # NVLink bandwidth for each link
    # high bandwidth link => alpha=0, beta=23/46
    invbws = [
        [0, 23, 46, 46, 23, 0, 0, 0],
        [23, 0, 46, 23, 0, 46, 0, 0],
        [46, 46, 0, 23, 0, 0, 23, 0],
        [46, 23, 23, 0, 0, 0, 0, 46],
        [23, 0, 0, 0, 0, 23, 46, 46],
        [0, 46, 0, 0, 23, 0, 46, 23],
        [0, 0, 23, 0, 46, 46, 0, 23],
        [0, 0, 0, 46, 46, 23, 23, 0]
    ]

    remote_invbw = 107  # approx IB alpha + beta = 2 + 105
    remote_alpha = 2.6  # IB alpha = 2.6
    remote_beta = 105   # IB beta = 105

    # self.symmetries = [
    #     [0, 1, 2, 3, 4, 5, 6, 7], #0 goes to itself
    #     [0, 1, 2, 3, 4, 5, 6, 7], #1 goes to itself
    #     [2, 3, 0, 1, 6, 7, 4, 5], #2 goes to 0, 3 goes to 1, ... top - bottom symmetry
    #     [2, 3, 0, 1, 6, 7, 4, 5], #3 goes to 1, 2 goes to 0, ... top - bottom symmetry
    #     [4, 5, 6, 7, 0, 1, 2, 3], #4 goes to 0, 5 goes to 1, ... left - right symmetry
    #     [4, 5, 6, 7, 0, 1, 2, 3], #5 goes to 1, 4 goes to 0, ... left - right symmetry
    #     [6, 7, 4, 5, 2, 3, 0, 1], #6 goes to 0, 7 goes to 1, ... top-bottom + left-right
    #     [6, 7, 4, 5, 2, 3, 0, 1]  #7 goes to 1, 6 goes to 0, ... top-bottom + left-right
    # ]

    # self.beta_bound = Fraction(7,6)
    # self.diameter = 2

    return Topology('DGX1', links, invbws=invbws, remote_invbw=remote_invbw, remote_alpha=remote_alpha, remote_beta=remote_beta)

# DGX1 config with high link latency
def dgx1_32KB():
    # (0 1 2 3) (4 5 6 7) are two sockets
    # 0 1 3 2 is the high bandwidth chain in socket 1
    # 4 5 7 6 is the high bandwidth chain in socket 2
    # 0 4 and 2 6 are high bandwidth intersocket links

    # Link connection matrix
    links = [
        #0  1  2  3  4  5  6  7
        [0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 0]
    ]

    # NVLink bandwidth for each link
    # 32KB size link => alpha=0.3, beta=1.44/0.72. Save invbws as (alpha + beta)*10 (10 is mult factor)
    invbws = [
        [0, 10, 17, 17, 10, 0, 0, 0],
        [10, 0, 17, 10, 0, 17, 0, 0],
        [17, 17, 0, 10, 0, 0, 10, 0],
        [17, 10, 10, 0, 0, 0, 0, 17],
        [10, 0, 0, 0, 0, 10, 17, 17],
        [0, 17, 0, 0, 10, 0, 17, 10],
        [0, 0, 10, 0, 17, 17, 0, 10],
        [0, 0, 0, 17, 17, 10, 10, 0]
    ]

    remote_invbw = 60    # IB alpha + beta = (2.7 + 3.3)*10
    remote_alpha = 27    # IB alpha = 2.7 * 10
    remote_beta = 33     # IB beta = 0

    return Topology('DGX1_32KB', links, invbws=invbws, remote_invbw=remote_invbw, remote_alpha=remote_alpha, remote_beta=remote_beta)

# DGX1 config with high link latency
def dgx1_2KB():
    # (0 1 2 3) (4 5 6 7) are two sockets
    # 0 1 3 2 is the high bandwidth chain in socket 1
    # 4 5 7 6 is the high bandwidth chain in socket 2
    # 0 4 and 2 6 are high bandwidth intersocket links

    # Link connection matrix
    links = [
        #0  1  2  3  4  5  6  7
        [0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 0]
    ]

    # NVLink bandwidth for each link
    # 32KB size link => alpha=0.3, beta=1.44/0.72. Save invbws as (alpha + beta)*10 (10 is mult factor)
    invbws = [
        [0, 35, 39, 39, 35, 0, 0, 0],
        [35, 0, 39, 35, 0, 39, 0, 0],
        [39, 39, 0, 35, 0, 0, 35, 0],
        [39, 35, 35, 0, 0, 0, 0, 39],
        [35, 0, 0, 0, 0, 35, 39, 39],
        [0, 39, 0, 0, 35, 0, 39, 35],
        [0, 0, 35, 0, 39, 39, 0, 35],
        [0, 0, 0, 39, 39, 35, 35, 0]
    ]

    remote_invbw = 281    # IB alpha + beta = (2.7 + 3.3)*10
    remote_alpha = 260    # IB alpha = 2.7 * 10
    remote_beta = 21     # IB beta = 0

    return Topology('DGX1_2KB', links, invbws=invbws, remote_invbw=remote_invbw, remote_alpha=remote_alpha, remote_beta=remote_beta)


# DGX1 config with high link latency
def dgx1_lat():
    # (0 1 2 3) (4 5 6 7) are two sockets
    # 0 1 3 2 is the high bandwidth chain in socket 1
    # 4 5 7 6 is the high bandwidth chain in socket 2
    # 0 4 and 2 6 are high bandwidth intersocket links

    # Link connection matrix
    links = [
        #0  1  2  3  4  5  6  7
        [0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 0]
    ]

    # NVLink bandwidth for each link
    # high latency link => alpha=0.3, beta=0
    invbws = [
        [0, 0.3, 0.3, 0.3, 0.3, 0, 0, 0],
        [0.3, 0, 0.3, 0.3, 0, 0.3, 0, 0],
        [0.3, 0.3, 0, 0.3, 0, 0, 0.3, 0],
        [0.3, 0.3, 0.3, 0, 0, 0, 0, 0.3],
        [0.3, 0, 0, 0, 0, 0.3, 0.3, 0.3],
        [0, 0.3, 0, 0, 0.3, 0, 0.3, 0.3],
        [0, 0, 0.3, 0, 0.3, 0.3, 0, 0.3],
        [0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0]
    ]

    remote_invbw = 3    # IB alpha + beta = 3 + 0
    remote_alpha = 3    # IB alpha = 3
    remote_beta = 0     # IB beta = 0

    return Topology('DGX1Lat', links, invbws=invbws, remote_invbw=remote_invbw, remote_alpha=remote_alpha, remote_beta=remote_beta)

def nvlink_only(nvidia_smi_topo=None):
    if nvidia_smi_topo == None:
        nvidia_smi_topo = _get_nvidia_smi_topo()
    links = _parse_nvidia_smi_topo(nvidia_smi_topo)
    return Topology('NVLinkOnly', links)

def _get_nvidia_smi_topo():
    output = subprocess.check_output("nvidia-smi topo -m".split())
    return output.decode("utf-8")

def _parse_nvidia_smi_topo(output):
    lines = output.splitlines()
    before_legend = []
    for l in lines[1:]:
        if l and l.startswith("GPU"):
            # Only look at the rows for GPU
            before_legend.append(l)
        else:
            break
    devices = [x.split("\t")[0] for x in before_legend]
    gpus = [i for i in range(len(before_legend))
            if before_legend[i].startswith("GPU")]
    matrix = [x.split("\t")[1:] for x in before_legend]
    nvlink_matrix = [[_nvlink_num(x[g]) for g in gpus] for x in matrix]
    return nvlink_matrix

def _nvlink_num(x):
    x = x.strip()
    if x.startswith("NV"):
        return int(x[2:])
    else:
        return 0