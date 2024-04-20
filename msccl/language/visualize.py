# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import graphviz as gz
from msccl.language.ir import *
from msccl.language.instruction_dag import *

def getOpcode(op):
    if op.inst == Instruction.start:
        opcode = (f'Chunk')
    elif op.inst == Instruction.send:
        opcode = (f'S')
    elif op.inst == Instruction.recv:
        opcode = (f'R')
    elif op.inst == Instruction.recv_reduce_copy:
        opcode = (f'RRC')
    elif op.inst == Instruction.recv_reduce_copy_send:
        opcode = (f'RRCS')
    elif op.inst == Instruction.recv_reduce_send:
        opcode = (f'RRS')
    elif op.inst == Instruction.recv_copy_send:
        opcode = (f'RCS')
    else:
        opcode = (f'{op.inst}')
    return opcode

num = 0
def visualize_instr_dag(program, dot_file):
    global num
    num = 0
    dot = gz.Digraph('Instruction DAG')
    for i, gpu in enumerate(program.gpus):
        for j, tb in enumerate(gpu.threadblocks):
            for op in tb.ops:
                opcode = getOpcode(op)
                op.dot_node = str(num)
                dot.node(str(num), label=f'{opcode} rank:{op.rank} chunk:{op.src.index}')
                num += 1
    for i, gpu in enumerate(program.gpus):
        for j, tb in enumerate(gpu.threadblocks):
            for op in tb.ops:
                if op.is_send():
                    dot.edge(op.dot_node, op.recv_match.dot_node, style='dotted')
                for dop in op.depends:
                    dot.edge(dop.dot_node, op.dot_node)
    with open(dot_file, 'w') as file:
        file.write(dot.source)


def visualize_msccl_ir(program, dot_file):
    global num
    num = 0
    dot = gz.Digraph('MSCCL-IR')

    for i, gpu in enumerate(program.gpus):
        with dot.subgraph(name=f'cluster_gpu{i}') as gpu_sub:
            gpu_sub.attr(label=f'GPU {i}')
            gpu_sub.attr(style='filled', color='lightgrey')
            for j, tb in enumerate(gpu.threadblocks):
                with gpu_sub.subgraph(name=f'cluster_tb{i}{j}') as tb_sub:
                    tb_sub.attr(label=f'TB {j}')
                    tb_sub.attr(style='filled', color='white')
                    for k, op in enumerate(tb.ops):
                        opcode = getOpcode(op)
                        op.dot_node = str(num)
                        tb_sub.node(str(num), label=f'{opcode} chunk:{op.src.index}')
                        # fake formatting edge
                        if (k > 0):
                            tb_sub.edge(str(num-1), str(num), color='white')
                        num += 1
    for i, gpu in enumerate(program.gpus):
        for j, tb in enumerate(gpu.threadblocks):
            for op in tb.ops:
                if op.is_send():
                    dot.edge(op.dot_node, op.recv_match.dot_node, style='dotted')
                for dop in op.cross_tb_depends:
                    dot.edge(dop.dot_node, op.dot_node)
    with open(dot_file, 'w') as file:
        file.write(dot.source)
