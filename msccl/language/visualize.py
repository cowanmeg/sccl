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
                    dot.edge(op.dot_node, dop.dot_node)
    with open(dot_file, 'w') as file:
        file.write(dot.source)


def visualize_msccl_ir(program, dot_file):
    global num
    num = 0
    dot = gz.Digraph('MSCCL-IR')

    for i, gpu in enumerate(program.gpus):
        print(f"gpu {i}")
        with dot.subgraph(name=f'cluster_{i}') as gpu_sub:
            gpu_sub.attr(label=f'GPU {i}')
            gpu_sub.attr(style='filled', color='lightgrey')
            for j, tb in enumerate(gpu.threadblocks):
                print(f"gpu {i}, threadblock {j}")
                with gpu_sub.subgraph(name=f'cluster_{i}{j}') as tb_sub:
                    tb_sub.attr(label=f'TB {j}')
                    tb_sub.attr(style='filled', color='white')
                    for k, op in enumerate(tb.ops):
                        opcode = getOpcode(op)
                        op.dot_node = str(num)
                        dot.node(str(num), label=f'{opcode} chunk:{op.src.index}')
                        if (k < len(tb.ops)-1):
                            tb_sub.edge(str(num), str(num+1))
                        num += 1
                    
                        # if (i < len(tb.ops)-1):
                        #     tb_sub.edge(str(num), str(num+1))
    for i, gpu in enumerate(program.gpus):
        for j, tb in enumerate(gpu.threadblocks):
            for op in tb.ops:
                if op.is_send():
                    dot.edge(op.dot_node, op.recv_match.dot_node, style='dotted')
    with open(dot_file, 'w') as file:
        file.write(dot.source)
