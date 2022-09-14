import argparse
import os
from xml.etree import ElementTree as ET

machine = 'mi200'
gpus_per_node = 16
home = os.getcwd()
CHANNELS = 32
SMS = 80

def allreduce_hierarchical():
    def run(instances, protocol):
        xml = f"{home}/xmls/allreduce/hierarchical_{instances}_{protocol}.xml"
        print(f'Generating {xml}')
        cmd = f'python3 sccl/examples/scclang/allreduce_mi200_hierarchical.py {nodes} {instances} --protocol={protocol} > {xml}'
        os.system(cmd)

    for protocol in ['LL', 'LL128', 'Simple']:
        for instances in [1, 2]:
            run(instances, protocol)

def check_create(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('nodes', type=int, help ='Number of nodes to run algorithms on')
    args = parser.parse_args()
    global nodes
    nodes = args.nodes
    gpus = gpus_per_node * nodes

    check_create(f'{machine}')
    check_create(f'{machine}/allreduce_{nodes}nodes')
    check_create(f'xmls')
    check_create(f'xmls/allreduce')

    allreduce_hierarchical()