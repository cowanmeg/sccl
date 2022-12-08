import argparse
import os
from scripts.common import *

# home = os.getcwd()
home = '/msrhyper-ddn/hai8/cowanmeg'
CHANNELS = 32
SMS = 80
gpus_per_node = 16

# NCCL 2.12
default_lower = '512'
LL_upper = '512M'
default_upper = '2G'
machine = 'dgx2'
MSCCL = f'{home}/msccl/build/lib/'
NCCL_TESTS = f'{home}/nccl-tests/build'

# NCCL 2.8.4
# default_lower = '512B'
# LL_upper = '512MB'
# default_upper = '2GB'
# machine = 'dgx2-2.8.4'
# MSCCL = f'{home}/msccl-2.8/build/lib/'
# NCCL_TESTS = f'{home}/nccl-tests-2.8/build'

# Runs through our multi-node algorithms for DGX2

def mpirun(collective, gpus, xml, txt, lower=default_lower, upper=default_upper, algo='MSCCL,RING,TREE'):
    # if not debug:
    #     for n in range(1, nodes):
    #         os.system(f'scp {xml} worker-{n}:{xml}')
    cmd = f'mpirun --tag-output --allow-run-as-root -np {gpus} --bind-to numa -hostfile ~/hostfile ' \
        f'-x NCCL_ALGO={algo} -x LD_LIBRARY_PATH={MSCCL} ' \
        f'-mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include enp134s0f1 -mca coll_hcoll_enable 0 '\
        f'-mca plm_rsh_no_tree_spawn 1 -mca plm_rsh_num_concurrent 8192 -x PATH  '\
        f'-x NCCL_UCX_TLS=rc_x,cuda_copy,cuda_ipc -x NCCL_UCX_RNDV_THRESH=0 -x NCCL_UCX_RNDV_SCHEME=get_zcopy '\
        f'-x UCX_RC_MLX5_TM_ENABLE=y -x UCX_MEMTYPE_CACHE=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 '\
        f'-x UCX_IB_PCI_RELAXED_ORDERING=on -x NCCL_NET_GDR_LEVEL=5 -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_PLUGIN_P2P=ib '\
        f'-x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 '\
        f'-x MSCCL_XML_FILES={xml} {NCCL_TESTS}/{collective}_perf ' \
        f'-g 1 -n 50 -w 25 -f 2 -c 1 -z 0 -b {lower} -e {upper} > {txt}'
    print(f'$ {cmd}')

    if not debug:
        if xml is None or valid_resources(xml, SMS, CHANNELS):
            os.system(cmd)
        else:
            print('Not enough resources to run')

def mpirun_nccl(collective, gpus, txt, lower=default_lower, upper=default_upper, algo='RING,TREE'):
    mpirun(collective, gpus, None, txt, lower, upper, algo=algo)


def allreduce_hierarchical():
    def run(instances, protocol, schedule, lower=default_lower, upper=default_upper):
        xml = f"{home}/xmls/allreduce_hierarchical_{instances}_{protocol}_{schedule}.xml"
        txt = f"{home}/{machine}/allreduce_{nodes}nodes/hierarchical_{instances}_{protocol}_{schedule}.txt"
        print(f'Generating {xml} {txt}')
        if compile:
            cmd = f'python3 sccl/examples/scclang/hierarchical_allreduce_blueconnect.py {gpus_per_node} {nodes} '\
                f'{instances} --protocol={protocol}  --schedule={schedule} --device=V100 --output={xml}'
            print(f'$ {cmd}')
            os.system(cmd)
        mpirun('all_reduce', gpus, xml, txt, lower, upper)

    # run(4, 'Simple', 'const')
    # run(4, 'LL128',  'const')
    # run(2, 'Simple', 'const')
    # run(2, 'LL128',  'const')

    for protocol in ['Simple', 'LL', 'LL128']:
        for instances in [1, 2, 4]:
            chunks = gpus * instances
            lower = chunks * 4
            if protocol == 'LL':
                upper = LL_upper
            else:
                upper = default_upper
            run(instances, protocol, 'const', upper=upper)

def allgather_hierarchical():
    def run(instances, protocol, channels, lower=default_lower, upper=default_upper):
        xml = f"{home}/xmls/allgather_hierarchical_{instances}_{protocol}_{channels}.xml"
        txt = f"{home}/{machine}/allgather_{nodes}nodes/hierarchical_{instances}_{protocol}_{channels}.txt"
        print(f'Generating {xml} {txt}')
        if compile:
            cmd = f'python3 sccl/examples/scclang/hierarchical_allgather.py {gpus_per_node} {nodes} '\
                f'{channels} {instances} --device=V100 --output={xml}'
            print(f'$ {cmd}')
            os.system(cmd)
        mpirun('all_reduce', gpus, xml, txt, lower, upper)

    for instances in range[1, 2, 4]:
        for channels in range[1, 2, 4]:
            for protocol in ['Simple', 'LL128', 'LL']:
                if protocol == 'LL':
                    upper = LL_upper
                else:
                    upper = default_upper
                run(instances, protocol, channels, upper=upper)

def alltoall_two_step():
    def run(instances, protocol, lower=default_lower, upper=default_upper):
        xml = f"{home}/xmls/alltoall_two_step_{nodes}_{instances}_{protocol}.xml"
        txt = f"{home}/{machine}/alltoall_{nodes}nodes/alltoall_two_step_{nodes}_{instances}_{protocol}.txt"
        print(f'Generating {xml} {txt}')
        if compile:
            cmd = f'python3 sccl/examples/scclang/alltoall_a100_two_step.py {nodes} {gpus_per_node} {instances} --protocol={protocol} --device=V100 --output={xml}'
            print(f'$ {cmd}')
            os.system(cmd)
        mpirun('alltoall', gpus, xml, txt, lower, upper)

    for instances in [1]:
        for protocol in ['Simple', 'LL128', 'LL']:
            if protocol == 'LL':
                    upper = LL_upper
            else:
                upper = default_upper
            run(instances, protocol, upper=upper)

def alltoall_three_step():
    def run(instances, protocol, lower=default_lower, upper=default_upper):
        xml = f"{home}/xmls/alltoall_three_step_{nodes}_{instances}_{protocol}.xml"
        txt = f"{home}/{machine}/alltoall_{nodes}nodes/alltoall_three_step_{nodes}_{instances}_{protocol}.txt"
        print(f'Generating {xml} {txt}')
        if compile:
            cmd = f'python3 sccl/examples/scclang/alltoall_a100_three_step.py {nodes} {gpus_per_node} {instances} --protocol={protocol} --device=V100 --output={xml}'
            print(f'$ {cmd}')
            os.system(cmd)
        mpirun('alltoall', gpus, xml, txt, lower, upper)

    for instances in [1]:
        for protocol in ['Simple', 'LL128', 'LL']:
            if protocol == 'LL':
                    upper = LL_upper
            else:
                upper = default_upper
            run(instances, protocol)

def alltonext():
    def run(instances, protocol, lower=default_lower):
        xml = f"{home}/xmls/forward_{nodes}_{instances}_{protocol}_half.xml"
        txt = f"{home}/{machine}/alltonext_{nodes}nodes/forward_{nodes}_{instances}_{protocol}_half.txt"
        print(f'Generating {xml} {txt}')
        if compile:
            cmd = f'python3 sccl/examples/scclang/alltonext_forward.py {gpus_per_node} {nodes} {instances} --version=half --device=V100 --output={xml}'
            print(f'$ {cmd}')
            os.system(cmd)
        mpirun('alltonext', gpus, xml, txt, lower)

    for instances in [1, 2, 4]:
        for protocol in ['Simple', 'LL', 'LL128']:
            run(instances, protocol)



def allgather_nccl():
    print("Run NCCL AllGather")
    mpirun_nccl('all_gather', gpus, f'{home}/{machine}/allgather_{nodes}nodes/nccl.txt')

def allreduce_nccl():
    print("Run NCCL AllReduce")
    mpirun_nccl('all_reduce', gpus, f'{home}/{machine}/allreduce_{nodes}nodes/nccl.txt')

def alltoall_cuda_two_step():
    print("Run CUDA Two-Step AllToAll")
    mpirun('alltoall', gpus, None, f'{home}/{machine}/alltoall_{nodes}nodes/nccl_two_step.txt', algo='2D')

def alltoall_nccl():
    print("Run NCCL AllToAll")
    mpirun_nccl('alltoall', gpus,  f'{home}/{machine}/alltoall_{nodes}nodes/nccl.txt')

def alltonext_nccl():
    print("Run AllToNext Baseline")
    mpirun_nccl('alltonext_nccl', gpus, f'{home}/{machine}/alltonext_{nodes}nodes/nccl.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('nodes', type=int, help ='Number of nodes to run algorithms on')
    parser.add_argument('--debug', default=False, action='store_true', help ='Precompile - generate xmls only')
    parser.add_argument('--nocompile', default=True, action='store_false', help ='Do not generate xmls - run if precompiled')
    args = parser.parse_args()
    global nodes, gpus, debug, compile
    debug = bool(args.debug)
    compile = bool(args.nocompile)
    nodes = args.nodes
    gpus = gpus_per_node * nodes

    check_create(f'{home}/{machine}')
    check_create(f'{home}/{machine}/allreduce_{nodes}nodes')
    check_create(f'{home}/{machine}/alltonext_{nodes}nodes')
    check_create(f'{home}/{machine}/alltoall_{nodes}nodes')
    check_create(f'{home}/{machine}/allgather_{nodes}nodes')
    check_create(f'xmls')

    #NCCL Baselines
    allreduce_nccl()
    allgather_nccl()
    alltoall_nccl()
    alltonext_nccl()
     
    alltoall_two_step()
    alltoall_three_step()
    alltoall_cuda_two_step()

    allreduce_hierarchical()

    alltonext()

    allgather_hierarchical()
    


    

