import argparse
import os
from xml.etree import ElementTree as ET

home = os.getcwd()
CHANNELS = 32
SMS = 80
dir = '/msrhyper-ddn/hai8/meghancowan'
# Runs through our multi-node algorithms for DGX2

def check_threadblocks(attributes):
    nthreadblocks = int(attributes['nthreadblocks'])
    valid = nthreadblocks <= SMS
    if not valid:
        print(f"WARNING: Using {nthreadblocks} while GPU can support up to {SMS} thread blocks")
    print(f"Using {nthreadblocks} thread blocks")
    return valid

def check_channels(attributes):
    nchannels = int(attributes['nchannels'])
    valid = nchannels <= CHANNELS
    if not valid:
        print(f"WARNING: Using {nchannels} while up to {CHANNELS} channels are allowed")
    print(f"Using {nchannels} channels")
    return valid

def valid_resources(xml):
    header = open(xml, 'r').readline().strip('\n\r') + "</algo>"
    txml = ET.fromstring(header)
    attributes = txml.attrib
    check_threadblocks(attributes)
    check_channels(attributes)


def mpirun(collective, gpus, xml, txt, lower='512B', upper='4GB'):
    # for n in range(1, nodes):
    #     os.system(f'scp {xml} worker-{n}:{xml}')
    cmd = f'mpirun --tag-output --allow-run-as-root -np {gpus} --bind-to numa -hostfile {home}/hostfile ' \
        f'-x NCCL_ALGO=MSCCL,RING,TREE -x LD_LIBRARY_PATH={dir}/msccl/build/lib/ ' \
        f'-mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include enp134s0f1 -mca coll_hcoll_enable 0 '\
        f'-mca plm_rsh_no_tree_spawn 1 -mca plm_rsh_num_concurrent 8192 -x PATH  '\
        f'-x NCCL_UCX_TLS=rc_x,cuda_copy,cuda_ipc -x NCCL_UCX_RNDV_THRESH=0 -x NCCL_UCX_RNDV_SCHEME=get_zcopy '\
        f'-x UCX_RC_MLX5_TM_ENABLE=y -x UCX_MEMTYPE_CACHE=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 '\
        f'-x UCX_IB_PCI_RELAXED_ORDERING=on -x NCCL_NET_GDR_LEVEL=5 -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_PLUGIN_P2P=ib '\
        f'-x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 '\
        f'-x MSCCL_XML_FILES={xml} {dir}/nccl-tests/build/{collective}_perf ' \
        f'-g 1 -n 50 -w 25 -f 2 -c 1 -z 0 -b {lower} -e {upper} > {txt}'
    print(f'Running {cmd}')
    os.system(cmd)

def mpirun_nccl(collective, gpus, txt, lower='512B', upper='4GB', algo='RING,TREE'):
    cmd = f'mpirun --tag-output --allow-run-as-root -np {gpus} -hostfile {home}/hostfile --bind-to numa ' \
        f'-x NCCL_ALGO={algo} -x LD_LIBRARY_PATH={dir}/msccl/build/lib/  ' \
        f'-mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include enp134s0f1 -mca coll_hcoll_enable 0 '\
        f'-mca plm_rsh_no_tree_spawn 1 -mca plm_rsh_num_concurrent 8192 -x PATH  '\
        f'-x NCCL_UCX_TLS=rc_x,cuda_copy,cuda_ipc -x NCCL_UCX_RNDV_THRESH=0 -x NCCL_UCX_RNDV_SCHEME=get_zcopy '\
        f'-x UCX_RC_MLX5_TM_ENABLE=y -x UCX_MEMTYPE_CACHE=n -x NCCL_IB_PCI_RELAXED_ORDERING=1 '\
        f'-x UCX_IB_PCI_RELAXED_ORDERING=on -x NCCL_NET_GDR_LEVEL=5 -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_PLUGIN_P2P=ib '\
        f'-x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 '\
        f'-x NCCL_PROTO=SIMPLE,LL128,LL {dir}/nccl-tests/build/{collective}_perf ' \
        f'-g 1 -n 50 -w 25 -f 2 -c 1 -z 0 -b {lower} -e {upper} > {txt}'
    print(f'Running {cmd}')
    os.system(cmd)

def allreduce_rexchange():
    assert nodes == 2, f"Rexchange hierarchical allreduce only works for 2 nodes"
    def run(instances, protocol, schedule, lower='512B'):
        xml = f"{home}/xmls/allreduce/rexchange_{instances}_{protocol}_{schedule}.xml"
        txt = f"{home}/{machine}/allreduce_{nodes}nodes/rexchange_{instances}_{protocol}_{schedule}.txt"
        print(f'Generating {xml} {txt}')
        cmd = f'python3 sccl/examples/scclang/allreduce_a100_hierarchical.py {gpus_per_node} {nodes} {instances} --protocol={protocol} --schedule={schedule} > {xml}'
        print(f'Running {cmd}')
        os.system(cmd)
        mpirun('all_reduce', gpus, xml, txt, lower)

    run(8, 'Simple', 'manual')
    run(2, 'LL128', 'auto')
    run(1, 'LL', 'auto')

def allreduce_hierarchical():
    def run(instances, protocol, version, schedule, lower='512B'):
        xml = f"{home}/xmls/allreduce/hierarchical_{instances}_{protocol}_{version}_{schedule}.xml"
        txt = f"{home}/{machine}/allreduce_{nodes}nodes/hierarchical_{instances}_{protocol}_{version}_{schedule}.txt"
        print(f'Generating {xml} {txt}')
        cmd = f'python3 sccl/examples/scclang/hierarchical_allreduce_blueconnect.py {gpus_per_node} {nodes} '\
            f'{instances} --protocol={protocol} --version={version} --schedule={schedule} > {xml}'
        print(f'Running {cmd}')
        os.system(cmd)
        mpirun('all_reduce', gpus, xml, txt, lower)

    run(1, 'LL', 'v1', 'auto')
    run(1, 'LL128', 'v1', 'auto')

    # run(4, 'Simple', 'v1', 'manual')
    run(4, 'LL128', 'v1', 'manual')
    run(2, 'Simple', 'v1', 'manual')
    run(2, 'LL128', 'v1', 'manual')

    # for version in ['v1']:
    #     for protocol in ['Simple', 'LL', 'LL128']:
    #         for instances in [1]:
    #             # Check that we don't go over the channel limit
    #             if 4 * instances <= 32:
    #                 chunks = gpus * instances
    #                 lower = chunks * 4
    #                 run(instances, protocol, version)

def alltoall_2d():
    def run(instances, protocol, lower='512B'):
        xml = f"{home}/xmls/alltoall/alltoall_2d_{nodes}_{instances}_{protocol}.xml"
        txt = f"{home}/{machine}/alltoall_{nodes}nodes/alltoall_2d_{nodes}_{instances}_{protocol}.txt"
        print(f'Generating {xml} {txt}')
        cmd = f'python3 sccl/examples/scclang/alltoall_a100_yifan.py {nodes} {gpus_per_node} {instances} --protocol={protocol} > {xml}'
        print(f'Running {cmd}')
        os.system(cmd)
        mpirun('alltoall', gpus, xml, txt, lower)
    for instances in [1]:
        for protocol in ['Simple', 'LL', 'LL128']:
            run(instances, protocol)

def alltonext():
    def run(instances, protocol, lower='512B'):
        xml = f"{home}/xmls/alltonext/forward_{nodes}_{instances}_{protocol}_half.xml"
        txt = f"{home}/{machine}/alltonext_{nodes}nodes/forward_{nodes}_{instances}_{protocol}_half.txt"
        print(f'Generating {xml} {txt}')
        cmd = f'python3 sccl/examples/scclang/alltonext_forward.py {gpus_per_node} {nodes} {instances} --version=half > {xml}'
        print(f'Running {cmd}')
        os.system(cmd)
        mpirun('alltonext', gpus, xml, txt, lower)
    run(2, 'Simple')
    run(4, 'Simple')
    # run(8, 'Simple')


def allgather_nccl():
    print("Run NCCL AllGather")
    mpirun_nccl('all_gather', gpus, f'{home}/{machine}/allgather_{nodes}nodes/nccl.txt')

def allreduce_nccl():
    print("Run NCCL AllReduce")
    mpirun_nccl('all_reduce', gpus, f'{home}/{machine}/allreduce_{nodes}nodes/nccl.txt')

def alltoall_cuda_2d():
    print("Run CUDA Two-Step AllToAll")
    mpirun('alltoall', gpus, '2D', f'{home}/{machine}/alltoall_{nodes}nodes/nccl_2d.txt')

def alltoall_nccl():
    print("Run NCCL AllToAll")
    mpirun_nccl('alltoall', gpus,  f'{home}/{machine}/alltoall_{nodes}nodes/nccl.txt')

def alltonext_nccl():
    print("Run AllToNext Baseline")
    mpirun_nccl('alltonext_nccl', gpus, f'{home}/{machine}/alltonext_{nodes}nodes/nccl.txt')

def check_create(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('nodes', type=int, help ='Number of nodes to run algorithms on')
    # args = parser.parse_args()
    # global machine, gpus, nodes
    # machine = 'dgx2'
    # gpus_per_node = 16
    # nodes = args.nodes
    # gpus = gpus_per_node * nodes

    # check_create(f'{machine}')
    # check_create(f'{machine}/allreduce_{nodes}nodes')
    # check_create(f'{machine}/alltonext_{nodes}nodes')
    # check_create(f'{machine}/alltoall_{nodes}nodes')
    # check_create(f'xmls')
    # check_create(f'xmls/allreduce')
    # check_create(f'xmls/alltonext')
    # check_create(f'xmls/alltoall')

    # #NCCL Baselines
    # # allreduce_nccl()
    # # alltoall_nccl()
     

    # # AllToAll
    # # alltoall_2d()
    # # alltoall_cuda_2d()

    # # AllReduce
    # # allreduce_rexchange()
    # # allreduce_hierarchical()

    # alltonext()
    # alltonext_nccl()


    

