import argparse
import os
import pandas

def find_best(directory, inplace):
    nccl = None
    results = {}
    for filename in os.listdir(directory):
        # if 'rexchange' in filename:
        #     continue
        f = os.path.join(directory, filename)
        name = f.split('/')[-1].split('.')[0]
        data = pandas.read_csv(f)
        if name == 'nccl':
            nccl = data
        else:
            results[name] = data
    if inplace:
        time = 'ip-time'
        correct = 'ip-error'
    else:
        time = 'oop-time'
        correct = 'oop-error'
    
    max_speedup = {}
    configuration = {}
    for size in nccl['size']:
        max_speedup[size] = 1
        configuration[size] = 'nccl'

    for size in nccl['size']:
        for config, data in results.items():
            result = data.query(f'size == {size}')
            if len(result) > 0:
                t = result[time].values[0]
                error = result[correct].values[0]
                speedup = nccl.query(f'size == {size}')[time].values[0] / t
                if speedup > max_speedup[size] and error < 2e-5:
                    max_speedup[size] = speedup
                    configuration[size] = config
    for size in configuration.keys():
        print(f"{size}, {configuration[size]}, {max_speedup[size]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help='Directory of nccl-test logs')
    parser.add_argument('--oop', dest='inplace', action='store_false', help='Analyze outofplace stats')
    parser.add_argument('--ip', dest='inplace', action='store_true')
    parser.set_defaults(inplace=True)
    args = parser.parse_args()
    find_best(args.directory, args.inplace)
    # find_best('alltoall', False)
    # find_best('allreduce', True)
    # find_best('allgather', True)

