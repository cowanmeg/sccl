import argparse
import csv
import os
import re

# Parse nccl logs in a directory into a csv


def check_create(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def parse(filename, output):
    parts = filename.split('.')
    # output = parts[0] + '.csv'
    # print("Reading log", filename)
    using_sccl = False
    labels = []
    results = []
    with open(filename, 'r') as f:
        
        line = f.readline()
        if "stdout" in line:
            stdout = True
            result_str = r"^\[1,0\]<stdout>:\s+[0-9]+\s+[0-9]+"
        else:
            stdout = False
            result_str = r"^\s+[0-9]+\s+[0-9]+"
        while line:
            if re.search(r"NCCL\sINFO\sConnected\s[0-9]+\sMSCCL\salgorithms", line):
                using_sccl = True
            elif re.search(r"size\s+count", line):
                labels = line.split()[1:]
            elif re.search(result_str, line):
                nums = line.split()
                if stdout:
                    results.append(nums[1:])
                else:
                    results.append(nums)
            line = f.readline()

    if not using_sccl:
        print(f"Using NCCL: {filename}")

    with open(output, 'w') as f:
        if len(results) > 0:
            print(len(results[0]), results[0])
            if len(results[0]) == 11:
                f.write("size,count,type,oop-time,oop-algbw,oop-busbw,oop-error,ip-time,ip-algbw,ip-busbw,ip-error\n")
            elif len(results[0]) == 12:
                f.write("size,count,type,redop,oop-time,oop-algbw,oop-busbw,oop-error,ip-time,ip-algbw,ip-busbw,ip-error\n")
            elif len(results[0]) == 13:
                f.write("size,count,type,redop,root,oop-time,oop-algbw,oop-busbw,oop-error,ip-time,ip-algbw,ip-busbw,ip-error\n")
            writer = csv.writer(f)
            writer.writerows(results)

def parse_directory(directory):
    home = os.getcwd()
    directory = directory.rstrip('/')
    csv_directory = f'{home}/{directory}_csv'
    check_create(csv_directory)
    
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and f.endswith('.txt'):
            name = f.split('/')[-1].split('.')[0]
            csv_file = f'{csv_directory}/{name}.csv'
            parse(f, csv_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help='Directory of nccl-test logs')
    args = parser.parse_args()
    parse_directory(args.directory)
    
