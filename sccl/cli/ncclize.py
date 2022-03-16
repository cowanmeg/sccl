# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sccl.ncclize import *
from .common import *

def make_handle_ncclize(cmd_parsers):
    cmd = cmd_parsers.add_parser('ncclize')
    read_algorithm = add_input_algorithm(cmd, multiple=True)
    validate_output_args, output_handler = add_output_file(cmd)
    remap_scratch_grp = cmd.add_mutually_exclusive_group()
    remap_scratch_grp.add_argument('--remap-scratch', action='store_true', default=None, help='remap scratch buffer indices into free input/output indices')
    remap_scratch_grp.add_argument('--no-remap-scratch', action='store_false', dest='remap_scratch', help='don\'t remap scratch buffer indices into free input/output indices')
    cmd.add_argument('--no-merge-contiguous', action='store_true', help='don\'t merge sends/receives from/to contiguous memory')
    cmd.add_argument('--no-pretty-print', action='store_true', help='don\'t pretty print the generated XML')
    # cmd.add_argument('--greedy-scratch-sorting', action='store_true', help='sort scratch buffer indices greedily to increase contiguous operations')
    # cmd.add_argument('--no-scratch', action='store_true', help='use extra space at the end of output buffer instead of the scratch buffer')
    # cmd.add_argument('--channel-policy', type=ChannelPolicy, choices=list(ChannelPolicy), default=ChannelPolicy.MatchTopology, help='channel allocation policy')
    # cmd.add_argument('--instances', type=int, default=1, help='number of interleaved instances of the algorithm to make')
    cmd.add_argument('--extra-contig', action='store_true', help='allow lucky contiguity')
    cmd.add_argument('--time-deps', action='store_true', help='add dependency between relay sends')
    cmd.add_argument('--old-format', action='store_true', help='use the old format')
    cmd.add_argument('--use-scratch', action='store_true', help='use the scratch buffer instead of extra space at the end of output buffer')
    cmd.add_argument('--channel-policy', type=ChannelPolicy, choices=list(ChannelPolicy), default=ChannelPolicy.MatchTopology, help='channel allocation policy')
    cmd.add_argument('--instances', type=int, default=1, help='number of interleaved instances of the algorithm to make')
    cmd.add_argument('--scale-remote', type=int, default=1, help='number of interleaved instances of the algorithm to make more for IB')
    cmd.add_argument('--prefix', type=str, default="", help='prefix to add to xmlfile')
    cmd.add_argument('--aid-IB-contig', action="store_true", help='sort scratch buffers of IB send first in order to aid its contiguous send')

    # def handle(args, command):
    #     if command != 'ncclize':
    #         return False

    #     input_algorithms = read_algorithm(args)
    #     validate_output_args(args)

    #     for algo in input_algorithms:
    #         ncclized = ncclize(algo,
    #             remap_scratch=args.remap_scratch,
    #             channel_policy=args.channel_policy,
    #             pretty_print=not args.no_pretty_print,
    #             use_scratch=not args.no_scratch,
    #             merge_contiguous=not args.no_merge_contiguous,
    #             greedy_scratch_sorting=args.greedy_scratch_sorting,
    #             instances=args.instances,
    #             logging=True)

    #         handled = output_handler(args, lambda: ncclized, name_sccl_object(algo.name, ending='sccl.xml'))

    #     return True
    
    # return handle

    def handle(args, command):
        if command != 'ncclize':
            return False

        input_algorithms = read_algorithm(args)
        validate_output_args(args)

        for algo in input_algorithms:
            ncclized = ncclize(algo,
                remap_scratch=args.remap_scratch,
                channel_policy=args.channel_policy,
                pretty_print=not args.no_pretty_print,
                old_format=True,
                use_scratch=args.use_scratch,
                merge_contiguous=not args.no_merge_contiguous,
                instances=args.instances,
                scale_remote=args.scale_remote,
                combine_contig=args.extra_contig,
                add_time_deps=args.time_deps,
                aid_IB_contig=args.aid_IB_contig,
                prefix=args.prefix,
                logging=True)

            algo_name = algo.name.replace("[",".")
            algo_name = algo_name.replace("]","")
            algo_name = algo_name.replace(" ","")
            suffix = ""
            if args.extra_contig:
                suffix += "_extraContig"
            if args.time_deps:
                suffix += "_timeDeps"
            if args.aid_IB_contig:
                suffix += "_IBContig"
            handled = output_handler(args, lambda: ncclized, name_sccl_object(algo_name + f"_i{args.instances}_scRemote{args.scale_remote}{suffix}{args.prefix}", ending='sccl.xml'))

        return True
    
    return handle
