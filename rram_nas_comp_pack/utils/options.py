
import argparse
import numpy as np
import configparser as cp
import yaml
import os

def get_args(config_path = os.path.join(os.getcwd(),"nas_perf_sim/configs","RRAM.yaml")):
    parser = argparse.ArgumentParser(description='RRAM Packing Nas arguments')

    parser.add_argument('--exp-id', type=str, default='dev', help='experiment ID')
    parser.add_argument('--log-pack', dest='log_pack', action='store_true')
    parser.set_defaults(log_pack=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    
    #packing
    parser.add_argument('--model', type=str, default="mbv3_small", help='model for packing: mbv3_small|squeezenet')
    # parser.add_argument('--split-mode', type=str, default="base", help='base | depth, depth for depthwise layer split into number of crossbar')
    parser.add_argument('--num-crossbars', type=int, default=10, help='number of crossbar')
    parser.add_argument('--depth-split-factor', type=int, default=1, help=' for depthwise layer split, factor to split it')
    parser.add_argument('--layer-threshold', type=int, default=1, help='threshold for layer conflict in xbar')
    parser.add_argument('--xbar-max-load', type=int, default=6, help='max number of layers in xbar')
    parser.add_argument('--no-pack-heuristic', dest='pack_heuristic', action='store_false')
    parser.set_defaults(pack_heuristic=True)
    parser.add_argument('--find-solution', dest='find_solution', action='store_true')
    parser.set_defaults(find_solution=False)
    
    #optimization
    parser.add_argument('--duplicate', dest='duplicate', action='store_true')
    parser.set_defaults(duplicate=False)
    
    #evaluation
    parser.add_argument('--num-sample', type=int, default=8, help='number of sample to evaluate in cycle/latency evaluation')
    
    #evolution
    parser.add_argument('--coef-acc', type=int, default=1, help=' weight of accuracy in fitness function')
    parser.add_argument('--coef-latency', type=int, default=1, help=' weight of latency in fitness function')
    parser.add_argument('--evolute-method', type=str, default="ndga", help='evolution method ndga | ga')
    parser.add_argument('--no-fine-tune',action='store_true', help='Forbidden fine tune')
    
    parser.add_argument('--no-cuda',action='store_true', help='Forbidden cuda')
    parser.add_argument('--device', type=int, default=0, help='Which GPU will be called')
    parser.add_argument('--seed',   type=int, default=123, help='Random seed')
    
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    

    args = parser.parse_args()
    if args.no_cuda: args.device = 'cpu'
    
    config = cp.ConfigParser()
    config.read(config_path, encoding='UTF-8')
    args.crossbar_size = config.get('Crossbar level', 'Xbar_Size')
    args.crossbar_size = (int(args.crossbar_size.split(",")[0]),int(args.crossbar_size.split(",")[1]))
    # config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    
    # args.config_path = config_path
    
    # args.crossbar_size = config['crossbar']['xbar_size']
    # args.crossbar_size = (int(args.crossbar_size.split(",")[0]),int(args.crossbar_size.split(",")[1]))
    
    if not args.exp_id:
        args.exp_id = f"test-evolute-{args.model}-num_crossbar:{args.num_crossbars}-depth_split_factor:${args.depth_split_factor}-num_sample:${args.num_sample}_fitness{args.coef_acc}:{args.coef_latency}"

    args.exp_id = args.exp_id + f"-seed:{args.seed}-xbar:{args.crossbar_size[0]}x{args.crossbar_size[1]}"
    
    return args

