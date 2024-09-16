import time 
import torch
from torchvision import transforms, datasets
import random
import numpy as np 
import os 
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler


from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3_small
from ofa.imagenet_classification.data_providers import CIFARDataProvider
from rram_nas_comp_pack.box_converter.LP.LPDuplicateOptimizer import LPDuplicateOptimizer
from rram_nas_comp_pack.box_converter.InputLayerBoxSpace import InputLayerBoxSpace
from rram_nas_comp_pack.evaluation.LatencyEvaluator import LatencyEvaluator
from rram_nas_comp_pack.packing import NetworkPacker
from rram_nas_comp_pack.box_converter.models import get_model
from rram_nas_comp_pack.utils.options import get_args
from rram_nas_comp_pack.utils.logger import EvolutionWriter, log_args
from rram_nas_comp_pack.evolution.accuracy.accuracy_generator import AccuracyGenerator
from rram_nas_comp_pack.evolution.evolution_finder import EvolutionFinder

import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)

def setup_logger(args):
    exp_id = args.exp_id
    
    log_writer_path = './log/evolution/{}'.format(exp_id)
    if not os.path.exists(log_writer_path):
        os.makedirs(log_writer_path)

    file_handler = logging.FileHandler(os.path.join(log_writer_path, 'logs.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    #for checkpoint
    args.ckpt_path = os.path.join(log_writer_path, 'checkpoint')
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
        
    csv_writer = EvolutionWriter(log_writer_path, resume=args.resume)
    
    return csv_writer


def main(args):
    csv_writer = setup_logger(args)
    log_args(args)
    timeStr = args.exp_id
    
    crossbar_size = args.crossbar_size
    number_of_crossbars = args.num_crossbars
    seed = args.seed
    
    quantize_config = {
        'weight_bit': 8,
    }
    
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)
        torch.cuda.set_device(args.device)
    
    cuda_available = torch.cuda.is_available()

    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #for baseline latency
    baseline_net = get_model("mbv3_small")
    
    #init super network (ofa) with pretrained weight
    ofa_net  = OFAMobileNetV3_small(
            width_mult=1,
            n_classes=10,
            ks_list=[3, 5, 7],
            expand_ratio_list=[3,4,6],
            depth_list=[2,3,4],
        )
    
    ofa_checkpoint_path = "ofa/checkpoint/mbv3_small_best.pth.tar"
    init = torch.load(ofa_checkpoint_path, map_location="cpu")["state_dict"]
    ofa_net.load_state_dict(init)
    print('The OFA Network is ready.')
    
    #prepare dataset 
    if cuda_available:
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = datasets.CIFAR10(root="./dataset/cifar10/", train=True, transform=train_transform, download=True)
    
        num_samples = len(train_dataset)
        indices = np.random.permutation(num_samples)
        split = num_samples // 50
        
        train_data_loader = DataLoader(
            train_dataset, 
            batch_size=64, 
            sampler=SubsetRandomSampler(indices[:split]),
            shuffle=False)
        
        OFA_batch_size_test = 1000

        data_loader = CIFARDataProvider(
                        test_batch_size=OFA_batch_size_test,
                        n_worker=4,
                        image_size=64,
        ).test
        print('The cifar dataloader is ready.')
    
    # accuracy generator
    accuracy_generator = AccuracyGenerator(ofa_net,train_data_loader, data_loader, device=device)
    print('The accuracy generator is ready!')
    
    
    #init box converter for partition and duplication
    layer_box_converter = InputLayerBoxSpace(input_size=(1, 3, 64, 64), 
                                             crossbar_size=crossbar_size, 
                                             num_crossbar=number_of_crossbars, 
                                             quantize_config=quantize_config,
                                             verbose=args.verbose)
    
    #init packing predictor with packing env -- pack and output cycle/latency
    packer = NetworkPacker(number_of_crossbars, 
                           crossbar_size=crossbar_size, 
                           pack_heuristics=args.pack_heuristic,
                           layer_threshold=args.layer_threshold,
                           find_solution=args.find_solution, 
                           verbose=args.verbose)
    
    print('The packer is ready!')
    
    duplicate_optimizer = None
    if args.duplicate:
        duplicate_optimizer = LPDuplicateOptimizer(args)
    
    
    latency_evaluator = LatencyEvaluator(num_sample=args.num_sample, verbose=args.verbose)
    
    
    #set baseline for latency evaluator
    # layer_boxes = layer_box_converter.process_network(baseline_net, depth_split_factor=1) #no depth split
    # seq = np.random.uniform(low=0.0, high=1.0, size=(len(layer_boxes)))
    # result = packer.placement(layer_boxes, seq, find_solution=True)
    # latency_evaluator.set_baselines(layer_box_converter.layer_box_info_list)
    # latency_evaluator.set_baselines_value(2086840) 
    
    
    constraint = 0
    P = 50  # The size of population in each generation, origin :100
    N = 100  # How many generations of population to be searched,origin :500
    r = 0.25  # The ratio of networks that are used as parents for next generation
    params = {
        'ofa_net': ofa_net,
        'constraint_type': "latency",
        'constraint': constraint,
        'crossbar_size': crossbar_size,
        'number_of_crossbars': number_of_crossbars,
        'mutate_prob': 0.1, # The probability of mutation in evolutionary search
        'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
        'layer_box_converter':layer_box_converter,
        'packer': packer,
        'duplicate_optimizer': duplicate_optimizer, # To use a predefined optimizer.
        "latency_evaluator":latency_evaluator,
        'accuracy_predictor': accuracy_generator, # To use a predefined accuracy_predictor predictor.
        'csv_writer': csv_writer,
        'population_size': P,
        'max_time_budget': N,
        'parent_ratio': r,
        "args": args,
    }
    
   
    #init evaluator accuracy predictor 
    evolution_finder = EvolutionFinder(**params)
    result_lis = []
    start_time = time.time()
    
    if args.evolute_method == "ndga":
        best_list, best_obj = evolution_finder.run_non_dominated_evolution_search(verbose=True)
    # else:
    #     best_valids, best_info = evolution_finder.run_evolution_search(verbose=True)
        
    #     result_lis.append(best_info)
    #     acc, net_config, latency_ratio = best_info
    #     logging.info('Final architecture config:')
    #     logging.info(net_config)
    #     logging.info('Final acc: '+str(acc))
    #     logging.info('Final latency ratio: '+str(latency_ratio))
    #     logging.info("Final score"+ best_valids)
        
    #     ofa_net.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
    #     logging.info('Architecture of the searched sub-net:')
    #     logging.info(ofa_net.module_str)
        
    end_time = time.time()
    print(end_time-start_time)
    
    

if __name__ == "__main__":
    test_SimConfig_path = os.path.join(os.getcwd(), "SimConfig.ini")
    args = get_args(test_SimConfig_path)
    main(args)