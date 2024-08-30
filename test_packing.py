
import os 
import numpy as np
import time
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)

# from rram_nas_comp_pack.evaluation.CycleEvaluator import CycleEvaluator
from rram_nas_comp_pack.evaluation.LatencyEvaluator import LatencyEvaluator
from rram_nas_comp_pack.packing import NetworkPacker
from rram_nas_comp_pack.box_converter.InputLayerBoxSpace import InputLayerBoxSpace
from rram_nas_comp_pack.box_converter.models import get_model
from rram_nas_comp_pack.utils.logger import PackingWriter, log_args
from rram_nas_comp_pack.utils.options import get_args 
from rram_nas_comp_pack.evaluation.network_sim_utlis import setup_sim_logging

def setup_logger(args):
    exp_id = args.exp_id
    packing_writer_path = './log/packing/{}'.format(exp_id)
  
    if not os.path.exists(packing_writer_path):
        os.makedirs(packing_writer_path)
    # else: # remove the old log
    #     os.system(f"rm -rf {log_writer_path}/*")

    # writer = SummaryWriter(logdir=log_writer_path)
    file_handler = logging.FileHandler(os.path.join(packing_writer_path, 'logs.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    setup_sim_logging(packing_writer_path)
    
    return packing_writer_path


def main(args):
    packing_writer_path = setup_logger(args)
    log_args(args)
    
    crossbar_size = args.crossbar_size
    number_of_crossbars = args.num_crossbars
    seed = args.seed

    np.random.seed(seed)
    
    quantize_config = {
        'weight_bit': 8,
    }
    
    #setup neural network 
    net = get_model(args.model)
    assert net, "Not supported model"
    
    start = time.time()
    #basic partition based on the size 
    layout_box_space = InputLayerBoxSpace((1, 3, 64, 64), crossbar_size, number_of_crossbars, quantize_config)
    layer_boxes = layout_box_space.process_network(net, depth_split_factor=args.depth_split_factor)
    packable_boxes = layout_box_space.packable_box()
    layout_box_space.save_to_json(packing_writer_path)
    
    end_partition = time.time()

    # random packing sequence
    seq = np.random.uniform(low=0.0, high=1.0, size=(len(layer_boxes)))

    pack_writer = None
    if args.log_pack:
        pack_writer = PackingWriter(dir=packing_writer_path)
    
    packer = NetworkPacker(number_of_crossbars, 
                           crossbar_size=crossbar_size, 
                           pack_heuristics=args.pack_heuristic,
                           layer_threshold=args.layer_threshold,
                           find_solution=args.find_solution, 
                           pack_writer=pack_writer,
                           verbose=args.verbose)
    
    result = packer.placement(layer_boxes, seq)

    assert result, "Infeasible to pack with given constraint"

    end_packing = time.time()
    logging.basicConfig(level=logging.DEBUG)
    
    #evaluation with differnet number of sample 
    for i in range(10):
        sim_logger = logging.getLogger("SIM")
        num_sample = pow(2,i)
        sim_logger.debug("Start evaluation with {} samples".format(num_sample))
        latency_evaluator = LatencyEvaluator(num_sample=num_sample, verbose=True)
        output = latency_evaluator.cal_ratio(layout_box_space.layer_box_info_list)
        
        # writer.add_scalar('Latency_per_sample', output.latency/num_sample , num_sample)
        
    end_evalute = time.time()
    
    logging.info('|')
    logging.info(f'|     Partitioning Time: {end_partition - start}')
    logging.info(f'|     Packing Time: {end_packing - end_partition}')
    logging.info(f'|     Evaluation Time: {end_evalute - end_packing}')
    logging.info(f'|     Total Time: {end_evalute - start}', )
    logging.info('|')
    logging.info('------------------------------------------------------------')

if __name__ == "__main__":
    test_SimConfig_path = os.path.join(os.getcwd(), "SimConfig.ini")
    args = get_args(test_SimConfig_path)
    main(args)