import simpy
import copy
from typing import NamedTuple, List
from rram_nas_comp_pack.box_converter.info.layer_box_info import LayerBoxInfo
from rram_nas_comp_pack.evaluation.sim_utils import LocationResource, NetworkLayerResource, SimOperation
import logging

class CycleOutput(NamedTuple):
    cycle: int
    ideal_cycle: int
    ratio: float
    baseline_ratio: float


class CycleEvaluator():
    def __init__(self, num_sample = 64, verbose=False):
        self.num_sample = num_sample
        self.verbose = verbose
        self.reset()
    
    def reset(self):
        self.env = simpy.Environment()
        self.location_res= LocationResource(self.env)
        self.network_layer_res = NetworkLayerResource(self.env)

    def set_baselines(self, layer_box_info_list: list[LayerBoxInfo]):
        
        self.reset()
        self._prepare_layer_store(layer_box_info_list)     
        cycle = self.process_env(self._cal_cycle(layer_box_info_list))
        
        self.baseline_cycle = cycle
        if self.verbose:
            print('|')
            print(f'|     baseline_cycle: {cycle}')
            print('|')
            print('------------------------------------------------------------')
    
    def _prepare_layer_store(self, layer_box_info_list):
        #create layer resource 
        for layer_box_info in layer_box_info_list:
            store = self.network_layer_res.create_network_store(layer_box_info.exc_order, layer_box_info.multiply)
           
            for box_operations in layer_box_info.child_boxes:
                operation_list = []
                #order the operation based on the cycle time asc
                box_operations = sorted(box_operations, key=lambda x: x.cycle)
                for box in box_operations:
                    operation = SimOperation(self.env, id=f"module_{box.exc_order}", cycle=box.cycle, crossbar_idx=box.crossbar_idx,exc_order=box.exc_order)
                    operation_list.append(operation)
                store.put(operation_list)
    
    def cal_ratio(self, layer_box_info_list: list[LayerBoxInfo]):
        
        self.reset()
        self._prepare_layer_store(layer_box_info_list)     
        cycle = self.process_env(self._cal_cycle(layer_box_info_list))
        
        self.reset()
        self._prepare_layer_store(layer_box_info_list)
        ideal_cycle = self.process_env(self._cal_ideal_cycle(layer_box_info_list))
        
        if not hasattr(self, 'baseline_cycle'):
            self.baseline_cycle = ideal_cycle
        
        
        logging.info(f"cycle: {cycle}, ideal_cycle: {ideal_cycle}")
        logging.info('|')
        logging.info(f'|     Cycle: {cycle}')
        logging.info(f'|     Ideal_cycle: {ideal_cycle}')
        logging.info(f'|     Cycle ratio (// pack): {ideal_cycle / cycle}')
        logging.info(f'|     Cycle ratio (baseline): {self.baseline_cycle / cycle}')
        logging.info('|')
        logging.info('------------------------------------------------------------')

        return CycleOutput(cycle, 
                             ideal_cycle, 
                             baseline_cycle=self.baseline_cycle,
                             ratio=ideal_cycle / cycle,
                             baseline_ratio=self.baseline_cycle / cycle)
    
    def process_env(self, generator_fn):
        self.env.process(generator_fn)
        self.env.run()
        return self.env.now

    def _cal_cycle(self, layer_box_info_list: list[LayerBoxInfo]):
        
        #number of samples to process
        for _ in range(self.num_sample):
            last_order_bottleneck_process = None
            
            # order of execution of the layers
            for layer_box_info in layer_box_info_list:
                store = self.network_layer_res.get_netowrk_res(layer_box_info.exc_order)
                
                operations = yield store.get()
                for operation in operations:
                    new_temp_bottleneck_process = self.env.process(operation.forward(last_order_bottleneck_process, location_res=self.location_res, verbose=self.verbose))
                    last_temp_bottleneck_process = new_temp_bottleneck_process

                yield store.put(operations)

                last_order_bottleneck_process = last_temp_bottleneck_process
    
    def _cal_ideal_cycle(self, layer_box_info_list: list[LayerBoxInfo]):
        #number of samples to process
        for _ in range(self.num_sample):
            last_order_bottleneck_process = None
            
            # order of execution of the layers
            for layer_box_info in layer_box_info_list:
                store = self.network_layer_res.get_netowrk_res(layer_box_info.exc_order)
                
                operations = yield store.get()
                for operation in operations:
                    new_temp_bottleneck_process = self.env.process(operation.forward(last_order_bottleneck_process, location_res=None, verbose=self.verbose))
                    last_temp_bottleneck_process = new_temp_bottleneck_process

                yield store.put(operations)

                last_order_bottleneck_process = last_temp_bottleneck_process   
 