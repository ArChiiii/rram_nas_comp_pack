import simpy
import copy
from typing import NamedTuple, List
from rram_nas_comp_pack.box_converter.info.layer_box_info import LayerBoxInfo
from rram_nas_comp_pack.evaluation.sim_utils import LocationResource, NetworkLayerResource, SimOperation
from rram_nas_comp_pack.evaluation.network_sim_utlis import neural_network_layer, CrossbarResourceStorePool, NetworkLayerResourceStorePool

import logging

import logging 

sim_logger = logging.getLogger("SIM")

class LatencyOutput(NamedTuple):
    latency: float
    ideal_latency: float
    baseline_latency: float
    ratio: float
    baseline_ratio: float


class LatencyEvaluator():
    def __init__(self, RRAM_sim=None, num_sample = 64, verbose=False):
        self.RRAM_sim = RRAM_sim
        self.num_sample = num_sample
        self.verbose = verbose
        self.reset()
    
    def reset(self):
        self.env = simpy.Environment()
        self.location_res= LocationResource(self.env)
        self.network_layer_res = NetworkLayerResource(self.env)

    def set_baselines_value(self, baseline_latency):
        self.baseline_latency = baseline_latency
        
    def set_baselines(self, layer_box_info_list: list[LayerBoxInfo], inprecision=8):
        
        self.reset()
        self._prepare_layer_store(layer_box_info_list, inprecision)     
        latency = self.process_env(self._cal_latency(layer_box_info_list))
        
        self.baseline_latency = latency
        if self.verbose:
            print('|')
            print(f'|     baseline_latency: {latency}')
            print('|')
            print('------------------------------------------------------------')
    
    def _prepare_layer_store(self, layer_box_info_list, inprecision):
        #create layer resource 
        for layer_box_info in layer_box_info_list:
            store = self.network_layer_res.create_network_store(layer_box_info.exc_order, layer_box_info.multiply)
           
            for box_operations in layer_box_info.child_boxes:
                operation_list = []
                #order the operation based on the cycle time asc
                box_operations = sorted(box_operations, key=lambda x: x.cycle)
                for box in box_operations:
                    operation = SimOperation(self.env, id=f"module_{box.exc_order}", cycle=box.latency(inprecision), crossbar_idx=box.crossbar_idx,exc_order=box.exc_order)
                    operation_list.append(operation)
                store.put(operation_list)
    
    def cal_ratio(self, layer_box_info_list: list[LayerBoxInfo], inprecision=8, mode="latency"):
        
        sim_logger.debug("Start latency evaluation")
        self.reset()
        # self._prepare_layer_store(layer_box_info_list, inprecision)     
        latency = self.process_env(self._cal_net_latency(layer_box_info_list, mode=mode))
        
        sim_logger.debug("Start parallel latency evaluation")
        self.reset()
        # self._prepare_layer_store(layer_box_info_list, inprecision)
        ideal_latency = self.process_env(self._cal_net_latency(layer_box_info_list, parallel=True))
        
        if not hasattr(self, 'baseline_latency'):
            self.baseline_latency = ideal_latency
        
        if self.verbose:
            logging.info(f"latency: {latency}, ideal_latency: {ideal_latency}")
            logging.info('|')
            logging.info(f'|     Latency: {latency}')
            logging.info(f'|     Ideal_latency: {ideal_latency}')
            logging.info(f'|     Latency ratio (// pack): {ideal_latency / latency}')
            logging.info(f'|     Latency ratio (baseline): {self.baseline_latency / latency}')
            logging.info('|')
            logging.info('------------------------------------------------------------')

        return LatencyOutput(latency, 
                             ideal_latency, 
                             baseline_latency=self.baseline_latency,
                             ratio=ideal_latency / latency,
                             baseline_ratio=self.baseline_latency / latency)
    
    def process_env(self, generator_fn):
        self.env.process(generator_fn)
        self.env.run()
        return self.env.now

    def _cal_latency(self, layer_box_info_list: list[LayerBoxInfo]):
        
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
    
    def _cal_ideal_latency(self, layer_box_info_list: list[LayerBoxInfo]):
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
 
 
    def _cal_net_latency(self, layer_box_info_list: list[LayerBoxInfo], mode="latency", parallel=False):
        crossbar_res = CrossbarResourceStorePool(self.env)
        
        weight_store_pool = NetworkLayerResourceStorePool(self.env, crossbar_res)
        for layer_box_info in layer_box_info_list:
            weight_store = weight_store_pool.create_network_store(layer_box_info.exc_order, layer_box_info.multiply)
            for box in layer_box_info.child_boxes:
                weight_store.put(box)
        
        last_sample_first_layer_process = None
        for _ in range(self.num_sample):
            last_order_bottleneck_process = last_sample_first_layer_process
            # order of execution of the layers
            for idx, layer_box_info in enumerate(layer_box_info_list):
            
                weight_store = weight_store_pool.get_netowrk_res(layer_box_info.exc_order)
                last_order_bottleneck_process = self.env.process(neural_network_layer(self.env, last_order_bottleneck_process, layer_box_info, 
                                                                                      weight_store ,crossbar_res, sample_size=self.num_sample, inprecision=8, mode=mode,
                                                                                      parallel=parallel))
                if idx == 0:
                    last_sample_first_layer_process = last_order_bottleneck_process
                            
         
        yield last_order_bottleneck_process