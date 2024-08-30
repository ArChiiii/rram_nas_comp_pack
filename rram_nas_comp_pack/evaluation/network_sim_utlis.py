import simpy
from simpy.resources.resource import Request
from simpy.resources.store import StoreGet
import functools

from rram_nas_comp_pack.box_converter.info.layer_box_info import LayerBoxInfo
import os
import logging 
import math

sim_logger = logging.getLogger("SIM")

def setup_sim_logging(packing_writer_path):
    FORMAT = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(FORMAT)
    file_handler = logging.FileHandler(os.path.join(packing_writer_path, 'sim.log'), mode='w')
    file_handler.setFormatter(FORMAT)
    sim_logger.setLevel(logging.DEBUG)
    # sim_logger.addHandler(console_handler)
    sim_logger.addHandler(file_handler)

class NetworkLayerResourceStorePool():
    def __init__(self,env, crossbar_res_pool) -> None:
        self.env=env
        self.crossbar_res_pool = crossbar_res_pool
        self.store_set = {}
    
    def create_network_store(self, exc_order, multiply=1):
        store = simpy.Store(self.env, capacity=multiply)
        self.store_set[exc_order] = store
        return store
        
    def get_netowrk_res(self, exc_order):
        return self.store_set.get(exc_order)


class NetworkLayerWeight(simpy.Store):
    def __init__(self, env, capacity, crossbar_res):
        
        super().__init__(env, capacity=capacity)
        self.crossbar_res = crossbar_res
        
    def request_crossbar_res(self, crossbar_idx) -> Request:
        return self.crossbar_res.get_crossbar_res(crossbar_idx)
        
    # def request(self):
    #     layer_box = self.get()
    #     with self.request_crossbar_res(layer_box.crossbar_res).request():
    #         return simpy.Resource.request(), layer_box
    
    def get(self) :
        layer_box = yield super().get()
        with self.request_crossbar_res(layer_box.crossbar_idx).request():
            get_event = self._trigger_get(None)
            get_event.succeed(layer_box)
            return get_event
    
    
class CrossbarResourceStorePool():
    def __init__(self,env) -> None:
        self.env=env
        self.res_set = {}

    def get_crossbar_res(self,crossbar_idx):
        crossbar_res = self.res_set.get(crossbar_idx)
        if not crossbar_res:
            crossbar_res = simpy.Resource(self.env, capacity=1)
            self.res_set[crossbar_idx] = crossbar_res
        
        return crossbar_res



def request_layer_weight_process(env, crossbar_res, layer_weight_res: simpy.Store ,num_required:int, compute_process, parallel=False):
    requests = [layer_weight_res.get() for _ in range(int(num_required))]
    # layer_boxes = yield simpy.AllOf(env, requests)   
    layer_boxes_results = yield simpy.AllOf(env, requests)   
    result_list=[]
    for req in layer_boxes_results:
        result_list.append(layer_boxes_results[req])
        
    flatten_result_list = [item for sublist in result_list for item in sublist]
    
    
    # Request crossbars for the layer weights
    # print(f'[{env.now}] Requesting crossbars for the layer weights')
    if parallel:
        yield env.process(compute_process(flatten_result_list))
    else:
        yield env.process(request_crossbars(env, crossbar_res, flatten_result_list, compute_process))
    # print(f'[{env.now}] Crossbars requested')
    
    
    for layer_box in result_list:
        yield layer_weight_res.put(layer_box)
        
    
def request_crossbars_old(env, location_res, flatten_result_list, compute_process):
    # print(f'[{env.now}] Starting request_crossbars')
    resources = [location_res.get_crossbar_res(layer_box.crossbar_idx) for layer_box in flatten_result_list]

    # requests = [resource.request(priority=-layer_box.exc_order) for resource, layer_box in zip(resources, flatten_result_list)]
    
    # print(f'[{env.now}] Waiting for crossbars: {requests}')
    # 
    # results = yield simpy.AllOf(env, requests)   
    # print(f'[{env.now}] Obtained crossbars for layer boxes: {flatten_result_list}')

    #main process 
    # yield env.process(compute_process(flatten_result_list))
    
    for resource, layer_box in zip(resources, flatten_result_list):
        # requests
        with resource.request(priority=-layer_box.exc_order) as req:
            yield req
            #main process 
            yield env.process(compute_process(layer_box))
    
    # print(f"{env.now}: released all resources")
        

def request_crossbars(env, location_res, flatten_result_list, compute_process):
    # print(f'[{env.now}] Starting request_crossbars')
    different_layer_boxes = [[]]
    different_crossbar_resources = [[]]
    
    
    for layer_box in flatten_result_list:
        crossbar_res = location_res.get_crossbar_res(layer_box.crossbar_idx)
        grouped = False
        for grouped_crossbar_res, grouped_layer_boxs in zip(different_crossbar_resources, different_layer_boxes):
            if crossbar_res not in grouped_crossbar_res:
                grouped_crossbar_res.append(crossbar_res)
                grouped_layer_boxs.append(layer_box)
                grouped = True
                break
        
        if not grouped:
            different_crossbar_resources.append([crossbar_res])
            different_layer_boxes.append([layer_box])

    
    # requests = [resource.request(priority=-layer_box.exc_order) for resource, layer_box in zip(different_crossbar_resources, different_layer_boxes)]
    
    # print(f'[{env.now}] Waiting for crossbars: {requests}')
    # 
    # results = yield simpy.AllOf(env, requests)   
    # print(f'[{env.now}] Obtained crossbars for layer boxes: {flatten_result_list}')

    #main process 
    # yield env.process(compute_process(different_layer_boxes))
    
    # for resource, req in zip(different_crossbar_resources, requests):
    #     resource.release(req)
    
    # print(f"{env.now}: released all resources")
    
    for resource, layer_boxs in zip(different_crossbar_resources, different_layer_boxes):
        requests = [resource.request() for resource, layer_box in zip(resource, layer_boxs)]
        # sim_logger.debug(f"Request list: {[l.exc_order for l in layer_boxs]}")
        results = yield simpy.AllOf(env, requests)   
        # sim_logger.debug(f"resources available")
        yield env.process(compute_process(layer_boxs))
        
        for resource, req in zip(resource, requests):
            resource.release(req)


def neural_network_layer(env, process, layer_box_info:LayerBoxInfo, weight_store, crossbar_res, sample_size, inprecision=8, mode="latency", parallel=False):
    
    def main_process(layer_boxes):
        
        crossbar_idx_list = [] 
        for box in layer_boxes:
            crossbar_idx_list.append(str(box.crossbar_idx))
        
        sim_logger.debug(f"Layer {layer_box_info.exc_order} start at time {env.now} with {num_required}x weights @ crossbars {','.join(crossbar_idx_list)}")
        
        box_list = sorted(layer_boxes, key=lambda x: x.latency(inprecision), reverse=True)
        # time = box_list[0].latency(inprecision) // num_required
        time = box_list[0].latency_by_cycle(inprecision) *  math.ceil(box_list[0].cycle // num_required)
        yield env.timeout(time) 
        sim_logger.debug(f"Layer {layer_box_info.exc_order} finished using {','.join(crossbar_idx_list)} at time {env.now}")
    
    def main_process_single(layer_box):
        sim_logger.debug(f"Layer {layer_box_info.exc_order} start at time {env.now} with {num_required}x weights @ crossbars {layer_box.crossbar_idx}")
        time = layer_box.latency(inprecision) // num_required
        yield env.timeout(time) 
        sim_logger.debug(f"Layer {layer_box_info.exc_order} finished using {layer_box.crossbar_idx} at time {env.now}")
    
    
    if process:
        yield process
    # Select the appropriate type of crossbars from the pool
    # crossbar = crossbar_pool[layer_id % len(crossbar_pool)]
    if mode=="latency":
        if layer_box_info.layer_info.class_name == "Linear":
            num_required = min(weight_store.capacity, layer_box_info.output_size[0]*layer_box_info.output_size[1])
        elif layer_box_info.layer_info.class_name == "Conv2d":
            num_required = min(weight_store.capacity, layer_box_info.output_size[0]*layer_box_info.output_size[1] * layer_box_info.groups)
            
    # elif mode=="throughput":
        # num_required = min(1, layer_box_info.output_size[1])  # Adjust this based on your requirement
        # if weight_store.capacity < sample_size:
        #     num_required = min(weight_store.capacity, layer_box_info.output_size[1])
        # else:
        #     num_required = weight_store.capacity // sample_size
    # num_required = min(weight_store.capacity, layer_box_info.output_size[1])  # Adjust this based on your requirement
    
    sim_logger.debug(f"Getting layer {layer_box_info.exc_order} weight at time {env.now}")
    yield env.process(request_layer_weight_process(env, crossbar_res, weight_store, num_required, main_process, parallel))

    



