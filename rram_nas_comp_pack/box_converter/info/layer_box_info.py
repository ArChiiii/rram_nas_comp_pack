

import copy
import math
from .layer_info import LayerInfo

from .utilis import Conv2dBoxInfo, LinearBoxInfo
from rram_nas_comp_pack.box_converter.info.LayerBox import LayerBox
from typing import Tuple
import json
import numpy as np 

BOX_INFO_MAPPING = {
    "Conv2d": Conv2dBoxInfo,
    "Linear": LinearBoxInfo,
}



class LayerBoxInfo():
    #init base on LayerInfo
    def __init__(self, layer_info: LayerInfo, exc_order:int, quantize_config):
        self.layer_info = layer_info
        self.exc_order = exc_order
        self.module = layer_info.module
        self.class_name = layer_info.class_name
        self.quantize_config = quantize_config
        self.boxes:list[LayerBox] = []
        #get the box info
        self.box_raw:LayerBox = self.get_box_info()
        self._child_boxes:list[LayerBox] = []
        self.temp_child_boxes = []
        
        if layer_info.class_name == "Linear":
            self.output_size = layer_info.output_size
        elif layer_info.class_name == "Conv2d":
            self.output_size = layer_info.output_size[2:4] #no batch and channel dim 
            self.groups = layer_info.module.groups

    
    def get_box_info(self):
        #base on the layer info, generate the box info
        layer_type = self.layer_info.class_name
        if layer_type not in BOX_INFO_MAPPING:
            return None
            # raise ValueError(f"Layer type {layer_type} not supported")

        return BOX_INFO_MAPPING[layer_type](self.layer_info,self.exc_order, self.quantize_config['weight_bit'])
        

    def split_box(self, crossbar_size:Tuple[int, int], depth_split_factor:int ):
        # #split the box into multiple boxes if oversize 
        box = self.box_raw
        h = box.h
        w = box.w
        cycle = box.cycle
        
        h_partition = h
        w_partition = w
        #split base on bank size
        if box.h > crossbar_size[0]:
            h_partition = crossbar_size[0]
        if box.w > crossbar_size[1]:
            w_partition = crossbar_size[1]
    
        # depthwise layer
        if self.class_name == "Conv2d" and self.module.groups!= 1: 
            output_cycle = cycle // w
            
            h_partition = h
            
            #split the depth
            if depth_split_factor > 1: 
                if w < depth_split_factor:
                    w_partition = 1
                else:
                    w_partition = math.ceil(w // depth_split_factor)

            h_factor = h // h_partition
            w_factor = w // w_partition
            cycle = cycle // w_factor #parallel in RRAM
            
            for i in range(h_factor):
                for j in range(w_factor):
                    self.boxes.append(LayerBox(h_partition, w_partition, w_partition*output_cycle, self.exc_order, self.quantize_config['weight_bit'], grouped=True))

            h_remain = h % h_partition
            w_remain = w % w_partition
            if h_remain > 0 and w_remain > 0:
                self.boxes.append(LayerBox(h_remain, w_remain, w_remain*output_cycle, self.exc_order, self.quantize_config['weight_bit'], grouped=True))
            if h_remain > 0:
                for i in range(w_factor):
                    self.boxes.append(LayerBox(h_remain, w_partition, w_partition*output_cycle, self.exc_order, self.quantize_config['weight_bit'], grouped=True))
            if w_remain > 0:
                for i in range(h_factor):
                    self.boxes.append(LayerBox(h_partition, w_remain, w_remain*output_cycle, self.exc_order, self.quantize_config['weight_bit'], grouped=True))
            
            
        else:
            h_factor = h // h_partition
            w_factor = w // w_partition
        
            for i in range(h_factor):
                for j in range(w_factor):
                    self.boxes.append(LayerBox(h_partition, w_partition, cycle, self.exc_order, self.quantize_config['weight_bit']))

            h_remain = h % h_partition
            w_remain = w % w_partition
            if h_remain > 0 and w_remain > 0:
                self.boxes.append(LayerBox(h_remain, w_remain, cycle, self.exc_order, self.quantize_config['weight_bit']))
            if h_remain > 0:
                for i in range(w_factor):
                    self.boxes.append(LayerBox(h_remain,w_partition, cycle, self.exc_order, self.quantize_config['weight_bit']))
            if w_remain > 0:
                for i in range(h_factor):
                    self.boxes.append(LayerBox(h_partition, w_remain, cycle, self.exc_order, self.quantize_config['weight_bit']))

        return self.boxes
    
    def get_packable_boxes(self, crossbar_size:Tuple[int, int]):
        #return the boxes that is not same size as the bank size
        assert len(self.boxes) > 0, "No boxes to pack"
        self.pack_boxes = [box for box in self.boxes if box.h != crossbar_size[0] or box.w != crossbar_size[1]]
        
        return self.pack_boxes

    @property
    def multiply (self):
        o = self.xtime.varValue if hasattr(self, 'xtime') else 1
        if o <= 0:
            o = 1
        return o

    @multiply.setter
    def multiply (self, value):
        self.xtime.varValue = value
    
    def reset_multiply (self):
        if hasattr(self, 'xtime'):
            delattr(self, 'xtime')
    
    @property
    def child_boxes(self):
        if len(self._child_boxes) == 0:
            self._child_boxes = [self.boxes]
        return self._child_boxes

    @child_boxes.setter
    def child_boxes(self, value):
        self._child_boxes = value

    @property
    def latency(self):
        assert len(self.boxes) > 0, "No boxes to pack"
        max_latency = max([box.latency() for box in self.child_boxes[0]])
        return max_latency

    @property
    def task_per_cycle(self):
        return round(1 / self.box_raw.cycle, 8)
    
    @property
    def task_per_latency(self):
        return round(1 / self.box_raw.latency(self.quantize_config['weight_bit']), 8)
    
    @property
    def cycle(self):
        assert len(self.boxes) > 0, "No boxes to pack"
        max_cycle = max([box.cycle for box in self.child_boxes[0]])
        return max_cycle

    def gen_temp_child_boxes(self):
        self.temp_child_boxes = copy.deepcopy(self.child_boxes)
        return self.temp_child_boxes

    def update_child_boxes(self):
        self.child_boxes = self.temp_child_boxes
    
    def lp_duplicate_box(self):
        
        def create_group(box_list, num_boxes, cycle_value):
            group_box = []
            for _ in range(num_boxes):
                group_box.extend(copy.deepcopy(box_list))
            for box in group_box:
                box.cycle = cycle_value
            return group_box
        
        
        temp_child_boxes = []
        # if self.groups > 1:
        boxes = copy.deepcopy(self.boxes)
        # for box in boxes:
            # box.cycle =  math.ceil(box.cycle / self.multiply)
        
        temp_child_boxes.extend([copy.deepcopy(boxes) for _ in range(int(self.multiply))])
        # else:
        #     group_multiple = self.box_raw.cycle // self.multiply
        #     rest_box = self.box_raw.cycle % self.multiply
            
        #     if group_multiple > 0:
        #         group_box = create_group(self.boxes, int(self.multiply), 1)
        #         for _ in range(int(group_multiple)):
        #             self.temp_child_boxes.append(group_box)
            
        #     reset_group_box = create_group(self.boxes, int(rest_box), math.ceil(self.box_raw.cycle / rest_box))    
        #     self.temp_child_boxes.append(reset_group_box)
    
        self.child_boxes = self.temp_child_boxes = temp_child_boxes
        return self.temp_child_boxes
    
    def temp_duplicate_box(self, times:int=1):
        multiply = self.multiply + times
        # multiply = pow(2, self.multiply-1 + 1)
        boxes = copy.deepcopy(self.boxes)
        for box in boxes:
            box.cycle =  math.ceil(box.cycle / multiply)
        self.temp_child_boxes =  [copy.deepcopy(boxes) for _ in range(multiply)]
        return self.temp_child_boxes
    
    def duplicate_box(self):
        self.multiply += 1
        # self.multiply = pow(2, self.multiply-1 + 1)
