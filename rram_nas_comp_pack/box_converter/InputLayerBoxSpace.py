
from torchinfo import summary
from torch import nn

from rram_nas_comp_pack.box_converter.info.LayerBox import LayerBox
from .info.layer_box_info import LayerBoxInfo
from typing import Tuple
import numpy as np
import json
import os 

# converter class for converting the network to a box

class InputLayerBoxSpace:
    def __init__(self, input_size:Tuple[int, int,int, int], crossbar_size:Tuple[int, int], num_crossbar:int, quantize_config, verbose=False):
        self.input_size = input_size #(1, 3, 32, 32)
        self.crossbar_size = crossbar_size #(128, 128)
        self.num_crossbar = num_crossbar
        self.box_converter = LayerBoxConverter(input_size, quantize_config)
        self.quantize_config = quantize_config

    def process_network(self, net, depth_split_factor=1):
        self.net = net
        output = self._convert(depth_split_factor)
        return output
    
    def _convert(self, depth_split_factor):
        output_box = []
        self.layer_box_info_list = self.box_converter.convert(self.net)
        
        for box_info in self.layer_box_info_list:
            output_box+=box_info.split_box(self.crossbar_size, depth_split_factor=depth_split_factor)
    
        return output_box

    def save_to_json(self, path, extra=None):
        def schema(o):
            if isinstance(o, LayerBox):
                return str(o)
                        
            return {
                "idx":o.class_name +"_"+ str(o.exc_order),
                "multiply":o.multiply,
                "box_raw":o.box_raw,
                "boxes":[box for box in o.boxes],
                "child_boxes":[box for box in o.child_boxes],
            }
        filename = "partition.json"
        if extra:
            filename = f"partition_{extra}.json"
        
        with open(os.path.join(path, filename), 'w') as f:
            json.dump(self.layer_box_info_list, 
                       default=schema,
                       indent=4,
                        fp=f)

    def packable_box(self):
        packable_boxes = []
        for box_info in self.layer_box_info_list:
            packable_boxes+=box_info.get_packable_boxes(self.crossbar_size)

        return packable_boxes

    def get_layer_child_box(self):
        child_boxes = []
        for box_info in self.layer_box_info_list:
            child_boxes = sum(box_info.child_boxes, child_boxes)
        return child_boxes
    
    

class LayerBoxConverter(object):
    def __init__(self, input_size:Tuple[int, int, int, int] = (1, 3, 64, 64), quantize_config=None):
        self.input_size = input_size 
        self.quantize_config = quantize_config
        
    def convert(self, net: nn.Module):
        summary_info = summary(net, self.input_size, verbose=0).summary_list
        
        filter_summary_info = [n for n in summary_info 
                               if n.class_name == "Conv2d" 
                                    or n.class_name == "Linear"]
        boxes_info = []

        for idx, n in enumerate(filter_summary_info):
            box_info = LayerBoxInfo(n, idx, self.quantize_config)
            if box_info.box_raw:
                boxes_info.append(box_info)

        return boxes_info