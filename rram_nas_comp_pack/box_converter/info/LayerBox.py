


import numpy as np

from MNSIM.Hardware_Model.PE import ProcessElement
from MNSIM.Latency_Model.PE_latency import PE_latency_analysis
import os
import numpy as np 
import math

config_path = os.path.join(os.getcwd(),"SimConfig.ini")
PE = ProcessElement(config_path)
weight_bit = math.floor(math.log2(PE.device_level))


'''
exc_order: the order of the layer in the network (1 layer can split into multiple layer boxes)
'''
class LayerBox(object):
    
    
    # class to represent the box
    def __init__(self, h, w, cycle, exc_order, weight_precision, grouped = False):
        self.h = h
        self.w = w
        self.size = (w, h)
        self.cycle = cycle 
        self.exc_order = exc_order
        self.grouped = grouped
        self.latency_analysis = None
        self.power_analysis = None
        self.weight_precision = weight_precision
        self.PE_num = math.ceil(weight_precision / PE.group_num)
        
    @property
    def area(self):
        return np.product(self.size)

    def generate_latency_analysis(self, input_bit=9):
        if self.latency_analysis is None:
            # grouped means depthwise layer, filter process per input channel 
            w = 1 if self.grouped else self.w 
            
            # inprecision = input_bit / PE.group_num
            inprecision = input_bit 
            self.latency_analysis = PE_latency_analysis(os.path.join(os.getcwd(),"SimConfig.ini"),self.h, w, 0, 0, inprecision)
    
    def latency(self, input_bit=8):
        self.generate_latency_analysis(input_bit)
        return self.latency_analysis.PE_latency * self.cycle
    
    def latency_by_cycle(self, input_bit=8):
        self.generate_latency_analysis(input_bit)
        return self.latency_analysis.PE_latency

    def __str__(self):
        return f"LayerBox(h={self.h}, w={self.w}, cycle={self.cycle} @layer{self.exc_order})"
    
    def __repr__(self):
        return f"LayerBox(h={self.h}, w={self.w}, cycle={self.cycle} @layer{self.exc_order})"

    def set_location(self, location):
        self.crossbar_idx = location[0]
        self.location = location[1:]
        
    def max_space(self):
        # y, x, y+h, x+w
        loc = np.array(self.location)
        return np.append(loc,loc+np.array(self.size))
    
    def to_log(self):

        ms = self.max_space()
        return {
            "layer_idx":self.exc_order,
            "crossbar_idx":self.crossbar_idx,
            "loc_x":ms[1],
            "loc_y":ms[0],
            "max_x":ms[3],
            "max_y":ms[2],
        }
        
    
    
    # def to_json(self):
    #     return {
    #         "exc_order":self.exc_order,
    #         "size": self.size,
    #         "crossbar_idx":self.crossbar_idx,
    #         "location": self.location,
    #         "max_space": self.max_space().tolist()
    #     }

if __name__ == "__main__":
    lb = LayerBox(h=100, w=100, cycle=10, exc_order=0)
    rram = PulseInputRRAM(os.path.join(os.getcwd(),"nas_perf_sim/configs","RRAM.yaml"))
    
    latency = lb.latency_sim(rram)
    print(latency)