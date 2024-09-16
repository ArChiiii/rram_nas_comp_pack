# https://github.com/dasvision0212/3D-Bin-Packing-Problem-with-BRKGA

import math
import copy
import random
import numpy as np
import pandas as pd
from rram_nas_comp_pack.packing.Crossbar import Crossbar
from rram_nas_comp_pack.box_converter.info.LayerBox import LayerBox 
from functools import reduce
from typing import NamedTuple, List
import logging


INFEASIBLE = 100000

class PackerOutput(NamedTuple):
    utilization: int
    final_num_crossbar: int
    left_over: int

''' Basic 
MS = [(0,0,0), (10,20,30)]  # [min coordinats, max coordinates]
EMS Empty Mximal-Space

'''

class NetworkPacker():
    def __init__(self, num_crossbar, crossbar_size=(128,128), pack_heuristics=True, layer_threshold=1, xbar_max_load=6, find_solution=False, pack_writer=None, verbose=False):
        
        self.crossbar_size = crossbar_size #(128,128)
        self.xbar_max_load = xbar_max_load
        self.crossbars = [Crossbar(self.crossbar_size) for i in range(num_crossbar)]
        
        # even infeasible solutions are possible
        self.find_solution = find_solution
        self.crossbars_constraint = len(self.crossbars)
        
         #random sequence of boxes to pack
        self.num_opend_crossbars = 1
        
        self.pack_heuristics=pack_heuristics
        self.layer_threshold = layer_threshold
        
        self.pack_writer = pack_writer
        self.verbose = verbose
       
        self.infeasible = False
    
    def reset(self):
        self.crossbars = [Crossbar(self.crossbar_size) for i in range(self.crossbars_constraint)]
        self.num_opend_crossbars = 1
        self.infeasible = False
        
        
    def placement(self, layer_boxes, packing_squence, find_solution=None, packer_write_extra=None):
        
        if self.pack_writer:
            if packer_write_extra:
                self.pack_writer.set_writer(extra = packer_write_extra)
            else:
                self.pack_writer.set_writer()
        
        self.packing_squence = np.argsort(packing_squence)
        if self.verbose:
            print('------------------------------------------------------------------')
            print('|   Packing Procedure')
            print('|    -> Boxes:', layer_boxes)
            print('|    -> Box Packing Sequence:', self.packing_squence)
            print('-------------------------------------------------------------------')
        
        items_sorted = [layer_boxes[i] for i in self.packing_squence]

        # Layer box Selection
        for i, layer_box in enumerate(items_sorted):
            if self.verbose:
                print('Select layer_box:', layer_box)
                
            # Crossbar and EMS selection
            selected_crossbar = None
            selected_EMS = None
            for k in range(self.num_opend_crossbars):
                # select EMS using DFTRC-2
                EMS = self.DFTRC_2(layer_box, k)

                # update selection if "packable"
                if EMS != None:
                    #check if there is layer_box with same exc_order in the same crossbar
                    if self.pack_heuristics:
                        if self.crossbars[k].is_layer_conflict(layer_box, self.layer_threshold) or self.crossbars[k].is_max_loaded(self.xbar_max_load):
                            continue
                        
                    selected_crossbar = k
                    selected_EMS = EMS
                    break
            
            # Open new empty bin
            if selected_crossbar == None:
                if self.num_opend_crossbars +1 > len(self.crossbars):
                    self.infeasible = True
                    if self.verbose:
                        print('No more crossbars . [Infeasible]')
                    
                    if self.find_solution or find_solution:
                        #add new crossbar
                        self.crossbars.append(Crossbar(self.crossbar_size))
                    else:
                        return None
                
                self.num_opend_crossbars += 1
                selected_crossbar = self.num_opend_crossbars - 1
                selected_EMS = self.crossbars[selected_crossbar].EMSs[0] # origin of the new bin
                if self.verbose:
                    print('No available bin... open crossbar', selected_crossbar)
            
            if self.verbose:
                print('Select EMS:', list(map(tuple, selected_EMS)))
                
            # elimination rule for different process
            min_area, min_dim = self.elimination_rule(items_sorted[i+1:])
            
            # pack the layer_box to the bin & update state information
            self.crossbars[selected_crossbar].update(layer_box, selected_EMS, min_area, min_dim)
            #update layer_box location info for evaluation and vis 
            layer_box.set_location((selected_crossbar, selected_EMS[0][0], selected_EMS[0][1]))
            if self.pack_writer:
                self.pack_writer.write_row(layer_box.to_log())
            
            if self.verbose:
                print('Add layer_box to crossbar',selected_crossbar)
                print(' -> EMSs:',self.crossbars[selected_crossbar].get_EMSs())
                print('------------------------------------------------------------')
        
        boxes_area = reduce(lambda x,y: x + y.area, layer_boxes, 0)
        container_area = np.prod(self.crossbar_size) * len(self.crossbars)
        utilization =  boxes_area / container_area

        if self.verbose:
            logging.info('|')
            logging.info(f'|     Number of box: {len(layer_boxes)}')
            logging.info(f'|     Crossbars constraint: {self.crossbars_constraint}')
            logging.info(f'|     Number of used crossbars: {self.num_opend_crossbars}')
            logging.info(f'|     Left over area: {container_area - boxes_area}')
            logging.info(f'|     Utilization: {utilization}')
            logging.info('|')
            logging.info('------------------------------------------------------------')

        return PackerOutput(utilization, self.num_opend_crossbars, left_over=container_area - boxes_area)
    
    
    # Distance to the Front-Top-Right Corner
    def DFTRC_2(self, box: LayerBox, crossbar_idx:int):
        maxDist = -1
        selectedEMS = None

        for EMS in self.crossbars[crossbar_idx].EMSs:
            W, H = self.crossbars[crossbar_idx].dimensions
            w, h = box.size
            if self.fitin((w, h), EMS):
                x, y = EMS[0]
                distance = pow(H-x-h, 2) + pow(W-y-w, 2)

                if distance > maxDist:
                    maxDist = distance
                    selectedEMS = EMS
        return selectedEMS
    
    def fitin(self, box, EMS):
        # all dimension fit
        for d in range(2):
            if box[d] > EMS[1][d] - EMS[0][d]:
                return False
        return True
    
    def elimination_rule(self, remaining_boxes):
        if len(remaining_boxes) == 0:
            return 0, 0
        
        min_area = 999999999
        min_dim = 9999
        for box in remaining_boxes:
            # minimum dimension
            dim = np.min(box.size)
            if dim < min_dim:
                min_dim = dim
                
            # minimum area
            if box.area < min_area:
                min_area = box.area
        return min_area, min_dim
    
    def evaluate(self):
        if self.infisible:
            return INFEASIBLE
        
        leastLoad = 1
        for k in range(self.num_opend_crossbars):
            load = self.crossbars[k].load()
            if load < leastLoad:
                leastLoad = load
        return self.num_opend_crossbars + leastLoad%1
    