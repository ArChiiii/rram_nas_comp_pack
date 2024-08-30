
import pulp as pl
import numpy as np
import logging 
from typing import  NamedTuple

from rram_nas_comp_pack.box_converter.info.layer_box_info import LayerBoxInfo



class OptimizerOutput(NamedTuple):
    result: list[int]

class LPDuplicateOptimizer():
    def __init__(self, args):
        self.verbose = args.verbose
        self.seed = args.seed
        self.crossbar_size = args.crossbar_size
        self.number_of_crossbars = args.num_crossbars
        # self.margin_factor = args.margin_factor
        self.reset()
        
    def reset(self):
        self.prob = pl.LpProblem("Maximize task per cycle", pl.LpMaximize)
        self.margin_factor_list = [i for i in np.arange(0.95, 0.55, -0.05)]
    
    def next_margin_factor(self):
        self.prob = pl.LpProblem("Maximize task per cycle", pl.LpMaximize)
        if len(self.margin_factor_list) == 0:
            return None
        return self.margin_factor_list.pop(0)
        
    def optimize(self, layer_box_info_list:list[LayerBoxInfo]):
        margin_factor = self.next_margin_factor()
        logging.debug(f"Set margin factor to {margin_factor}")
        
        if not margin_factor:
            return None
        
        area_list = []
        
        # order_list = np.argsort([lbi.task_per_cycle for lbi in layer_box_info_list])
        order_list = np.argsort([lbi.task_per_latency for lbi in layer_box_info_list])
        
        
        #time consuming layer
        time_layer_box_info = layer_box_info_list[order_list[0]]
        # k = pl.LpVariable("k_{}".format(time_layer_box_info.exc_order), cat=pl.LpInteger)
        time_layer_box_info.xtime = pl.LpVariable("xtime_{}".format(time_layer_box_info.exc_order), lowBound=1, cat='Integer')
        area_list.append(time_layer_box_info.xtime * time_layer_box_info.box_raw.area)
        #objective: optimize the task per cycle of the time consuming layer
        # self.prob += time_layer_box_info.xtime == k * time_layer_box_info.output_size[1]
        self.prob += time_layer_box_info.xtime * time_layer_box_info.task_per_cycle
        # self.prob += time_layer_box_info.xtime >= 1
        
        for order_idx in order_list[1:]:
            layer_box_info = layer_box_info_list[order_idx]
            # k = pl.LpVariable("k_{}".format(layer_box_info.exc_order), cat=pl.LpInteger)
            layer_box_info.xtime = pl.LpVariable("xtime_{}".format(layer_box_info.exc_order), lowBound=1,  cat='Integer')
            # fix generality
            # self.prob += layer_box_info.xtime >= 1
            # self.prob += layer_box_info.xtime == k * layer_box_info.output_size[1]
            self.prob += layer_box_info.xtime * layer_box_info.task_per_cycle >= time_layer_box_info.xtime * time_layer_box_info.task_per_cycle
            area_list.append(layer_box_info.xtime * layer_box_info.box_raw.area)
        
        
        #area constraint
        self.prob += pl.lpSum(area_list) <= self.crossbar_size[0]*self.crossbar_size[1]*self.number_of_crossbars * margin_factor
        
        solve_result = self.prob.solve(pl.apis.PULP_CBC_CMD(msg=False))

        result = []

        if solve_result == 1:
            for layer_box_info in layer_box_info_list:
                layer_box_info.lp_duplicate_box()
                result.append(layer_box_info.xtime.varValue)
            if self.verbose:
                logging.debug(f"Status: {pl.LpStatus[self.prob.status]}", )
                logging.debug("Optimal Configuration:") 
                logging.debug(result)
            return OptimizerOutput(result)
        else:
            if self.verbose:
                logging.debug(f"Status: {pl.LpStatus[self.prob.status]}", )
            return None

        