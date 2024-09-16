from typing import  NamedTuple
import copy
from functools import reduce
import logging
import random
from tqdm import tqdm
import numpy as np
import torch
import os

from rram_nas_comp_pack.utils.logger import PackingWriter

__all__ = ["EvolutionFinder"]


class ArchManager:
    def __init__(self):
        self.num_blocks = 20
        self.num_stages = 4
        self.kernel_sizes = [3, 5, 7]
        self.expand_ratios = [3, 4, 6]
        self.depths = [2, 3, 4]
        # self.resolutions = [160, 176, 192, 208, 224]

    def random_sample(self):
        sample = {}
        d = []
        e = []
        ks = []
        for i in range(self.num_stages):
            d.append(random.choice(self.depths))

        for i in range(self.num_blocks):
            e.append(random.choice(self.expand_ratios))
            ks.append(random.choice(self.kernel_sizes))

        sample = {
            "wid": None,
            "ks": ks,
            "e": e,
            "d": d,
            # "r": [random.choice(self.resolutions)],
        }

        return sample

    def random_resample(self, sample, i):
        assert i >= 0 and i < self.num_blocks
        sample["ks"][i] = random.choice(self.kernel_sizes)
        sample["e"][i] = random.choice(self.expand_ratios)

    def random_resample_depth(self, sample, i):
        assert i >= 0 and i < self.num_stages
        sample["d"][i] = random.choice(self.depths)

    # def random_resample_resolution(self, sample):
    #     sample["r"][0] = random.choice(self.resolutions)

class ProcessSampleResult(NamedTuple):
    result: int
    acc: float = None
    latency_ratio: float  = None


class EvolutionFinder:

    #need to change constraint 
    valid_constraint_range = {
        "crossbar": [80, 300],
        "latency": [0, 4]
    }

    def __init__(
        self,
        ofa_net,
        constraint_type,
        constraint,
        crossbar_size,
        number_of_crossbars,
        layer_box_converter,
        packer,
        duplicate_optimizer,
        latency_evaluator,
        accuracy_predictor,
        csv_writer,
        args,
        **kwargs
    ):
        self.args = args
        self.constraint_type = constraint_type
        if not constraint_type in self.valid_constraint_range.keys():
            self.invite_reset_constraint_type()
        self.constraint = constraint
        if not (
            constraint <= self.valid_constraint_range[constraint_type][1]
            and constraint >= self.valid_constraint_range[constraint_type][0]
        ):
            self.invite_reset_constraint()
        
        self.crossbar_size = crossbar_size
        if constraint_type == "crossbar":
            self.parameters_constraint = self.crossbar_size[0] * self.crossbar_size[1] * self.constraint
        else:
            self.parameters_constraint = self.crossbar_size[0] * self.crossbar_size[1] * number_of_crossbars

        self.ofa_net = ofa_net
        self.layer_box_converter = layer_box_converter
        self.packer = packer
        self.duplicate_optimizer = duplicate_optimizer
        self.latency_evaluator = latency_evaluator
        self.accuracy_predictor = accuracy_predictor
        self.arch_manager = ArchManager()
        self.num_blocks = self.arch_manager.num_blocks
        self.num_stages = self.arch_manager.num_stages
        self.csv_writer = csv_writer
        
        self.counter = 0

        self.mutate_prob = kwargs.get("mutate_prob", 0.1)
        self.population_size = kwargs.get("population_size", 100)
        self.max_time_budget = kwargs.get("max_time_budget", 500)
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)
        
    def invite_reset_constraint_type(self):
        print(
            "Invalid constraint type! Please input one of:",
            list(self.valid_constraint_range.keys()),
        )
        new_type = input()
        while new_type not in self.valid_constraint_range.keys():
            print(
                "Invalid constraint type! Please input one of:",
                list(self.valid_constraint_range.keys()),
            )
            new_type = input()
        self.constraint_type = new_type

    def invite_reset_constraint(self):
        print(
            "Invalid constraint_value! Please input an integer in interval: [%d, %d]!"
            % (
                self.valid_constraint_range[self.constraint_type][0],
                self.valid_constraint_range[self.constraint_type][1],
            )
        )

        new_cons = input()
        while (
            (not new_cons.isdigit())
            or (int(new_cons) > self.valid_constraint_range[self.constraint_type][1])
            or (int(new_cons) < self.valid_constraint_range[self.constraint_type][0])
        ):
            print(
                "Invalid constraint_value! Please input an integer in interval: [%d, %d]!"
                % (
                    self.valid_constraint_range[self.constraint_type][0],
                    self.valid_constraint_range[self.constraint_type][1],
                )
            )
            new_cons = input()
        new_cons = int(new_cons)
        self.constraint = new_cons

    def set_constraint(self, new_constraint):
        self.constraint = new_constraint
        
    
    def process_sample(self, sample):
        
        def placement():
            self.packer.reset()
            layer_boxes = self.layer_box_converter.get_layer_child_box()
        
            np.random.seed(self.args.seed)
            seq = np.random.uniform(low=0.0, high=1.0, size=(len(layer_boxes)))
            result = self.packer.placement(layer_boxes, seq)
            logging.debug(f"[Net-{self.counter}] finish packing network")
            logging.debug(f"[Net-{self.counter}] packing result {result}")
            return result
        
        self.counter += 1
        
        self.packer.reset()
        self.latency_evaluator.reset()
        if self.duplicate_optimizer:
            self.duplicate_optimizer.reset()
        
        self.ofa_net.set_active_subnet(ks=sample["ks"], d=sample["d"], e=sample["e"])
        net = self.ofa_net.get_active_subnet()
        logging.debug(f"[Net-{self.counter}] ")        
        logging.debug(sample)        
        
        layer_boxes = self.layer_box_converter.process_network(net, depth_split_factor=self.args.depth_split_factor)
        # the number of parameter is less then the total crossbar (area)
        logging.debug(f"[Net-{self.counter}] finish partition network")        
        
        
        if reduce(lambda x,y: x + y.area, layer_boxes, 0) > self.parameters_constraint:
            logging.debug(f"[Net-{self.counter}] cannot fit into the crossbars")
            return ProcessSampleResult(result = False)
        
        #for LP duplication 
        if self.duplicate_optimizer:
            output = self.duplicate_optimizer.optimize(self.layer_box_converter.layer_box_info_list)
            logging.debug(f"[Net-{self.counter}] finish duplication")
            logging.debug(f"| Duplication result : {output.result}")
            logging.debug(f"---------------------------------------------------------")
        
        placement_result = placement()
        
        if self.duplicate_optimizer:
            while not placement_result:
                if not self.duplicate_optimizer.optimize(self.layer_box_converter.layer_box_info_list):
                    break
                else:
                    placement_result = placement()
        
        if not placement_result:
            logging.debug(f"[Net-{self.counter}] cannot fit into the crossbars")
            #try again with multiply as 1, no duplication
            if self.duplicate_optimizer:
                for lbi in self.layer_box_converter.layer_box_info_list:
                    lbi.reset_multiply()
                
                placement_result = placement()
                logging.debug(f"[Net-{self.counter}] finish packing network")
                
                if not placement_result:
                    logging.debug(f"[Net-{self.counter}] cannot fit into the crossbars")
                    return ProcessSampleResult(result = False)
            else:
                return ProcessSampleResult(result = False)

        logging.debug(f"| Left over area: {placement_result.left_over}")
        logging.debug(f"| utilization: {placement_result.utilization}")
        logging.debug(f"| #crossbar: {placement_result.final_num_crossbar}")
        logging.debug(f"---------------------------------------------------------")
        
        # for naive duplication
        # if self.duplicate_optimizer:
        #     output = self.duplicate_optimizer.optimize(self.layer_box_converter.layer_box_info_list, result.left_over)
        #     logging.debug(f"[Net-{self.counter}] finish duplication")
        #     logging.debug(f"| Left over area: {output.left_over}")
        #     logging.debug(f"| Area improvement: {output.area_improvement}")
        #     logging.debug(f"---------------------------------------------------------")
        
        
        ## accuracy evaluation
        acc = self.accuracy_predictor.predict_accuracy_by_sameple(sample, fine_tune=(not self.args.no_fine_tune))
        logging.debug(f"[Net-{self.counter}] finish evaluating acc")
        logging.debug(f"| Accuracy: {acc}")
        logging.debug(f"---------------------------------------------------------")
        
        
        ## latency evaluation    
        latency_output = self.latency_evaluator.cal_ratio(self.layer_box_converter.layer_box_info_list)
        
        if self.constraint_type == "latency" and latency_output.latency < self.constraint:
            logging.debug(f"[Net-{self.counter}] cannot fit into the latency constraint")
            return ProcessSampleResult(result = False)
        
        logging.debug(f"[Net-{self.counter}] finish evaluating latency")
        logging.debug(f"| Latency: {latency_output.latency}")
        logging.debug(f"| Baseline: {latency_output.baseline_ratio}")   
        logging.debug(f"---------------------------------------------------------")
        
        
        return ProcessSampleResult(placement_result, acc, latency_ratio=latency_output.baseline_ratio)
    

    def random_sample(self):
        # constraint = self.constraint
        while True:
            sample = self.arch_manager.random_sample()
            
            process_output = self.process_sample(sample)
        
            if process_output.result:
                return sample, process_output
            
            # if valid:
            #     return sample, packing_score.item()
            else:
                # print("Over packing internal holder constraint")
                pass
            
            # if no_bank <= constraint:
            #     return sample, no_bank
            # else:
            #     print("Over bank constraint")

    def mutate_sample(self, sample):
        constraint = self.constraint
        while True:
            new_sample = copy.deepcopy(sample)

            # if random.random() < self.mutate_prob:
            #     self.arch_manager.random_resample_resolution(new_sample)

            for i in range(self.num_blocks):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample(new_sample, i)

            for i in range(self.num_stages):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample_depth(new_sample, i)

            process_output = self.process_sample(new_sample)
        
            if process_output.result:
                return new_sample, process_output
            
            #valid, packing_score = self.packing_predictor.predict_packing(new_sample)
            #if valid:
                #return sample, packing_score
            else:
                # print("Over packing internal holder constraint")
                pass
            # no_bank = self.packing_predictor.predict_packing(new_sample)
            # if no_bank <= constraint:
            #     return new_sample, no_bank

    def crossover_sample(self, sample1, sample2):
        constraint = self.constraint
        while True:
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list):
                    continue
                for i in range(len(new_sample[key])):
                    new_sample[key][i] = random.choice(
                        [sample1[key][i], sample2[key][i]]
                    )

            process_output = self.process_sample(new_sample)
        
            if process_output.result:
                return new_sample, process_output
            #valid, packing_score = self.packing_predictor.predict_packing(new_sample)
            #if valid:
                #return new_sample, packing_score
            else:
                # print("Over packing internal holder constraint")
                pass
            # no_bank = self.packing_predictor.predict_packing(new_sample)
            # if no_bank <= constraint:
            #     return new_sample, no_bank

    def non_dominated_sorting(self, population_size, chroms_obj_record):
        idx_superiorList, idx_inperiorCount={},{}
        frontRank_idxList,idx_rank={},{}
        frontRank_idxList[0]=[]
        for p_idx in range(population_size):
            idx_superiorList[p_idx]=[]
            idx_inperiorCount[p_idx]=0
            for compare_p_idx in range(population_size):
                
                if ((chroms_obj_record[p_idx][0] < chroms_obj_record[compare_p_idx][0] and chroms_obj_record[p_idx][1] < chroms_obj_record[compare_p_idx][1]) or
                    (chroms_obj_record[p_idx][0] <= chroms_obj_record[compare_p_idx][0] and chroms_obj_record[p_idx][1] < chroms_obj_record[compare_p_idx][1]) or 
                    (chroms_obj_record[p_idx][0] < chroms_obj_record[compare_p_idx][0] and chroms_obj_record[p_idx][1] <= chroms_obj_record[compare_p_idx][1])):
                    idx_inperiorCount[p_idx] = idx_inperiorCount[p_idx] + 1
                    
                elif ((chroms_obj_record[p_idx][0] > chroms_obj_record[compare_p_idx][0] and chroms_obj_record[p_idx][1] > chroms_obj_record[compare_p_idx][1]) or 
                      (chroms_obj_record[p_idx][0] >= chroms_obj_record[compare_p_idx][0] and chroms_obj_record[p_idx][1] > chroms_obj_record[compare_p_idx][1]) or 
                      (chroms_obj_record[p_idx][0] > chroms_obj_record[compare_p_idx][0] and chroms_obj_record[p_idx][1] >= chroms_obj_record[compare_p_idx][1])):
                    if compare_p_idx not in idx_superiorList[p_idx]:
                        idx_superiorList[p_idx].append(compare_p_idx)
            
            # if no one is superior to p_idx, it rank is 0
            if idx_inperiorCount[p_idx]==0:
                idx_rank[p_idx]=0
                if p_idx not in frontRank_idxList[0]:
                    frontRank_idxList[0].append(p_idx)
        
        rank_idx=0
        while (frontRank_idxList[rank_idx]!=[]):
            Q=[]
            for p_idx in frontRank_idxList[rank_idx]:
                for q_idx in idx_superiorList[p_idx]:
                    idx_inperiorCount[q_idx]=idx_inperiorCount[q_idx]-1
                    if idx_inperiorCount[q_idx]==0:
                        idx_rank[q_idx]=rank_idx+1
                        if q_idx not in Q:
                            Q.append(q_idx)
            rank_idx=rank_idx+1
            frontRank_idxList[rank_idx]=Q
                    
        del frontRank_idxList[len(frontRank_idxList)-1]
        return frontRank_idxList
    
    def calculate_crowding_distance(self, front,chroms_obj_record):
    
        distance={m:0 for m in front}
        for o in range(2):
            obj={m:chroms_obj_record[m][o] for m in front}
            sorted_keys=sorted(obj, key=obj.get)
            distance[sorted_keys[0]]=distance[sorted_keys[len(front)-1]]=999999999999
            for i in range(1,len(front)-1):
                if len(set(obj.values()))==1:
                    distance[sorted_keys[i]]=distance[sorted_keys[i]]
                else:
                    distance[sorted_keys[i]]=distance[sorted_keys[i]]+(obj[sorted_keys[i+1]]-obj[sorted_keys[i-1]])/(obj[sorted_keys[len(front)-1]]-obj[sorted_keys[0]])
                
        return distance
    
    def selection(self, population_size,front,chroms_obj_record,total_chromosome):   
        N=0
        new_pop=[]
        while N < population_size:
            for i in range(len(front)):
                N=N+len(front[i])
                if N > population_size:
                    distance=self.calculate_crowding_distance(front[i],chroms_obj_record)
                    sorted_cdf=sorted(distance, key=distance.get)
                    sorted_cdf.reverse()
                    for j in sorted_cdf:
                        if len(new_pop)==population_size:
                            break                
                        new_pop.append(j)              
                    break
                else:
                    new_pop.extend(front[i])
        
        population_list=[]
        for n in new_pop:
            population_list.append(total_chromosome[n])
        
        return population_list,new_pop

    def run_evolution_search(self, verbose=False):
        def cal_search_score (acc, latency_ratio):
            return self.args.coef_acc*acc + self.args.coef_latency*latency_ratio
        
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        max_time_budget = self.max_time_budget
        population_size = self.population_size
        mutation_numbers = int(round(self.mutation_ratio * population_size))
        parents_size = int(round(self.parent_ratio * population_size))
        constraint = self.constraint

        best_valids = [-100]
        population = []  # (validation, sample, latency) tuples
        child_pool = []
        no_bank_pool = []
        acc_pool = []
        latency_ratio_pool = []
        best_info = None
        if verbose:
            print("Generate random population...")
            
        for _ in range(population_size):
            sample, output = self.random_sample()
            child_pool.append(sample)
            acc_pool.append(output.acc)
            latency_ratio_pool.append(output.latency_ratio)
            
            
        # may be replaced by acc predictor with encoded population
        # accs = self.accuracy_predictor.predict_accuracy(child_pool)
        # logging.debug(f"Finish predict accuracy for {population_size} samples")
        
        
        for i in range(population_size):
            population.append((child_pool[i], acc_pool[i], latency_ratio_pool[i]))

        logging.info("Start Evolution...")  
        # After the population is seeded, proceed with evolving the population.
        for iter in tqdm(
            range(max_time_budget),
            desc="Searching with %s constraint (%s)"
            % (self.constraint_type, self.constraint),
        ):
            
            sorted_population = sorted(population, key=lambda x: cal_search_score(x[1], x[2]))[::-1]
            
            parents = sorted_population[:parents_size]
            
            acc = parents[0][1]
            latency_ratio = parents[0][2]
            total_score = cal_search_score(acc, latency_ratio)
            
            #csv log 
            if self.csv_writer:
                for i in range(population_size):
                    self.csv_writer.write_row({
                        "net_idx": f"{iter}-{i}",
                        "acc": sorted_population[i][1],
                        "latency_ratio": sorted_population[i][2],
                        "total_score": cal_search_score(sorted_population[i][1], sorted_population[i][2]),
                        "sample": sorted_population[i][0]
                    })
            
            if verbose:
                print("Iter: {} total_score: {}".format(iter - 1, total_score))
            
            if total_score > best_valids[-1]:
                best_valids.append(total_score)
                best_info = parents[0]
            else:
                best_valids.append(best_valids[-1])
            
            # if writer:
            #     writer.add_scalar('Evolution/total_score', total_score, iter)
            #     writer.add_scalar('Evolution/acc', acc, iter)
            #     writer.add_scalar('Evolution/latency_ratio', latency_ratio, iter)
            #     writer.add_scalar('Evolution/best_score', best_valids[-1], iter)
            
            logging.debug(f"[iter-{iter}] best_valids: {best_valids[-1]} from {parents[0][1]}")
            
            
            population = parents
            child_pool = []
            no_bank_pool = []
            acc_pool = []
            latency_ratio_pool = []
            
            logging.debug(f"Start mutate for {mutation_numbers}")        
            
            for i in range(mutation_numbers):
                par_sample = population[np.random.randint(parents_size)][0]
                # Mutate
                new_sample, output = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                acc_pool.append(output.acc)
                latency_ratio_pool.append(output.latency_ratio)
                
            logging.debug(f"Start crossover for {population_size - mutation_numbers}")        
            

            for i in range(population_size - mutation_numbers):
                par_sample1 = population[np.random.randint(parents_size)][0]
                par_sample2 = population[np.random.randint(parents_size)][0]
                # Crossover
                new_sample, output = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                acc_pool.append(output.acc)
                latency_ratio_pool.append(output.latency_ratio)

            logging.debug(f"Start predict accuracy for {population_size} samples")        
            
            # may be replaced by acc predictor with encoded population
            # accs = self.accuracy_predictor.predict_accuracy(child_pool)
            # logging.debug(f"Finish predict accuracy for {population_size} samples")
            
            for i in range(population_size):
                population.append((child_pool[i], acc_pool[i], latency_ratio_pool[i]))

        return best_valids, best_info

    def run_non_dominated_evolution_search(self, verbose=False):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        max_time_budget = self.max_time_budget
        population_size = self.population_size
        mutation_numbers = int(round(self.mutation_ratio * population_size))
        parents_size = int(round(self.parent_ratio * population_size))
        constraint = self.constraint

        best_valids = [-100]
        population_list = [] 
        # child_pool = []
        no_bank_pool = []
        acc_pool = []
        latency_ratio_pool = []
        best_info = None
        
        chroms_obj_record = {} # record the accuracy and latency of each chromosome
        
        
        if self.args.resume:
            # Load checkpoint.
            assert os.path.isdir(self.args.ckpt_path), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(os.path.join(self.args.ckpt_path, "ckpt.pth" ))
            population_list = checkpoint['population']
            best_list=copy.deepcopy(population_list)
            best_obj=checkpoint['best_obj']
            
            last_iter = checkpoint['iter']
            print(f'==> Resuming from checkpoint.. Iter: {last_iter}')
        
        else:
            last_iter = 0
            if verbose:
                print("Generate random population...")
                
            for _ in range(population_size):
                sample, output = self.random_sample()
                population_list.append(sample)
            
            
        # may be replaced by acc predictor with encoded population
        # accs = self.accuracy_predictor.predict_accuracy(child_pool)
        # logging.debug(f"Finish predict accuracy for {population_size} samples")

        logging.info("Start Evolution...")  
        # After the population is seeded, proceed with evolving the population.
        for iter in tqdm(
            range(max_time_budget),
            desc="[%s] Searching with %s constraint (%s)"
            % (self.args.exp_id, self.constraint_type, self.constraint),
        ):
            
            iter = last_iter + iter
            
            chroms_obj_record = {}
            
            parent_list=copy.deepcopy(population_list)
            
            child_pool = []

            logging.debug(f"Start mutate for {mutation_numbers}")        
            
            for i in range(mutation_numbers):
                idx = np.random.randint(population_size)
                par_sample = population_list[idx]
                # Mutate
                new_sample, output = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                chroms_obj_record[i] = (output.acc, output.latency_ratio)
                
            logging.debug(f"Start crossover for {population_size - mutation_numbers}")        
            

            for i in range(population_size - mutation_numbers):
                par_sample1 = population_list[np.random.randint(population_size)]
                par_sample2 = population_list[np.random.randint(population_size)]
                # Crossover
                new_sample, output = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                chroms_obj_record[mutation_numbers + i] = (output.acc, output.latency_ratio)

            # logging.debug(f"Start predict accuracy for {population_size} samples")        
            
            # may be replaced by acc predictor with encoded population
            # accs = self.accuracy_predictor.predict_accuracy(child_pool)
            # logging.debug(f"Finish predict accuracy for {population_size} samples")
        
            frontRank_idxList = self.non_dominated_sorting(population_size,chroms_obj_record)
            population_list ,new_pop = self.selection(population_size, frontRank_idxList, chroms_obj_record, child_pool)
            new_pop_obj=[chroms_obj_record[k] for k in new_pop]    
            
            if iter==0:
                best_list=copy.deepcopy(population_list)
                best_obj=copy.deepcopy(new_pop_obj)
            else:            
                total_list=copy.deepcopy(population_list)+copy.deepcopy(best_list)
                total_obj=copy.deepcopy(new_pop_obj)+copy.deepcopy(best_obj)
                
                frontRank_idxList=self.non_dominated_sorting(population_size*2,total_obj)
            
                for i in frontRank_idxList[0]:
                    logging.debug(f"[iter-{iter}] Rank 0")
                    logging.debug(f"best_valids: {total_obj[i]} from {total_list[i]}")
                    
                best_list,best_pop=self.selection(population_size,frontRank_idxList,total_obj,total_list)
                best_obj=[total_obj[k] for k in best_pop]
                
                population_list = best_list
            
             #csv log 
            if self.csv_writer:
                for i in range(population_size):
                    self.csv_writer.write_row({
                        "net_idx": f"{iter}-{i}",
                        "acc": best_obj[i][0],
                        "latency_ratio": best_obj[i][1],
                        "sample": population_list[i],
                        "total_score": 0
                    })

            # if iter-1 % 10 == 0:
            torch.save({
                'iter': iter,
                'population': population_list,
                'best_obj': best_obj
            }, os.path.join(self.args.ckpt_path, "ckpt.pth"))
            
        
        if self.csv_writer:
            for i in range(population_size):
                self.csv_writer.write_row({
                    "net_idx": f"best-{i}",
                    "acc": new_pop_obj[i][0],
                    "latency_ratio": new_pop_obj[i][1],
                    "sample": population_list[i],
                    "total_score": 0
                })    

        return best_list, best_obj