
import math
import numpy as np

from rram_nas_comp_pack.box_converter.info.LayerBox import LayerBox

'''
Crossbar as the 2D container 
'''

class Crossbar():
    def __init__(self, dim, verbose=False):
        self.dimensions = dim
        self.EMSs = [[np.array((0,0)), np.array(dim)]]
        self.load_items = []
        
        if verbose:
            print('Init EMSs:',self.EMSs)
    
    def __getitem__(self, index):
        return self.EMSs[index]
    
    def __len__(self):
        return len(self.EMSs)
    
    def update(self, layer_box:LayerBox, selected_EMS, min_area = 1, min_dim = 1, verbose=False):

        # 1. place box in a EMS
        boxToPlace = np.array(layer_box.size)
        selected_min = np.array(selected_EMS[0])
        ems = [selected_min, selected_min + boxToPlace]
        self.load_items.append(layer_box)
        
        if verbose:
            print('------------\n*Place Box*:\nEMS:', list(map(tuple, ems)))
        
        # 2. Generate new EMSs resulting from the intersection of the box
        for EMS in self.EMSs.copy():
            if self.overlapped(ems, EMS):
                
                # eliminate overlapped EMS
                self.eliminate(EMS)
                
                if verbose:
                    print('\n*Elimination*:\nRemove overlapped EMS:',list(map(tuple, EMS)),'\nEMSs left:', list(map( lambda x : list(map(tuple,x)), self.EMSs)))
                
                # 2 new EMSs in 2 dimensions
                x1, y1 = EMS[0]; x2, y2 = EMS[1]
                x3, y3 = ems[0]; x4, y4 = ems[1]
                new_EMSs = [
                    [np.array((x4, y1)), np.array((x2, y2))],
                    [np.array((x1, y4)), np.array((x2, y2))],
                ]
                

                for new_EMS in new_EMSs:
                    new_box = new_EMS[1] - new_EMS[0]
                    isValid = True
                    
                    if verbose:
                        print('\n*New*\nEMS:', list(map(tuple, new_EMS)))

                    # 3. Eliminate new EMSs which are totally inscribed by other EMSs
                    for other_EMS in self.EMSs:
                        if self.inscribed(new_EMS, other_EMS):
                            isValid = False
                            if verbose:
                                print('-> Totally inscribed by:', list(map(tuple, other_EMS)))
                            
                    # 4. Do not add new EMS smaller than the volume of remaining boxes
                    if np.min(new_box) < min_dim:
                        isValid = False
                        if verbose:
                            print('-> Dimension too small.')
                        
                    # 5. Do not add new EMS having smaller dimension of the smallest dimension of remaining boxes
                    if np.product(new_box) < min_area:
                        isValid = False
                        if verbose:
                            print('-> Volumne too small.')

                    if isValid:
                        self.EMSs.append(new_EMS)
                        if verbose:
                            print('-> Success\nAdd new EMS:', list(map(tuple, new_EMS)))

        if verbose:
            print('\nEnd:')
            print('EMSs:', list(map( lambda x : list(map(tuple,x)), self.EMSs)))
    
    def overlapped(self, ems, EMS):
        if np.all(ems[1] > EMS[0]) and np.all(ems[0] < EMS[1]):
            return True
        return False
    
    def inscribed(self, ems, EMS):
        if np.all(EMS[0] <= ems[0]) and np.all(ems[1] <= EMS[1]):
            return True
        return False
    
    def eliminate(self, ems):
        # numpy array can't compare directly
        ems = list(map(tuple, ems))    
        for index, EMS in enumerate(self.EMSs):
            if ems == list(map(tuple, EMS)):
                self.EMSs.pop(index)
                return
    
    def get_EMSs(self):
        return  list(map( lambda x : list(map(tuple,x)), self.EMSs))
    
    def load(self):
        return np.sum([ np.product(item.size[1] - item.size[0]) for item in self.load_items]) / np.product(self.dimensions)
    
    
    def is_layer_conflict(self, layer_box:LayerBox, layer_threshold = 1):
        # check if the layer_box is in conflict with other layer_box in the crossbar
        # consider conflict if there is load layer box share same exc_order
        # layer_threshold is the threshold for the layer difference
        
        for loaded_box in self.load_items:
            if abs(loaded_box.exc_order - layer_box.exc_order) <= layer_threshold:
               return True
        return False
    
    
    def is_max_loaded(self, max_load):
        if len(self.load_items) >= max_load:
            return True
        return False