
import logging
import simpy

class SimOperation:
    def __init__(self,env,id ,cycle, crossbar_idx,exc_order) -> None:
        self.id = id
        self._cycle = cycle
        self.crossbar_idx = crossbar_idx
        self.exc_order = exc_order
        self.set_env(env)

    @property
    def cycle(self):
        return self._cycle
    
    @property
    def latency(self):
        return self._cycle

    def set_extra_cycle(self, extra_cycle):
        self.extra = extra_cycle
    
    def included(self): 
        #when the layer is included in the next layer with same bank loc and same exc_order(same layer), then the cycle is 0
        self._cycle = 0
    
    def set_env(self,env):
        self.env = env
        self.res = simpy.Resource(env, capacity=1)
    
    def forward(self, process, location_res=None, verbose=False):
        #ensure forward pass, waiting for previous result 
        if process:
            yield process
        
        #base on the loading result 
        if location_res:
            res = location_res.get_location_res(self.crossbar_idx)

            #ensure to process later layer first if they are in the same crossbar
            with res.request(priority=-self.exc_order) as req:
                yield req

                #only handle one input at a time
                with self.res.request() as req:
                    yield req
                    
                    # logging.debug('%s@ crossbar#%s forward start at %.2f with latency %.2f' % (self.id, self.crossbar_idx, self.env.now, self.cycle))
                    yield self.env.timeout(self.cycle)
        
        #for parallel processing
        else:
            #only handle one input at a time
            with self.res.request() as req:
                yield req
                # print('%s forward start at %d with cycle %d' % (self.id, self.env.now, self.cycle))
                yield self.env.timeout(self.cycle)

'''
    Network Layer Resource
    create network resource for each exc_order base on the multiply in layer box info
'''
class NetworkLayerResource():
    def __init__(self,env) -> None:
        self.env=env
        self.store_set = {}
    
    def create_network_store(self, exc_order, multiply=1):
        store = simpy.Store(self.env, capacity=multiply)
        self.store_set[exc_order] = store
        return store
        
    def get_netowrk_res(self, exc_order):
        return self.store_set.get(exc_order)
      

class LocationResource():
    def __init__(self,env) -> None:
        self.env=env
        self.res_set = {}

    def get_location_res(self,location_idx):
        location_res = self.res_set.get(location_idx)
        if not location_res:
            location_res = simpy.PriorityResource(self.env, capacity=1)
            self.res_set[location_idx] = location_res
        
        return location_res