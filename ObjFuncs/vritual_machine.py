import numpy as np
from typing import List, Union, Optional, Callable
from abc import ABC, abstractmethod
from scipy.optimize import rosen


def rosen_bilog(x):
    y=rosen(x)
    return [np.sign(y)*np.log(1+np.abs(y)),]


def zscore_mean(x,abs_z: Optional[float] = None):
    x = np.array(x)
    mean = np.mean(x)
    if abs_z is None:
        return mean
    std = np.std(x)
    if np.any(std==0.):
        return mean
    return np.mean(x[ np.logical_and(mean-abs_z*std < x, x < mean+abs_z*std) ])


class VM():
    def __init__(self,
                 x0: List[float],
                 func: Optional[Callable] = rosen_bilog,
                 decision_CSETs: Optional[List[str]] = None,
                 objective_RDs: Optional[List[str]] = None,
                 objective_RD_noises: Optional[List[float]] = None,
                 dt: Optional[float] = 0.2,
                 ):
        self.dim = len(x0)
        self.decision_CSET_vals = x0
        self.decision_CSETs = decision_CSETs or ['x'+str(i) for i in range(dim)]
        assert len(self.decision_CSETs) == self.dim
        
        
        
        self.objective_RDs = objective_RDs or ['y',]
        self.func = func
        self.objective_RD_vals = self.func(self.decision_CSET_vals)
        assert len(self.objective_RD_vals) == len(self.objective_RDs)
        self.objective_RD_noises = np.array( 
            objective_RD_noises or np.zeros(len(self.objective_RDs)) 
            )

        
        self.dt = dt or 0.2
        self.t = 0
        
        self.history = {'t':[self.t]}
        for pv,val in zip(self.decision_CSETs, self.decision_CSET_vals):
            self.history[pv] = [val]
        for pv,val in zip(self.objective_RDs, self.objective_RD_vals):
            self.history[pv] = [val]
        

    def __call__(self):
        self.t += self.dt
        self.objective_RD_vals = np.array(self.func(self.decision_CSET_vals)) \
                               + self.objective_RD_noises*np.random.randn(len(self.objective_RDs))
        self.history['t'].append(self.t)
        for pv,val in zip(self.decision_CSETs, self.decision_CSET_vals):
            self.history[pv].append(val)
        for pv,val in zip(self.objective_RDs, self.objective_RD_vals):
            self.history[pv].append(val)
        
        
    def ensure_set(self, 
                   setpoint_pv: Union[str, List[str]], 
                   readback_pv: Union[str, List[str]], 
                   goal: Union[float, List[float]], 
                   tol: Union[float, List[float]] = 0.01, 
                   timeout: float = 10.0, 
                   verbose: bool = False):
        for i,pv in enumerate(self.decision_CSETs):
            for j,pv_sp in enumerate(setpoint_pv):
                if pv==pv_sp:
                    self.decision_CSET_vals[i]=goal[j]
                    
                    
    def fetch_data(self, 
                   pvlist: List[str], 
                   time_span: float = 5.0, 
                   abs_z: Optional[float] = None, 
                   with_data=False, 
                   verbose=False):
        for pv in pvlist:
            if pv not in self.objective_RDs + self.decision_CSETs:
                raise ValueError(pv+" is not recognized")       
        t0=self.t
        raw_data = {pv:[] for pv in pvlist}
        raw_data['t']=[]
        while(self.t - t0 <= time_span):
            self()
            raw_data['t'].append(self.history['t'][-1])
            for pv in pvlist:
                raw_data[pv].append(self.history[pv][-1])       
                
        ave_data = {pv:zscore_mean(raw_data[pv],abs_z) for pv in pvlist}
        
        return ave_data, raw_data
                    
        
        
        
        
        
        
      