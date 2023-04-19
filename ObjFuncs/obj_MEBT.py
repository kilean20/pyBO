from warnings import warn
import numpy as np
from typing import List, Union, Optional
from objFuncs import singleReadbackObj, objFuncBase, get_RDnorm, calculate_objective
from phantasy import caget, fetch_data, caput, ensure_set
# def ensure_set(xset,xrd,val,tol):
#     pass
# def caput(pv,val):
#     pass
        
        
class obj_MEBT(objFuncBase):
    def __init__(self,
        decision_CSETs = ['FE_LEBT:PSC2_D0979:I_CSET',
                          'FE_LEBT:PSC1_D0979:I_CSET'],
        decision_min = [-5,-5],
        decision_max = [ 5, 5],        
        objective_goal = { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1056:YPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1056:PHASE_RD': 77.38,   #78.98, #(deg)
            'FE_MEBT:BPM_D1056:MAG_RD'  : np.inf,
            'FE_MEBT:BPM_D1072:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1072:YPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1072:PHASE_RD':-26.38,    #-26.71, #(deg)
            'FE_MEBT:BPM_D1072:MAG_RD'  : np.inf,
            'FE_MEBT:BPM_D1094:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1094:YPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1094:PHASE_RD':-16.54,    #-19.41 , #(deg)
            'FE_MEBT:BPM_D1094:MAG_RD'  : np.inf,
            'FE_MEBT:BCM_D1055:AVGPK_RD/FE_LEBT:BCM_D0989:AVGPK_RD': {'min':0, 'max':1},
            'FE_MEBT:FC_D1102:PKAVG_RD': np.inf,
                           },
        objective_weight = { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1056:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1056:PHASE_RD': 1., 
            'FE_MEBT:BPM_D1056:MAG_RD'  : 0., 
            'FE_MEBT:BPM_D1072:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1072:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1072:PHASE_RD': 1., 
            'FE_MEBT:BPM_D1072:MAG_RD'  : 0., 
            'FE_MEBT:BPM_D1094:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1094:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1094:PHASE_RD': 1.,
            'FE_MEBT:BPM_D1094:MAG_RD'  : 0.,
            'FE_MEBT:BCM_D1055:AVGPK_RD/FE_LEBT:BCM_D0989:AVGPK_RD': 10,
            'FE_MEBT:FC_D1102:PKAVG_RD': 15,
            },
#         objective_norm = None,
        objective_norm = { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1056:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1056:PHASE_RD': 1., 
            'FE_MEBT:BPM_D1056:MAG_RD'  : None, 
            'FE_MEBT:BPM_D1072:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1072:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1072:PHASE_RD': 1., 
            'FE_MEBT:BPM_D1072:MAG_RD'  : None, 
            'FE_MEBT:BPM_D1094:XPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1094:YPOS_RD' : 1.,     
            'FE_MEBT:BPM_D1094:PHASE_RD': 1.,
            'FE_MEBT:BPM_D1094:MAG_RD'  : None,
            'FE_MEBT:BCM_D1055:AVGPK_RD/FE_LEBT:BCM_D0989:AVGPK_RD': 1,
            'FE_MEBT:FC_D1102:PKAVG_RD': None,
            },
        objective_RD_avg_time = 2,
        decision_L2reg = 0.01,
        minimize = False,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        ):
        for pv in decision_CSETs:
            assert type(pv) is str
        super().__init__(
            decision_CSETs=decision_CSETs,
            decision_min=decision_min,
            decision_max=decision_max,
            decision_RDs=decision_RDs,
            decision_tols=decision_tols,
            decision_L2reg=decision_L2reg,
            minimize = minimize,
            )
        

        self.objective_weight = {key:val for key,val in objective_weight.items() if val != 0.}
        self._check_device_init() # objective_weight must be defined first
        self._get_xinit()
        
        self.objective_goal   = {key:val for key,val in objective_goal.items() if key in self.objective_weight.keys()}
        self.objective_norm = {key:get_RDnorm(key,objective_norm) for key in self.objective_weight.keys()}
        assert self.objective_goal.keys() == self.objective_weight.keys() == self.objective_norm.keys()
        
        wtot = np.sum(list(self.objective_weight.values()))
        for key in self.objective_weight.keys():
            self.objective_weight[key] /= wtot
            
        self.objective_RD = {}
        for key in self.objective_goal.keys():
            iratio = key.find('/')
            if i!=-1:
                self.objective_RD[key[:iratio]] = []
                self.objective_RD[key[iratio+1:]] = []
            else:
                self.objective_RD[key] = []

        
    def _check_device_init(self):
        '''
        check devices status,
        '''
        if "FE_MEBT:FC_D1102:PKAVG_RD" in self.objective_weight.keys():
            if caget('FE_MEBT:FC_D1102:RNG_CMD')==1:
                warn("FC_D0998 range is set to 1uA. Changing to 1055uA.")
                caput('FE_MEBT:FC_D1102:RNG_CMD',0)
            if caget("FE_MEBT:FC_D1102:LMIN_RSTS") == 0:
                raise RuntimeError("FC_D1102 is not in.") 
            if caget("DIAG-RIO01:FC_ENABLED2") != 1:
                raise RuntimeError("FC_D1102 is not enabled. Check bottom of FC overview in CSS") 
            if caget("DIAG-RIO01:PICO_ENABLED2") != 1:
                raise RuntimeError("FC_D1102 pico is not enabled. Check bottom of FC overview in CSS") 
        if caget("ACS_DIAG:CHP:STATE_RD") != 3:
            raise RuntimeError("Chopper blocking.") 
        if caget("FE_LEBT:ATT1_D0957:LMOUT_RSTS") == 0:
            warn("ATT1_D0957 is in.") 
        if caget("FE_LEBT:ATT2_D0957:LMOUT_RSTS") == 0:
            warn("ATT2_D0957 is in.")  
        if caget("FE_LEBT:ATT1_D0974:LMOUT_RSTS") == 0:
            warn("ATT1_D0974 is in.") 
        if caget("FE_LEBT:ATT2_D0974:LMOUT_RSTS") == 0:
            warn("ATT2_D0974 is in.")  

            
            
    def _check_device_runtime(self):
        '''
        check devices status,
        '''
        if caget("ACS_DIAG:CHP:STATE_RD") != 3:
            raise RuntimeError("Chopper blocking.") 

            
    def __call__(self,x,objective_RD_avg_time=None,abs_z=3):
        self._check_device_runtime()
        objective_RD_avg_time = objective_RD_avg_time or self.objective_RD_avg_time
        #set
        ensure_set(self.decision_CSETs,self.decision_RDs,x,self.decision_tols)
        #read
        ave_data, _ = fetch_data(self.objective_RD.columns,
                                 objective_RD_avg_time,
                                 abs_z=abs_z)
        self.objective_RD.loc[len(self.objective_RD)] = list(ave_data.values())
        #regularize
        obj = calculate_objective(self.objective_goal,self.objective_weight,self.objective_norm,ave_data) \
              +self.regularize(x)
        if self.minimize:
            return -obj
        else:
            return  obj