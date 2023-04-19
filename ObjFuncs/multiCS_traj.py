from warnings import warn
import numpy as np
from typing import List, Union, Optional
from objFuncs import singleReadbackObj, objFuncBase
from phantasy import caget, fetch_data #, caput, ensure_set
def ensure_set(xset,xrd,val,tol):
    pass
def caput(pv,val):
    pass


class multiCS_traj_FS2(objFuncBase):
    def __init__(self,
#         decision_CSETs = ['FS1_CSS:PSC2_D2381:I_CSET',
#                           'FS1_CSS:PSC1_D2381:I_CSET',
#                           'FS1_CSS:PSC2_D2367:I_CSET',
#                           'FS1_CSS:PSC1_D2367:I_CSET'],
        decision_CSETs = ['FS2_BTS:PSC2_D3930:I_CSET',
                          'FS2_BTS:PSC1_D3930:I_CSET',
                          'FS2_BTS:PSC2_D3945:I_CSET',
                          'FS2_BTS:PSC1_D3945:I_CSET',],
        decision_min = [-4,-4,-4,-4],
        decision_max = [ 4, 4, 4, 4],
#         CSselector_SETs = [[-17,10],[0,10],[17,10]],   # 2023-02-08 124Xe, Q = 51,50,49  case         
        CSselector_SETs = [[-17,10],[17,10]],   # 2023-02-08 124Xe, Q = 51,49  case 
#         objective_BPMxys = ['FS1_BMS:BPM_D2502:XPOS_RD',
#                             'FS1_BMS:BPM_D2502:YPOS_RD',
#                             'FS1_BMS:BPM_D2537:XPOS_RD',
#                             'FS1_BMS:BPM_D2537:YPOS_RD'],
        objective_BPMxys = ['FS2_BMS:BPM_D4142:XPOS_RD',
                            'FS2_BMS:BPM_D4142:YPOS_RD',
                            'FS2_BMS:BPM_D4164:XPOS_RD',
                            'FS2_BMS:BPM_D4164:YPOS_RD'],
        objective_BPMxys_weights = 0.5,
#         objective_BPMphs = ['FS1_BMS:BPM_D2502:PHASE_RD',
#                             'FS1_BMS:BPM_D2537:PHASE_RD',],
        objective_BPMphs = ['FS2_BMS:BPM_D4142:PHASE_RD',
                            'FS2_BMS:BPM_D4164:PHASE_RD'],
        objective_BPMphs_weights = 1,
#         objective_minimize_RDs = [],
#         objective_minimize_RDs_weights = [],
#         objective_maximize_RDs = ['FS1_BMS:BPM_D2502:MAG_RD',
#                                   'FS1_BMS:BPM_D2537:MAG_RD',],
#         objective_maximize_RDs_weights = [1,1],
        objective_RD_avg_time = 3,
        minimize = True
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        ):
        super().__init__(
            decision_CSETs,
            decision_min,
            decision_max,
            decision_RDs = decision_RDs,
            decision_tols = decision_tols,
            minimize = minimize
            )
        
        
        self.CSselector_SETs = CSselector_SETs
        self.CS_CSETs = ['FS1_BBS:CSEL_D2405:CTR_MTR.VAL','FS1_BBS:CSEL_D2405:GAP_MTR.VAL']
        self.CS_RDs   = ['FS1_BBS:CSEL_D2405:CTR_MTR.RBV','FS1_BBS:CSEL_D2405:GAP_MTR.RBV']
        self.CS_tols  = [0.01,0.01]  #mm
        
        self.objective_BPMxys = objective_BPMxys
        self.objective_BPMphs = objective_BPMphs
        if type(objective_BPMxys_weights) is float:
            objective_BPMxys_weights = [objective_BPMxys_weights]*len(self.objective_BPMxys)
        if type(objective_BPMphs_weights) is float or type(objective_BPMphs_weights) is int:
            objective_BPMphs_weights = [objective_BPMphs_weights]*len(self.objective_BPMphs)
        
        self.objective_BPMxys_weights = np.array(objective_BPMxys_weights,dtype=np.float)
        self.objective_BPMphs_weights = np.array(objective_BPMphs_weights,dtype=np.float)
        print(self.objective_BPMxys_weights.shape)
        print(self.objective_BPMphs_weights.shape)
        
        wTot = self.objective_BPMxys_weights.mean() + self.objective_BPMphs_weights.mean()
        self.objective_BPMxys_weights /= (wTot*len(self.objective_BPMxys_weights))
        self.objective_BPMphs_weights /= (wTot*len(self.objective_BPMphs_weights))
        
        self.objective_RD_avg_time = objective_RD_avg_time
        
        
            
    def _check_device(self):
        if caget("ACS_DIAG:CHP:STATE_RD") != 3:
            raise RuntimeError("Chopper not running.") 

    def prepare_objective(self):
        self._check_device()
        self._get_xinit()
        
        
        
    def selectCS(self,i):
        assert i in list(range(len(self.CSselector_SETs)))
        ensure_set(self.CS_CSETs,
                   self.CS_RDs,
                   self.CSselector_SETs[i],
                   self.CS_tols)
        
        
    def measure_allCS(self):
        data = np.zeros((len(self.CSselector_SETs),len(self.objective_BPMxys)+len(self.objective_BPMphs)))
        for i in range(len(self.CSselector_SETs)):
            self.selectCS(i)
            ave,_ = fetch_data(self.objective_BPMxys+self.objective_BPMphs, self.objective_RD_avg_time, 3)
            data[i,:] = ave
        return data
    
    
    def __call__(self,decision_vals):
        
        decision_vals = np.atleast_2d(decision_vals)
        batch_size, dim = decision_vals.shape
        
        
        data = 
        
        for icharge in range(len(self.CSselector_SETs)):
            # select charge state
            ensure_set(self.CS_CSETs,
                   self.CS_RDs,
                   self.CSselector_SETs[i],
                   self.CS_tols)
            
            for ibatch, decision_val in enumerate(decision_vals):
                # set quads, corrs, etc
                ensure_set(self.decision_CSETs,
                           self.decision_RDs,
                           decision_val,
                           self.decision_tols)
                
                
                

                data = self.measure_allCS()

            xy_std = np.sum( data[:,:len(self.objective_BPMxys)].std(axis=0)
                             *self.objective_BPMxys_weights )  
            xy_mean = np.sum( data[:,:len(self.objective_BPMxys)].mean(axis=0)
                              *self.objective_BPMxys_weights ) # beam centroid averaged over all CS
            ph_std = np.sum( data[:,len(self.objective_BPMxys):len(self.objective_BPMphs)].std(axis=0)
                             *self.objective_BPMphs_weights  )  
            cost = xy_std + ph_std +0.5*xy_mean

            obj = 1-2*cost

            self.history['decision_vals'].append(decision_vals)
            self.history['data'].append(data)
            self.history['xy_std'].append(xy_std)
            self.history['xy_mean'].append(xy_mean)
            self.history['obj'].append(obj)

            return obj
        
        