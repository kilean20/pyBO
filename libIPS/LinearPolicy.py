from warnings import warn
import numpy as np
from typing import List, Union, Optional
from phantasy import caget, fetch_data #, caput, ensure_set
def ensure_set(xset,xrd,val,tol):
    pass
def caput(pv,val):
    pass


class LinearPolicy():
    def __init__(self,

        decision_CSETs = [
                          ['LS1_CB11:RFC_D1918:PHA_CSET'],
                          ['FS1_CH01:RFC_D2137:PHA_CSET',
                           'FS1_CH01:RFC_D2141:PHA_CSET',
                           'FS1_CH01:RFC_D2148:PHA_CSET',
                           'FS1_CH01:RFC_D2152:PHA_CSET'],
                          ],
        decision_min: Union[float,List[float]] = -5,
        decision_max: Union[float,List[float]] =  5,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        decision_coupling_coefficient = None,  
        decision_L2reg: float = 0.,
        minimize: Optional[bool] = False,
        history_buffer_size: Optional[int] = 32768,          
        ):
    super().__init__(
            decision_CSETs =decision_CSETs,
            decision_min = decision_min,
            decision_max = decision_max,
            decision_coupling_coefficient = decision_coupling_coefficient,  
            decision_L2reg=decision_L2reg,
            decision_RDs = decision_RDs,
            decision_tols = decision_tols,
            minimize = minimize,
            )
 
        self.history['']
            
        
    def _check_device(self):
        if caget("ACS_DIAG:CHP:STATE_RD") != 3:
            raise RuntimeError("Chopper not running.")

    def prepare_objective(self,avetime):
        '''
        -check if beam is present, relavant decices are ready. 
        -collect statistics of beam energy and BPM phase fluctuation.
        '''
        self._check_device()
        self._get_xinit()
        
    def _get_xinit(self):
        x0, _ = fetch_data(self.decision_CSETs,0.01)
        self.x0 = x0
        
    def get_state(self):
        cashed = False
        
        
