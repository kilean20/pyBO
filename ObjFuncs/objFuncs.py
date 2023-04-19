import numpy as np
from typing import List, Dict, Union, Optional
from abc import ABC, abstractmethod
import datetime
from phantasy import caget, fetch_data, caput, ensure_set
from copy import deepcopy as copy
# def ensure_set(xset,xrd,val,tol):
#     pass
# def caput(pv,val):
#     pass
    

# __all__ = ['objFuncBase','singleReadbackObj']
    
    
def get_tolerance(PV_CSETs: List[str]):
    '''
    Automatically define tolerance in terms of ramping rate (per sec)
    PV_CSETs: list of CSET-PVs 
    '''
    pv_tol = []
    for pv in PV_CSETs:
        pv_tol.append(pv[:pv.rfind(':')]+':RSSV_RSET')
    tol,_ = fetch_data(pv_tol,0.01)
    return tol
       
    

class objFuncBase(ABC):
    def __init__(self,
        decision_CSETs: List[Union[str,List[str]]],
        decision_min: Union[float,List[float]],
        decision_max: Union[float,List[float]],
        decision_couplings: Optional[Dict] = None,  
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        minimize: Optional[bool] = False,
        history_buffer_size: Optional[int] = None,
        ):
        '''
        decision_CSETs: [List of] List of CSET-PVs for control. 
        decision_min: Float or List of minimum of 'decision_CSETs', must be len(decision_min) == len(decision_CSETs) 
        decision_max: Float or List of maximum of 'decision_CSETs', must be len(decision_min) == len(decision_CSETs) 
        decision_couplings: CSETs that need to be coupled with one of decision_CSETs. 
            (e.g) decision_couplings = {"EQUAD1_D0000": {"EQUAD2_D0000":-1}, }    <- here "EQUAD1_D0000" should be in decision_CSETs and "EQUAD2_D0000" is coupled to "EQUAD1_D0000" with coupling coefficient -1.            
        decision_RDs: List of RD-PVs for control used to ensure ramping completion. If 'None' automatically decided by changing 'CSET' to 'RD' in PV name 
        decision_tols: List or numpy array of tolerance used to ensure ramping completion. If 'None' automatically decided based on ramping rate definition.
        minimize: True/False. By default False -> objective is to be maximized. 
        '''
#         is_decision_coupled = False
#         for decision in decision_CSETs:
#             if type(decision) is list:
#                 is_decision_coupled = True
#         if is_decision_coupled:
#             self.decision_coupling_index = []
#             for i,decision in enumerate(decision_CSETs):
#                 if type(decision) is not list:
#                     decision_CSETs[i] = [decision]
#                 self.decision_coupling_index += [i]*len(decision_CSETs[i])
#             self.decision_CSETs = [pv for sublist in decision_CSETs for pv in sublist]  #flatten the nested list represent coupling of the decision_CSETs
#         else:
#             self.decision_coupling_index = None
#             self.decision_CSETs = decision_CSETs

        self.decision_CSETs = decision_CSETs
        
        if type(decision_min) is float or type(decision_min) is int:
            decision_min = [float(decision_min)]*len(decision_CSETs)
        assert len(decision_min) == len(decision_CSETs)
        self.decision_min = np.array(decision_min)
        if type(decision_max) is float or type(decision_max) is int:
            decision_max = [float(decision_max)]*len(decision_CSETs)
        assert len(decision_max) == len(decision_CSETs)
        self.decision_max = np.array(decision_max)
        self.decision_bounds = np.array([(d_min,d_max) for (d_min,d_max) in zip(decision_min, decision_max)])
        
#         self.decision_coupling_coefficient = decision_coupling_coefficient
#         if decision_coupling_coefficient is not None:
#             assert is_decision_coupled,  "decision_coupling_coefficient is required only when decision_CSETs shows coupling"
#             assert len(decision_coupling_coefficient) == len(self.decision_CSETs)
#             j=0
#             for i,decision in enumerate(decision_CSETs):
#                 assert decision_coupling_coefficient[j] == 1.,  "decision_coupling_coefficient of the element represnting coupling must be 1"
#                 j+=len(decision_CSETs[i])
        
        if decision_RDs is None:
            self.decision_RDs = [pv.replace('CSET','RD') for pv in decision_CSETs]
            try:
                _ = fetch_data(self.decision_RDs,0.1)
            except:
                raise ValueError("Automatic decision of 'decision_RDs' failed. Check 'decision_CSETs' or provide 'decision_RDs' manually")
        else:
            self.decision_RDs = decision_RDs
        if decision_tols is None:
            try:
                self.decision_tols = get_tolerance(decision_CSETs)
            except:
                raise ValueError("Automatic decision of the 'decision_tols' failed. Check 'decision_CSETs' or provide 'decision_tols'")
        else:
            self.decision_tols = decision_tols   
            
            
        self.decision_couplings = decision_couplings
        if self.decision_couplings is not None:
            assert type(self.decision_couplings) is dict
            self._coupled_decision_CSETs = {"CSETs":list[self.decision_CSETs], "index":[], "coeff":[]}
            for pv,couples in self.decision_couplings.items():
                assert pv in self.decision_CSETs
                icouple = self.decision_CSETs.index(pv)
                assert type(couples) is dict
                for cpv,coeff in couples.items():
                    assert cpv not in self._coupled_decision_CSETs["CSETs"]
                    assert type(coeff) in [float,int]
                    self._coupled_decision_CSETs["CSETs"].append(cpv)
                    self._coupled_decision_CSETs["index"].append(icouple)
                    self._coupled_decision_CSETs["coeff"].append(float(coeff))
            
            self._coupled_decision_CSETs["coeff"] = np.array(_coupled_decision_CSETs["coeff"])
            self._coupled_decision_CSETs["RDs"] = [pv.replace('CSET','RD') for pv in self._coupled_decision_CSETs["CSETs"]]
            try:
                _ = fetch_data(self._coupled_decision_CSETs["RDs"],0.1)
            except:
                raise ValueError("Automatic decision of 'decision_RDs' of coupled CSETs failed.")
            try:
                self._coupled_decision_CSETs["tols"] = get_tolerance(self._coupled_decision_CSETs["CSETs"]) 
            except:
                raise ValueError("Automatic decision of the 'decision_tols' of coupled CSETs failed.")
            
        self.minimize = minimize
        
        self.history_buffer_size = history_buffer_size
        self.history = {'time':[],
                        'decision_RDs':{pv:[] for pv in self.decision_RDs}}
        
    @abstractmethod
    def _check_device_init(self):
        pass
    
    @abstractmethod
    def _check_device_runtime(self):
        pass
    
    def _set_decision(self,x):
        assert x.dim==1
        assert len(x) == len(self.decision_CSETs)
        if self.decision_couplings is None:
            ensure_set(self.decision_CSETs,self.decision_RDs,x,self.decision_tols)
        else:
            n = len(self.decision_CSETs) 
            x_ = np.zeros(len(self._coupled_decision_CSETs["CSETs"]))
            x_[:n] = x[:]
            count = 0
            for i,coeff in zip(self._coupled_decision_CSETs["index"],self._coupled_decision_CSETs["coeff"])
                x_[n+count] = x[i]*coeff
                count += 1
            ensure_set(self._coupled_decision_CSETs["CSETs"],
                       self._coupled_decision_CSETs["RDs"],
                       x_,
                       self._coupled_decision_CSETs["tols"])
        
    
    def _get_xinit(self):
        x0, _ = fetch_data(self.decision_CSETs,0.01)
        self.x0 = x0
               
        
    def prepare_objective(self):
        self._check_device_init()
        self._get_xinit()
        
        
#     def regularize(self,x):
#         if self.decision_L2reg > 0:
#             x_reg = 2*(x- self.decision_min)/(self.decision_max-self.decision_min) - 1
#             reg = self.decision_L2reg*np.mean(x_reg**2)
#             if self.minimize:
#                 return reg
#             else:
#                 return -reg
#         else:
#             return 0.


        
class singleReadbackObj(objFuncBase):
    def __init__(self,
        decision_CSETs: List[str],
        decision_min: Union[float,List[float]],
        decision_max: Union[float,List[float]],
        objective_RD: str,
                 
        decision_couplings: Optional[Dict] = None,  
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        
        objective_RD_min: Optional[float] = 0.,
        objective_RD_max: Optional[float] = 1., 
        objective_RD_avg_time: Optional[float] = 2,
        minimize: Optional[bool] = False,
        history_buffer_size: Optional[int] = None,
        ):
        '''
        decision_CSETs: List of CSET-PVs for control
        decision_min: List or numpy array of minimum of 'decision_CSETs'
        decision_max: List or numpy array of maximum of 'decision_CSETs'
        decision_couplings: CSETs that need to be coupled with one of decision_CSETs. 
            (e.g) decision_couplings = {"EQUAD1_D0000": {"EQUAD2_D0000":-1}, }    <- here "EQUAD1_D0000" should be in decision_CSETs and "EQUAD2_D0000" is coupled to "EQUAD1_D0000" with coupling coefficient -1.            
        minimize: True/False. (False for maximization)
        objective_RD: single RD-PV for objective
        objective_RD_min: expected minimum of objective_RD for objective value regularization
        objective_RD_max: expected maximum of objective_RD for objective value regularization
        objective_RD_avg_time: wait time for readback averaging
        decision_RDs: List of RDs for control used to ensure ramping completion. If 'None' automatically decided by changing 'CSET' to 'RD' in PV name 
        decision_tols: List or numpy array of tolerance used to ensure ramping completion. If 'None' automatically decided based on ramping rate definition.
        '''
        super().__init__(
            decision_CSETs= decision_CSETs,
            decision_min = decision_min,
            decision_max = decision_max,
            decision_couplings = decision_couplings,  
            decision_L2reg = decision_L2reg,
            decision_RDs = decision_RDs,
            decision_tols = decision_tols,
            minimize = minimize,
            history_buffer_size = history_buffer_size,
        )    
        if type(objective_RD) is not str:
            raise ValueError("PV name for objective_RD that must be provided in string format")
        self.objective_RD = objective_RD
        self.objective_RD_min = objective_RD_min
        self.objective_RD_max = objective_RD_max
        self.objective_RD_avg_time = objective_RD_avg_time
        self.history[self.objective_RD] = []
        
        
    def __call__(self,x,objective_RD_avg_time=None,abs_z=3):
        objective_RD_avg_time = objective_RD_avg_time or self.objective_RD_avg_time
        #set
        self._set_decision(x)
        #read
        self._check_device_runtime()
        RD,_ = fetch_data(self.decision_RDs+[self.objective_RD],objective_RD_avg_time,abs_z=abs_z)
        self._check_device_runtime()
        i=0
        for key in self.history["decision_RDs"].keys():
            self.history["decision_RDs"][key].append(RD[i])
            i+=1
        self.history[self.objective_RD].append(RD[-1])
        #regularize
        obj = 2*(RD[-1]-self.objective_RD_min) / (self.objective_RD_max-self.objective_RD_min)-1 #\
#               +self.regularize(x)
        if self.minimize:
            return -obj
        else:
            return obj
        
                 
            
class multiReadbackObj(objFuncBase):
    def __init__(self,
        decision_CSETs: List[str],
        decision_min: Union[float,List[float]],
        decision_max: Union[float,List[float]],
        objective_goal:  Dict, 
        objective_weight:  Dict,
        objective_norm: Dict,
        objective_p_order:Optional[float] = 2,
        objective_RD_avg_time:Optional[float] = 2,
        apply_bilog:Optional[bool] = False,
                 
        decision_couplings: Optional[Dict] = None,
        decision_RDs: Optional[List[str]] = None,
        decision_tols: Optional[List[float]] = None,
        minimize: Optional[bool] = False,
        history_buffer_size: Optional[int] = None,
        ):
        '''
        e.g.)
        objective_goal = { 
            'FE_MEBT:BPM_D1056:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1056:YPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1056:PHASE_RD': 77.38,   #78.98, #(deg)
            'FE_MEBT:BPM_D1056:MAG_RD'  : {'goal': np.inf, 'zero_ref': None}
            'FE_MEBT:BPM_D1072:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1072:YPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1072:PHASE_RD':-26.38,    #-26.71, #(deg)
            'FE_MEBT:BPM_D1072:MAG_RD'  : {'goal': np.inf, 'zero_ref': None}
            'FE_MEBT:BPM_D1094:XPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1094:YPOS_RD' : 0.0,     #(mm)
            'FE_MEBT:BPM_D1094:PHASE_RD':-16.54,    #-19.41 , #(deg)
            'FE_MEBT:BPM_D1094:MAG_RD'  : {'goal': np.inf, 'zero_ref': None}
            'FE_MEBT:BCM_D1055:AVGPK_RD/FE_LEBT:BCM_D0989:AVGPK_RD': {'goal': np.inf, 'zero_ref': None},
            'FE_MEBT:FC_D1102:PKAVG_RD': {'goal': np.inf, 'zero_ref': None},
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
        '''
        super().__init__(
            decision_CSETs=decision_CSETs,
            decision_min=decision_min,
            decision_max=decision_max,
            decision_RDs=decision_RDs,
            decision_tols=decision_tols,
            minimize = minimize,
            history_buffer_size = history_buffer_size
            )        
        
        # objective_weight must be defined first
        self.objective_weight = {key:val for key,val in objective_weight.items() if val != 0.}
        self.objective_goal   = {key:val for key,val in objective_goal.items() if key in self.objective_weight.keys()}
        for key,goal in self.objective_goal.items():
            if type(goal) is dict:
                assert "goal" in dict.keys()
                assert "zero_ref" in dict.keys()
            else:
                assert type(goal) in [int,float]
                self[key] = {"goal":goal, "zero_ref":0}
        
        self._check_device_init() 
        self._get_xinit()
        self.objective_norm = {}
        for key in self.objective_weight.keys():
            if type(objective_norm) is dict:
                if key in objective_norm.keys():
                    self.objective_norm[key] = objective_norm[key]
                else:
                    self.objective_norm[key] = None
            else:
                self.objective_norm[key] = None
                
        assert self.objective_goal.keys() == self.objective_weight.keys() == self.objective_norm.keys()
        self.objective_RDs = list(self.objective_goal.keys())
        self._getset_objRD_zero_ref()
        self._getset_objRD_norm()
        
        
        wtot = np.sum(list(self.objective_weight.values()))
        for key in self.objective_weight.keys():
            self.objective_weight[key] /= wtot
            
        self.history["objective_RD"] = {}
        for key in self.objective_goal.keys():
            iratio = key.find('/')
            if iratio!=-1:
                self.history["objective_RD"][key[:iratio]] = []
                self.history["objective_RD"][key[iratio+1:]] = []
            else:
                self.history["objective_RD"][key] = []
            
        objective_p_order = objective_p_order or 2+np.clip(np.log(len(self.objective_goal.keys())),a_min=0,a_max=4)
        self.objective_RD_avg_time = objective_RD_avg_time
        self.apply_bilog = objective_RD_avg_time
            
    def _check_device_runtime(self):
        '''
        check devices status,
        '''
        if caget("ACS_DIAG:CHP:STATE_RD") != 3:
            raise RuntimeError("Chopper blocking.") 
            
            
    def _getset_objRD_zero_ref(self):
        pvRDs = []
        for key, goal_dict in self.objective_goal.items():
            if goal_dict["zero_ref"] is None:
                iratio = key.find('/')
                if iratio!=-1:
                    pvRDs.append(key[:iratio])
                    pvRDs.append(key[iratio+1:])
                else:
                    pvRDs.append(key)
                    
        self.()_check_device_runtime
        RDs,_ = fetch_data(pvRDs,2,abs_z=3)
        self.()_check_device_runtime
        
        i=0
        for key, goal_dict in self.objective_goal.items():
            if goal_dict["zero_ref"] is None:
                iratio = key.find('/')
                if iratio!=-1:
                    goal_dict["zero_ref"] = RDs[i]/RDs[i+1]
                    i+=2
                else:
                    goal_dict["zero_ref"] = RDs[i]
                    i+=1
            
            
    def _getset_objRD_norm(self, pv: str, norm=None):
        
        pvRDs = []
        for key, norm in self.objective_norm.items():
            if norm is None:
                iratio = key.find('/')
                if iratio!=-1:
                    pvRDs.append(key[:iratio])
                    pvRDs.append(key[iratio+1:])
                elif not ("POS" in key or "PHASE" in key):
                    pvRDs.append(key)
                    
        self.()_check_device_runtime
        RDs,_ = fetch_data(pvRDs,2,abs_z=3)
        self.()_check_device_runtime
        
        i=0
        for key, norm in self.objective_norm.items():
            if norm is None:
                iratio = key.find('/')
                if iratio!=-1:
                    self.objective_norm[key] = 0.2*np.abs(RDs[i]/RDs[i+1])
                    i+=2
                elif "POS" in key or "PHASE" in key:
                    self.objective_norm[key] = 1
                elif "FC_" key and "PKAVG" in key:
                    self.objective_norm[key] = 0.2*np.clip(RDs[i],a_min=2,a_max=None)
                    i+=1
                elif "BPM" in key and "MAG_RD" in key:
                    self.objective_norm[key] = 0.2*np.clip(RDs[i],a_min=1e-2,a_max=None)
                    i+=1
                else:
                    raise ValueError("Could not decide normalization factor for "+pv+
                                     " automatically. Please provide normalization factor manually")
        
        
    def _calculate_objectives(self,
                              RD_data,
#                               objective_goal,
#                               objective_weight,
#                               objective_norm,
#                               p_order
                             ):
        
        objective_goal = self.objective_goal
        objective_weight = self.objective_weight
        objective_norm = self.objective_norm
        p_order = self.objective_p_order
        
        eps = 1e-6
        objs = []
        obj_tot = 0
        for key,goal_dict in objective_goal.items():
            
            goal = goal_dict["goal"]
            zero_ref = goal_dict["zero_ref"]
            
            iratio = key.find('/')
            if iratio==-1:
                value = RD_data[key]
            else:
                numerator = key[:iratio]
                denominator = key[iratio+1:]
                value = RD_data[numerator]/RD_data[denominator]
                
            if goal == np.inf:
                obj = (value-zero_ref)/(objective_norm[key] +eps)
                if obj < -1:
                    obj = -1 -np.abs((obj+1)**p_order)/p_order
                objs.append(obj)
                obj_tot += objective_weight[key]*obj
            elif goal == -np.inf:
                obj = (value-zero_ref)/(objective_norm[key] +eps)
                if obj < 1:
                    obj = 1 + np.abs((obj-1)**p_order)/p_order
                objs.append(obj)
                obj_tot += objective_weight[key]*obj
            else:
                obj = 1 -2*np.abs(((value-goal)/(objective_norm[key] +eps))**p_order)
                objs.append(obj)
                obj_tot += objective_weight[key]*obj
                
        if self.apply_bilog:
            obj_tot = np.sign(obj_tot)*np.log(1+np.abs(obj_tot))
                
        return obj_tot, objs

            
    def __call__(self,x,objective_RD_avg_time=None,abs_z=3):
        objective_RD_avg_time = objective_RD_avg_time or self.objective_RD_avg_time
        #set
        self._set_decision(x)
#         ensure_set(self.decision_CSETs,self.decision_RDs,x,self.decision_tols)
        #read
        self._check_device_runtime()
        ave_data, _ = fetch_data(self.objective_RDs+self.decision_RDs,
                                 objective_RD_avg_time,
                                 abs_z=abs_z)
        self._check_device_runtime()
        
        #regularize
        obj_tot,objs = calculate_objective(ave_data) #\
#               +self.regularize(x)
        now = str(datetime.datetime.now())
        now = now[:now.rfind('.')]
        self.history['time'].append(now)
        for pv in self.decision_RDs:
            self.history['decision_RDs'][pv].append(ave_data[pv])
        for pv in self.objective_RDs:
            self.history['objective_RDs'][pv].append(ave_data[pv])
        
        if self.minimize:
            return -obj_tot
        else:
            return  obj_tot
    