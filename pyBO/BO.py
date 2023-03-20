import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as copy
import os
import datetime
import pickle
import time
import sys
import warnings

import concurrent

from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union

from . import model as Model
from . import covfunc
from . import acquisition
from . import util
from .util import Auto
# import warnings
# warnings.filterwarnings("ignore")
dtype = np.float64


class BO:
    def __init__(self,
                 model=None,
                 x=None,
                 y=None,
                 yvar=None,
                 bounds=None,
                 noise_constraint = None,
                 load_log_fname='',
                 batch_size = 1,
                 acquisition_func = None,
                 acquisition_optimize_options = {"num_restarts":100},
                 scipy_minimize_options=None,
                 prior_mean_model=None,
                 path="./log/",
                 tag=""
                ):
        
        if load_log_fname != '':
            self.load_from_log(load_log_fname,
                               acquisition_optimize_options=acquisition_optimize_options,
                               scipy_minimize_options=scipy_minimize_options,
                               prior_mean_model=prior_mean_model,
                               path=path,
                               tag=tag)
            return

        
        self.prior_mean_model = prior_mean_model
        if model is None:
            self.model = Model.GaussianProcess(covfunc.matern52(),prior_mean_model=prior_mean_model)
        else:
            model.prior_mean_model = prior_mean_model
            self.model = model
        
        if bounds is not None:
            self.dim = len(bounds)
            self.bounds = np.array(bounds,dtype=np.float64)
            for i in range(self.dim):
                assert self.bounds[i,0] < self.bounds[i,1]

        if x is None:
            assert y is None
            self.x = None
            self.y = None
        else:
            assert y is not None
            self.x = np.array(x,dtype=dtype)
            self.y = np.array(y,dtype=dtype).reshape(-1,1)
            b,d = x.shape
            if bounds is not None:
                assert d == self.dim    
            self.dim = d

        if yvar is None:
            self.yvar = None
        else:
            raise NotYetImplementedError("pyBO does not support 'yvar' yet. botorch can support this functionality")
            # self.yvar = np.array(yvar,dtype=np.float64)
        
        self.noise_constraint = noise_constraint
        if noise_constraint is not None:
            raise NotYetImplementedError("pyBO does not support 'noise_constraint' yet. botorch can support this functionality")
            
        self.batch_size = batch_size
        if batch_size != 1:
            raise NotYetImplementedError("pyBO support batch_size=1 only yet. botorch can support this functionality")
                 
        if acquisition_func in ["EI","ExpectedImprovement"] or acquisition_func is None:
            self.acquisition_func = acquisition.ExpectedImprovement(model)
        elif acquisition_func in ["UCB","UpperConfidenceBound"]:
            self.acquisition_func = acquisition.UpperConfidenceBound(model)
#         elif acquisition_func in ["KG","qKnowledgeGradient","KnowledgeGradient"]:
#             self.acquisition_func = qKnowledgeGradient
        elif callable(acquisition_func):
            self.acquisition_func = acquisition_func
        else:
            raise TypeError("acquisition_func input is not recognized. currently supports EI, UCB or callable")
        
        self.acquisition_optimize_options = acquisition_optimize_options
        self.scipy_minimize_options = scipy_minimize_options

        if path[-1]!="/":
            path += "/"
        self.path = path
        self.tag = tag

        self.history = []
        self.update_model()
            

    def update_model(self,x1=None,y1=None,y1var=None,
                  x=None,y=None,yvar=None,
                  # recycleGP=Auto,
                  noise_constraint=None,
                  write_log=True,
                  append_hist=True):
#                          , write_gp_on_grid=False, grid_ponits_each_dim=25):
        '''
        recycleGP: bool or Auto
            True: train starting from current GP hyper-parameters
            False: re-train from fresh new GP
        x,y: optional, if want to replace the input/output data
        x1,y1: optional, if newly evaluated data available
        '''
        start_time = time.monotonic()
    
    
        if x is None:
            assert y is None
            x = self.x
            y = self.y
            yvar = self.yvar
        else:
            x = np.atleast_2d(x).astype(dtype)
            y = np.array(y,dtype=dtype).reshape(-1, 1)
            if yvar is not None:
                yvar = np.array(yvar,dtype=dtype).reshape(-1, 1)

        if x1 is not None or y1 is not None:
            assert x1 is not None and y1 is not None
            x1 = np.atleast_2d(x1).astype(dtype)
            y1 = np.array(y1,dtype=dtype).reshape(-1, 1)
            x = np.concatenate((x,x1), axis=0)
            y = np.concatenate((y,y1), axis=0)
            if yvar is not None:
                assert y1var is not None
                y1var = np.array(y1var,dtype=dtype).reshape(-1, 1)
                yvar =np.concatenate((yvar,y1var), axis=0)
                
        if x is None:
            print("data is empty. model fit will not proceed")
            return
        
        if len(self.history)>0:
            if np.all(self.history[-1]['x'] == x):
                print('model is most recent. no update needed')
        
        self.x = x
        self.y = y
        self.yvar = yvar
        noise_constraint = noise_constraint or self.noise_constraint
        
        self.model.fit(x,y)
  
        if append_hist:
            hist = { 'x':x,
                     'y':y,
                     'model':self.model,
                     'model_fit_time':time.monotonic()-start_time,
                     'acquisition':[],
                     'acquisition_args':[],
                     'bounds':[],
                     'x1':[],
                     'query_time':[],
                    }
            self.history.append(hist)
        else:
            hist = self.history[-1]
            hist['x']=x,
            hist['y']=y,
            hist['model']=self.model,
            hist['model_fit_time']=time.monotonic()-start_time,
            
        
        if write_log:
            self.write_log(path=self.path, tag=self.tag)
            
                            
    def query_candidates(self,
                         batch_size =None,
                         bounds = None,
                         acquisition_func=None,
                         acquisition_func_args=None,
                         fixed_values_for_each_dim=None,
                         UCB_beta = None,
                         best_y = None,
                         acquisition_optimize_options = {"num_restarts":100},
                         scipy_minimize_options = None,
                         X_penal = Auto, 
                         L_penal = None,
                         C_penal = None,
                         X_favor = None,
                         L_favor = None,
                         C_favor = None,
                         X_current = Auto,
                         polarity_penalty = None,
                         x_candidate = None,
                         y_candidate_future = None,
                         optmization_stopping_criteria = Auto,
                         write_log =True,
                         ):
        
        start_time = time.monotonic()
        if x_candidate is not None:
            x_candidate = np.atleast_2d(x_candidate)
        if X_current is Auto:
            X_current = x_candidate

    
        if optmization_stopping_criteria is Auto:
            if y_candidate_future is not None:
                optmization_stopping_criteria = y_candidate_future.done
            else:
                optmization_stopping_criteria = None
        
        if bounds is None:
            if self.bounds is None:
                raise ValueError("bounds could not inferred. provide bounds")
            bounds = self.bounds
            
        if scipy_minimize_options is None:
            scipy_minimize_options = {}
    
        if acquisition_func is None:
            acquisition_func = self.acquisition_func
                
        if acquisition_func_args is None:
            acquisition_func_args = {}
        else:
            acquisition_func_args = copy(acquisition_func_args)
        if acquisition_func.name == 'ExpectedImprovement':
            if best_y is None:
                if self.y is not None:
                    best_y = np.max(self.y)
                else:
                    raise ValueError("'best_y' is required for EI")
            acquisition_func_args['best_y'] = best_y
            
        if acquisition_func.name == 'UpperConfidenceBound':
            if UCB_beta is None:
                beta = 9.0
            else:
                beta = UCB_beta
            acquisition_func_args['beta'] = beta
                    

        def acqu(x):
            return acquisition_func(x,**acquisition_func_args)
        penal_favor_acqu_args = self._auto_penal_favor(X_penal,L_penal,C_penal,
                                                       X_favor,L_favor,C_favor,
                                                       X_current,polarity_penalty,
                                                       acquisition=acqu)
        for key, val in penal_favor_acqu_args.items():
            acquisition_func_args[key] = val                                               
                             
                             
        acqu_bounds = bounds                         
        if fixed_values_for_each_dim is not None:
            def acqu(x,fixed_values_for_each_dim=fixed_values_for_each_dim):
                # x_ = np.zeros(dim)
                # ifixed = np.sort(list(fixed_values_for_each_dim.keys()))
                # i0 = -1
                # count = 0
                # for i in ifixed:
                    # print(i0+1,i,i0+1-count,i-count)
                    # x_[i0+1:i] = x[i0+1-count:i-count]
                    # x_[i] = fixed_values_for_each_dim[i]
                    # count += 1
                    # i0 = i
                return float(acquisition_func(self._insert_fixed_values(x,fixed_values_for_each_dim),
                                              **acquisition_func_args))
            acqu_bounds = []
            for i,b in enumerate(bounds):
                if i not in fixed_values_for_each_dim.keys():
                    acqu_bounds.append(b)
            acqu_bounds = np.array(acqu_bounds)

        acquisition_optimize_options['optmization_stopping_criteria'] = optmization_stopping_criteria
        result = util.maximize_with_restarts(acqu,bounds=acqu_bounds,
                                             **acquisition_optimize_options,
                                             **scipy_minimize_options)
        if not hasattr(result,"x"):
            print("=== debug ===")
            print("result",result)
            print("acquisition_optimize_options",acquisition_optimize_options)
            print("scipy_minimize_options",scipy_minimize_options)
            print("acqu_bounds",acqu_bounds)
            if 'x0' in acquisition_optimize_options:
                print("acqu(acquisition_optimize_options['x0'])",acqu(acquisition_optimize_options['x0']))
        if fixed_values_for_each_dim is None:
            x1 = np.atleast_2d(result.x)
        else:
            x1 = np.atleast_2d(self._insert_fixed_values(result.x,fixed_values_for_each_dim)).astype(dtype)
            
        # polarity_penalty does not keep track of past candidate. Only current value used
        # strict zero makes polarity_penalty mis-behave
        is_zero = x1==0
        if x_candidate is not None and np.any(is_zero):
            x1[is_zero] = 1e-6*np.sign(x_candidate[is_zero])   
        
        hist = self.history[-1] 
        hist['acquisition'].append(acquisition_func)
        hist['bounds'].append(bounds)
        hist['acquisition_args'].append(acquisition_func_args)
        hist['x1'].append(x1)
        if write_log:
            self.write_log(path=self.path, tag=self.tag)
        hist['query_time'].append(time.monotonic()-start_time)
        return x1
    
    
    def step(self,
             func_obj,
             x_candidate,
             y_candidate_future,
             batch_size =None,
             bounds = None,
             acquisition_func=None,
             acquisition_func_args=None,
             fixed_values_for_each_dim=None,
             UCB_beta = None,
             best_y = None,
             acquisition_optimize_options = {"num_restarts":100},
             scipy_minimize_options = None,   
             X_penal = Auto,
             L_penal = None,
             C_penal = None,
             X_favor = None,
             L_favor = None,
             C_favor = None,
             X_current = Auto,
             polarity_penalty = None,
             optmization_stopping_criteria = Auto,
             x=None,y=None,yvar=None,
             write_log =True,
            ):

        
        # query new candidate point while evaluating
        x_new_candidate = self.query_candidates(
                         batch_size =batch_size,
                         bounds = bounds,
                         acquisition_func=acquisition_func,
                         acquisition_func_args=acquisition_func_args,
                         fixed_values_for_each_dim=fixed_values_for_each_dim,
                         UCB_beta = UCB_beta,
                         best_y = best_y,
                         acquisition_optimize_options = acquisition_optimize_options,
                         scipy_minimize_options = scipy_minimize_options,
                         X_penal = X_penal, 
                         L_penal = L_penal,
                         C_penal = C_penal,
                         X_favor = X_favor,
                         L_favor = L_favor,
                         C_favor = C_favor,
                         X_current = X_current,
                         polarity_penalty = polarity_penalty,
                         x_candidate = x_candidate,
                         y_candidate_future = y_candidate_future,
                         optmization_stopping_criteria=optmization_stopping_criteria,
                         write_log =write_log,)
                         
        
        new_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        y_new_candidate_future = new_executor.submit(func_obj,x_new_candidate.flatten())
        
        
        # parallel model update
        y_candidate = [y_candidate_future.result()]
        self.update_model(
                          x1=x_candidate,y1=y_candidate,
                          write_log=write_log)
        
        
        return x_new_candidate, y_new_candidate_future
        
        

    
    def remove_qeury_record(self,iquery=None):
        if iquery is None:
            iquery = -1
        del self.history[-1]['acquisition'][iquery] 
        del self.history[-1]['bounds'][iquery] 
        del self.history[-1]['acquisition_args'][iquery] 
        del self.history[-1]['x1'][iquery] 
        

        
    def write_log(self,fname=None,path="./log/",tag=""):
        if path[-1]!="/":
            path += "/"
        if not os.path.isdir(path):
            os.mkdir(path)
        now = str(datetime.datetime.now())
        now = now[:now.rfind(':')].replace(' ','_').replace(':',';')
        if tag != '':
            tag = '_'+tag
        tag = "pyBO_history_"+now+tag
        
        data = []
        for hist in self.history:
            hist_ = {}
            for key,val in hist.items():
                if key == 'model':
                    hist_[key]=val.name
                elif key == 'acquisition':
                    hist_[key] = [acqu.name for acqu in val ]
                else:
                    hist_[key] = val
#             print("hist_",hist_)
            data.append(hist_)
    
        if fname is None or type(fname) is not str:
            pickle.dump(self.history,open(path+tag+".pickle","wb"))
        else:
            pickle.dump(self.history,open(fname+".pickle","wb"))
        
        
    def load_from_log(self,
                      fname=Auto,
                      acquisition_optimize_options = {"num_restarts":100},
                      scipy_minimize_options=None,
                      prior_mean_model=None,
                      prior_mean_model_env=None,
                      prior_mean_model_kwargs={},
                      path="./log/",
                      tag=""):
        if fname is Auto:
            flist = np.sort(os.listdir('./log/'))[::-1]
            found=False
            for f in flist:
                if  f[:13]=='pyBO_history_' and f[-7:]=='.pickle':
                    found=True
                    break
            if not found:
                raise RuntimeError('Auto search of recent log failed. Input log file manually')
            fname = './log/'+f
            
        if fname[-7:]=='.pickle':
            fname = fname[:-7]
            
        hist = pickle.load(open(fname+'.pickle',"rb"))
        self.x = hist[-1]['x']
        self.y = hist[-1]['y']
        if len(hist[-1]['bounds'])>0:
            self.bounds = hist[-1]['bounds'][-1]
        elif len(hist)>1 and len(hist[-2]['bounds'])>0:
            self.bounds = hist[-2]['bounds'][-1]
        else:
            self.bounds = None
        
        if len(hist[-1]['acquisition'])>0:
            acquisition_func = hist[-1]['acquisition'][-1]
        elif len(hist)>1 and len(hist[-2]['acquisition'])>0:
            acquisition_func = hist[-2]['acquisition'][-1]
        else:
            acquisition_func = None
        
        self.__init__(
                 x=self.x,
                 y=self.y,
                 bounds=self.bounds,
                 load_log_fname='',
#                  batch_size = self.batch_size,
                 acquisition_func = acquisition_func,
                 acquisition_optimize_options = acquisition_optimize_options,
                 scipy_minimize_options=scipy_minimize_options,
                 prior_mean_model=prior_mean_model,
                 path=path,
                 tag=tag
                )
        self.history = hist
        self.history[-1]['model']=self.model
        if len(self.history[-1]['acquisition'])>0:
            self.history[-1]['acquisition'][-1]=self.acquisition_func
        elif len(hist)>1 and len(hist[-2]['acquisition'])>0:
            self.history[-2]['acquisition'][-1]=self.acquisition_func
 
    
    def plot_best(self,ax=None):
        y_best_hist = [np.max(self.history[-1]['y'][:i+1]) for i in range(len(self.history[-1]['y']))]
        if ax is None:
            fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(y_best_hist)
        ax.set_xlabel("evaluation budget")
        ax.set_ylabel("objective")
        return ax
    
    
    def plot_obj_history(self,plot_best_only=False,ax=None):
#         y_best_hist_init = [np.min(self.history['y'][0][:i+1]) for i in range(len(self.history['y'][0]))]
        y_best_hist = [np.max(self.history[-1]['y'][:i+1]) for i in range(len(self.history[-1]['y']))]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(y_best_hist,color='C0',label='best objective')
        if not plot_best_only:
            ax1 = ax.twinx()
            ax1.plot(self.history[-1]['y'],color='C1',label='objective history')
            ax1.set_ylabel('objective history')
            ax1.legend(loc='lower left')
        ax.legend(loc='upper right')
        ax.set_xlabel("evaluation budget")
        ax.set_ylabel("best objective")
        return ax


    def plot_model_2D_projection(self,epoch = None,
                                 i_query = None,
                                 dim_xaxis = 0,
                                 dim_yaxis = 1,
                                 bounds = None,
                                 grid_ponits_each_dim = 25, 
                                 project_minimum = False,
                                 project_maximum = False,
                                 project_mean = False,
                                 fixed_values_for_each_dim = None, 
                                 overdrive = False,
                                 fig = None,
                                 ax = None,
                                 colarbar = True,
                                 dtype = dtype):
        '''
        fixed_values_for_each_dim: dict of key: dimension, val: value to fix for that dimension
        '''
        if epoch is None:
            epoch = len(self.history)-1
        if i_query is None:
            i_query = -1
        n_query = len(self.history[epoch]['acquisition'])
        X_penal = None
        X_favor = None
        x1 = None
        if n_query>0:
            x1 = self.history[epoch]['x1'][i_query]
            acquisition_func_args = self.history[epoch]['acquisition_args'][i_query]
            if 'X_penal' in acquisition_func_args:
                X_penal = acquisition_func_args['X_penal']
            if 'X_favor' in acquisition_func_args:     
                X_favor = acquisition_func_args['X_favor']
                
        x = self.history[epoch]['x']
        y = self.history[epoch]['y']
        if bounds is None:
            if self.bounds is not None:
                bounds = self.bounds
            elif n_query==0:
                raise ValueError("bounds could not inferred. It needs to be provided")
            else:
                bounds = self.history[epoch]['bounds'][i_query]
                if bounds is None:
                    raise ValueError("bounds could not inferred. It needs to be provided")
        bounds[:,0] = np.min((bounds[:,0],x.min(axis=0)),axis=0)
        bounds[:,1] = np.max((bounds[:,1],x.max(axis=0)),axis=0)
                
        model = self.history[epoch]['model']
        if model is None or type(model)==str:
            if overdrive:
                raise RuntimeError("model at epoch "+str(epoch)+" is not saved into memory. Turn on 'overdrive' to train a new model using model.GaussianProcess with Matern52 kernel (default).")
            else:
                print("model function at epoch "+str(epoch)+" is not saved into memory. Trying to restore based on training data. Restoration will not be exactly same.")

                model = Model.GaussianProcess(covfunc.matern52())
                model.fit(x,y)
        
        def func(x):
            return model(x,return_var=False)

        if ax is None:
            fig, ax = plt.subplots(figsize=(3.5,3))
            
        util.plot_2D_projection(
                        func,
                        bounds,
                        dim_xaxis,
                        dim_yaxis,
                        grid_ponits_each_dim, 
                        project_minimum,
                        project_maximum,
                        project_mean,
                        fixed_values_for_each_dim, 
                        overdrive,
                        fig,
                        ax,
                        colarbar,
                        dtype)
                         
        ax.scatter(x [:,dim_xaxis],x [:,dim_yaxis],c="b", alpha=0.7,label = "training data")
        
        if X_penal is not None:
            ax.scatter(X_penal[:,dim_xaxis],X_penal[:,dim_yaxis],s=50, c="r", marker='x', label = "penal") 
        if X_favor is not None:
            ax.scatter(X_favor[:,dim_xaxis],X_favor[:,dim_yaxis],s=50, c="g", marker='+', label = "favor") 
        if x1 is not None:
            ax.scatter(x1[:,dim_xaxis],x1[:,dim_yaxis],c="r", alpha=0.7,label = "candidate")
            #print('x1:',x1)
        ax.legend()      
        
                
    def plot_acquisition_2D_projection(self,
                                 acquisition_func = None,
                                 acquisition_func_args = None,
                                 epoch = None,
                                 i_query = None,
                                 dim_xaxis = 0,
                                 dim_yaxis = 1,
                                 bounds = None,
                                 grid_ponits_each_dim = 25, 
                                 project_minimum = False,
                                 project_maximum = False,
                                 project_mean = False,
                                 fixed_values_for_each_dim = None, 
                                 overdrive = False,
                                 fig = None,
                                 ax = None,
                                 colarbar = True,
                                 dtype = dtype):
        '''
        fixed_values_for_each_dim: dict of key: dimension, val: value to fix for that dimension
        '''
        if epoch is None:
            epoch = len(self.history)-1
        if i_query is None:
            i_query = -1
        n_query = len(self.history[epoch]['acquisition'])
        X_penal = None
        X_favor = None
        x1 = None
        if n_query>0:
            x1 = self.history[epoch]['x1'][i_query]
            acquisition_func_args = self.history[epoch]['acquisition_args'][i_query]
            if 'X_penal' in acquisition_func_args:
                X_penal = acquisition_func_args['X_penal']
            if 'X_favor' in acquisition_func_args:     
                X_favor = acquisition_func_args['X_favor']
        
        x = self.history[epoch]['x']
        y = self.history[epoch]['y']
        if bounds is None:
            if self.bounds is not None:
                bounds = self.bounds
            elif n_query==0:
                raise ValueError("bounds could not inferred. It needs to be provided")
            else:
                bounds = self.history[epoch]['bounds'][i_query]
                if bounds is None:
                    raise ValueError("bounds could not inferred. It needs to be provided")
        bounds[:,0] = np.min((bounds[:,0],x.min(axis=0)),axis=0)
        bounds[:,1] = np.max((bounds[:,1],x.max(axis=0)),axis=0)
       
                
        if acquisition_func_args is None:
            if n_query > 0:
                acquisition_func_args = self.history[epoch]['acquisition_args'][i_query]
            else:
                acquisition_func_args = {}
                       
        if acquisition_func is None:
            if n_query > 0:
                acquisition_func = self.history[epoch]['acquisition'][i_query]
            else:
                acquisition_func = 'UpperConfidenceBound'
        if type(acquisition_func) is str:
            model = self.history[epoch]['model']
            if model is None or type(model)==str:
                model = Model.GaussianProcess(covfunc.matern52())
                model.fit(x,y)
            if acquisition_func in ['EI','ExpectedImprovement']:
                acquisition_func = acquisition.ExpectedImprovement(model)
                if 'best_y' not in acquisition_func_args:
                    acquisition_func_args['best_y'] = np.max(y)
            elif acquisition_func in ["UCB","UpperConfidenceBound"]:
                acquisition_func = acquisition.UpperConfidenceBound(model)
                if 'beta' not in acquisition_func_args:
                    acquisition_func_args['beta'] = 9.0

        def func(x):
            return acquisition_func(x,**acquisition_func_args)
                

        if ax is None:
            fig, ax = plt.subplots(figsize=(3.5,3))
            
        util.plot_2D_projection(
                        func,
                        bounds,
                        dim_xaxis,
                        dim_yaxis,
                        grid_ponits_each_dim, 
                        project_minimum,
                        project_maximum,
                        project_mean,
                        fixed_values_for_each_dim, 
                        overdrive,
                        fig,
                        ax,
                        colarbar,
                        dtype)
                         
        ax.scatter(x [:,dim_xaxis],x [:,dim_yaxis],c="b", alpha=0.7,label = "training data")
        if X_penal is not None:
            ax.scatter(X_penal[:,dim_xaxis],X_penal[:,dim_yaxis],s=50, c="r", marker='x', label = "penal") 
        if X_favor is not None:
            ax.scatter(X_favor[:,dim_xaxis],X_favor[:,dim_yaxis],s=50, c="g", marker='+', label = "favor") 
        if x1 is not None:
            ax.scatter(x1[:,dim_xaxis],x1[:,dim_yaxis],c="r", alpha=0.7,label = "candidate")
            #print('x1:',x1)
        ax.legend()                 
                    
            
    def _auto_penal_favor(self,
                          X_penal, 
                          L_penal,
                          C_penal,
                          X_favor,
                          L_favor,
                          C_favor,
                          X_current,
                          polarity_penalty,
                          acquisition = None):
                          
        x = self.x
        _,xdim = x.shape
        if acquisition is None:
            y = self.y
        else:
            y = acquisition(x)
            
        if X_penal is Auto:
            if X_current is not None and y is not None and len(x) >= 4:
                X_penal = X_current
            else:
                X_penal = None
        if X_favor is Auto:
            if X_current is not None and y is not None and len(x) >= 4:
                X_favor = X_current
            else:
                X_favor = None
                
        nsample = min(int(0.1*len(x))+2*self.dim, len(x)-1)  
        if X_penal is not None and x is not None:
            X_penal = np.atleast_2d(X_penal)
            _, d = X_penal.shape
            assert d==xdim
            if L_penal is None or C_penal is None:
                L = np.zeros(X_penal.shape)
                C = np.zeros((len(X_penal),1))
                for i in range(len(X_penal)):
                    # find the proper length scale based on neighboring (of each dim) data 
                    # legnth scale for each dim is nsample-th smallest distance from the choosen point that is X_penal[i,:]
                    L[i,:] = np.partition(np.abs(X_penal[i:i+1,:]-x),nsample,axis=0)[1:nsample+1,:].mean(axis=0)
                    # find neighbor based on the length scale
                    imask = np.argpartition(np.sum( ((X_penal[i:i+1,:]-x)/L[i:i+1,:])**2,axis=1),nsample)[:nsample+1]
                    C[i,0] = y[imask].max()-y[imask].min()
                if L_penal is None:
                    L_penal = 0.5*L
                if C_penal is None:
                    C_penal = 0.2*C
                    
        if X_favor is not None and x is not None:
            X_favor = np.atleast_2d(X_favor)
            _, d = X_favor.shape
            assert d==xdim
            if L_favor is None or C_favor is None:
                L = np.zeros(X_favor.shape)
                C = np.zeros((len(X_favor),1))
                for i in range(len(X_favor)):
                    # find the proper length scale based on neighboring (of each dim) data 
                    # legnth scale for each dim is nsample-th smallest distance from the choosen point that is X_favor[i,:]
                    L[i,:] = np.partition(np.abs(X_favor[i:i+1,:]-x),nsample,axis=0)[1:nsample+1,:].mean(axis=0)
                    # find neighbor based on the length scale
                    imask = np.argpartition(np.sum( ((X_favor[i:i+1,:]-x)/L[i:i+1,:])**2,axis=1),nsample)[:nsample+1]
                    C[i,0] = y[imask].max()-y[imask].min()
                if L_favor is None:
                    L_favor = 5*L
                if C_favor is None:
                    C_favor = 0.1*C
                    
        if X_current is not None:
            if polarity_penalty is Auto:
                if y is not None:
                    polarity_penalty = 0.1*(np.max(y) - np.min(y))
                else:
                    polarity_penalty = 0

                            
        return {
                "X_penal":X_penal,
                "L_penal":L_penal,
                "C_penal":C_penal,
                "X_favor":X_favor,
                "L_favor":L_favor,
                "C_favor":C_favor,
                "X_current":X_current,
                "polarity_penalty":polarity_penalty
               }
    
    def best_sofar(self):
        i=np.argmax(self.y)
        return self.x[i,:], self.y[i]
        

    def _insert_fixed_values(self,x,fixed_values_for_each_dim):
        ifixed = np.sort(list(fixed_values_for_each_dim.keys()))
        if len(x.shape)==1:
            x_ = np.zeros(len(x)+len(fixed_values_for_each_dim))
            i0 = -1
            count = 0
            for i in ifixed:
                x_[i0+1:i] = x[i0+1-count:i-count]
                x_[i] = fixed_values_for_each_dim[i]
                count += 1
                i0 = i
            x_[i0+1:] = x[i0+1-count:]  
            return x_
        else:
            b,d = x.shape
            x_ = np.zeros((b,d+len(fixed_values_for_each_dim)))
            i0 = -1
            count = 0
            for i in ifixed:
                x_[:,i0+1:i] = x[:,i0+1-count:i-count]
                x_[:,i] = fixed_values_for_each_dim[i]
                count += 1
                i0 = i 
            x_[:,i0+1:] = x[:,i0+1-count:]
            return x_
            