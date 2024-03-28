import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy as copy
import os
import datetime
import pickle
import time
import sys
# from warnings import warn as _warn
# def warn(x):
#     return _warn(x)#,stacklevel=2)   

import warnings
# from warnings import warn as _warn
def _warn(message, *args, **kwargs):
    return 'warning: ' +str(message) + '\n'
#     return _warn(x,stacklevel=2)  

warnings.formatwarning = _warn
def warn(x):
    return warnings.warn(x)

import concurrent
from IPython.display import display, HTML

from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union

from . import model as Model
from . import covfunc
from . import acquisition
from . import util

# import warnings
dtype = np.float64


class bo_controller:
    '''
    control budget, intialization sample, acqusition hyper-parameters, global to local optimization transition etc.
    This routine requires objFuncGoal instance input 
    '''
    def __init__(self,
                 obj,         #objFuncGoal instance
                 x0 = None,
                 n_init = None,
                 budget = None,
                 bounds = None,
                 local_optimization = False,
                 local_bound_size = None,
                 global_init_bounds_fraction = None,
                 plot_callbacks = [],
                 ):
        self.obj = obj
        if bounds is None:
            self.bounds = obj.decision_bounds   #obj is objFuncGoals instance
        else:
            self.bounds = np.array(bounds)
            assert self.bounds.ndim == 2
            
        self.ndim   = len(self.bounds)
        self.local_optimization = local_optimization

        if local_bound_size is None:
            self.local_bound_size = 0.1*(self.bounds[:,1] - self.bounds[:,0])
        else:
            self.local_bound_size = np.array(local_bound_size)
        assert np.all(self.local_bound_size>0)
        
        if x0 is not None:
            x0 = np.array(x0)
            assert x0.shape == (self.ndim,)
        elif hasattr(obj,'x0'):
            x0 = obj.x0    #obj is objFuncGoals instance
            assert x0.shape == (self.ndim,)
        self.x0 = x0

        if local_optimization:
            if self.x0 is None:
                raise ValueError('x0 must be provided to start local optimization from begining')   
            local_min   = np.clip(self.x0 -0.5*self.local_bound_size, a_min=self.bounds[:,0], a_max=None)
            local_max   = np.clip(self.x0 +0.5*self.local_bound_size, a_min=None, a_max=self.bounds[:,1])
            self.init_bounds = np.array(list(zip(local_min, local_max)))
            if not n_init:
                n_init = self.ndim
        else:
            n_init = n_init or 8*self.ndim
            if budget:
                if n_init > 0.3*budget:
                    n_init = int(0.3*budget)
#             global_init_bounds_fraction = global_init_bounds_fraction or 0.8 
#             x0 = (self.bounds[:,0]+self.bounds[:,1])/2
#             dx = (self.bounds[:,1]-self.bounds[:,0])*global_init_bounds_fraction
#             local_min   = np.clip(x0 -0.5*dx, a_min=self.bounds[:,0], a_max=None)
#             local_max   = np.clip(x0 +0.5*dx, a_min=None, a_max=self.bounds[:,1])
#             self.init_bounds = np.array(list(zip(local_min, local_max)))
            self.init_bounds = self.bounds
            
        self.n_init = n_init
        self.budget = budget
        
        print(f"init will random sample within the following bounds:")
#         print(f"  n_init: {n_init}")
#         print(f"  bounds: ")
        display(pd.DataFrame(self.init_bounds, columns=['min','max'], index = obj.decision_CSETs))  

        # ===  plots ======
        if len(plot_callbacks) == 0:
#             # decision_CSET keys
#             kdecisionSET = [
#                 [key for key in obj.decision_CSETs if ':PSC_'   in key],   #obj is objFuncGoals instance
#                 [key for key in obj.decision_CSETs if  'SOL_'   in key],
#                 [key for key in obj.decision_CSETs if ':PSQ_'   in key],
#                 [key for key in obj.decision_CSETs if ':PHASE_' in key],
#             ]
#             kLefts  =  []
#             for key in obj.decision_CSETs:
#                 isin = False
#                 for l in kdecisionSET:
#                     if key in l:
#                         isin = True
#                 if not isin:
#                     kLefts.append(key)
#             kdecisionSET.append(kLefts)
#             kdecisionSET = [l for l in kdecisionSET if len(l)>0]
            
            
            # decision_RD keys
            kdecisionRD = [
                [key for key in obj.decision_RDs if ':PSC_'   in key],   #obj is objFuncGoals instance
                [key for key in obj.decision_RDs if  'SOL_'   in key],
                [key for key in obj.decision_RDs if ':PSQ_'   in key],
                [key for key in obj.decision_RDs if ':PHASE_' in key],
            ]
            kLefts  =  []
            for key in obj.decision_RDs:
                isin = False
                for l in kdecisionRD:
                    if key in l:
                        isin = True
                if not isin:
                    kLefts.append(key)
            kdecisionRD.append(kLefts)
            kdecisionRD = [l for l in kdecisionRD if len(l)>0]

            # objective_RD keys   
            kobjectiveRD = [
                [key for key in obj.objective_RDs if ':XPOS_' in key or ':YPOS_' in key],   #obj is objFuncGoals instance
                [key for key in obj.objective_RDs if ':PHASE_' in key],
                [key for key in obj.objective_RDs if ':MAG_' in key],
                [key for key in obj.objective_RDs if ':AVGPK_' in key or ':CURRENT_' in key or ':PKAVG_' in key],
            ]
            kLefts  =  []
            for key in obj.objective_RDs:
                isin = False
                for l in kobjectiveRD:
                    if key in l:
                        isin = True
                if not isin:
                    kLefts.append(key)
            kobjectiveRD.append(kLefts)
            kobjectiveRD = [l for l in kobjectiveRD if len(l)>0]
            
            # callbacks for plot
            plot_callbacks = [obj.plot_time_val(
                                  history=obj.machineIO.history,
                                  keys = kdecisionRD + kobjectiveRD #+ kdecisionSET
                                  ),
                              obj.plot_obj_history(
                                  obj.history['objectives'],
                                  title = 'objectives',
                                  add_y_data = obj.history['objectives']['total'],
                                  add_y_label = 'total obj'
                                  )
                             ] 
        self.plot_callbacks = plot_callbacks
        def obj_w_plot(x):
            return obj(x,callbacks=self.plot_callbacks)
        self.obj_w_plot = obj_w_plot
        self.dp = None
        
    
    def init(self,
             n_init = None,
             init_bounds = None):
        
        if hasattr(self,'bo'):
            warn('bo is already initialized.')
            return
        
        self.dp = display(HTML('<center><h3>init BO: random sampling...</h3></center>'),display_id=True)
        n_init = n_init or self.n_init
        assert n_init>2
        self.n_init = n_init
        if init_bounds is None:
            init_bounds = self.init_bounds
        self.bo, self.X_pending, self.Y_pending_future = runBO(
                                                            self.obj_w_plot,  
                                                            bounds = init_bounds,
                                                            n_init = n_init,
                                                            x0 = self.x0,
                                                            budget = n_init,
                                                            batch_size=1,
                                                            )
            
#     def optimize(self,
#                 niter = None,
#                 fine_tune_niter = None):
#         '''
#         optimize for niter
#         initiaization must be done priori
#         '''
        
#         if niter is None:
#             if self.budget:
#                 niter = self.budget - self.n_init
#             else:
#                 if self.local_optimization:
#                     niter = 10*self.n_init
#                 else:
#                     niter =  3*self.n_init
#                 warn(f'niter is not provided. will use niter = {niter}')
#         assert niter>1
        
#         fine_tune_niter = fine_tune_niter or self.ndim

#         # global optim
#         if not self.local_optimization:
#             global_opt_niter = max(int(0.5*(niter - fine_tune_niter)), 2)
#             if global_opt_niter>0:
#                 def beta_scheduler(n,rate=global_opt_niter):
#                     return 4+5*max(1 - n/rate,0)
#                 self.optimize_global(global_opt_niter, beta_scheduler = beta_scheduler)
#         else:
#             global_opt_niter = 0
            
#         # local optim    
#         local_opt_niter = max(niter - global_opt_niter -fine_tune_niter, 2)
#         if local_opt_niter>0:
#             def beta_scheduler(n,rate=local_opt_niter):
#                 return 1+8*max(1 - n/rate,0)
#             self.optimize_local(local_opt_niter, beta_scheduler = beta_scheduler)
            
#         # fine tune
#         self.fine_tune(fine_tune_niter)
#         self.bo.update_model(X_pending=self.X_pending,
#                              Y_pending_future = self.Y_pending_future)
        
#         for f in self.plot_callbacks:
#             f.close()

        
    def optimize_global(self,
                        niter,
                        bounds=None,
                        beta_scheduler=None,
                        ):
        if self.dp is not None:
            self.dp.update(HTML('<center><h3>Global BO...</h3></center>'))
        else:
            self.dp=display(HTML('<center><h3>Global BO...</h3></center>'),display_id=True)
        
        if not hasattr(self,'bo'):
            raise ValueError('bo is not initializaed')
            
        if beta_scheduler in ['Auto','auto']:
            def beta_scheduler(n):
                return 4+5*max(1 - n/niter,0)
        
        acquisition_func_args = None
        if bounds is None:
            bounds = self.bounds

        for i in range(niter): 
            if beta_scheduler is not None:
                acquisition_func_args = {'beta':beta_scheduler(i)}
            self.X_pending, self.Y_pending_future= self.bo.loop( 
                n_loop=1,
                func_obj = self.obj_w_plot, 
                bounds = bounds,
                acquisition_func_args = acquisition_func_args,
                X_pending = self.X_pending, 
                Y_pending_future = self.Y_pending_future,
                )
            
    def optimize_local(self,
                       niter,
                       local_bound_size=None,
                       beta_scheduler=None,
                       ):
        
        if self.dp is not None:
            self.dp.update(HTML('<center><h3>Local BO...</h3></center>'))
        else:
            self.dp=display(HTML('<center><h3>Local BO...</h3></center>'),display_id=True)
        
        if not hasattr(self,'bo'):
            raise ValueError('bo is not initializaed')
            
        acquisition_func_args = None
        if local_bound_size is None:
            local_bound_size = self.local_bound_size
            
        if beta_scheduler in ['Auto','auto']:
            def beta_scheduler(n):
                return 1+8*max(1 - n/niter,0)
            
        for i in range(niter): 
            x_best,y_best = self.bo.best_sofar()
            local_min   = np.clip(x_best -0.5*local_bound_size, a_min=self.bounds[:,0], a_max=None)
            local_max   = np.clip(x_best +0.5*local_bound_size, a_min=None, a_max=self.bounds[:,1])
            bounds = np.array(list(zip(local_min, local_max)))   
            if beta_scheduler is not None:
                acquisition_func_args = {'beta':beta_scheduler(i)}
            self.X_pending, self.Y_pending_future= self.bo.loop( 
                n_loop=1,
                func_obj = self.obj_w_plot,  
                bounds = bounds,
                acquisition_func_args = acquisition_func_args,
                X_pending = self.X_pending, 
                Y_pending_future = self.Y_pending_future,
                )
    
    def fine_tune(self,niter,
                  local_bound_size=None,
                  ):
        
        if self.dp is not None:
            self.dp.update(HTML('<center><h3>Fine Tune...</h3></center>'))
        else:
            self.dp=display(HTML('<center><h3>Fine Tune...</h3></center>'),display_id=True)
        
        if local_bound_size is None:
            local_bound_size = self.local_bound_size
        for i in range(niter):
            x_best,y_best = self.bo.best_sofar()
            local_min   = np.clip(x_best -0.5*local_bound_size, a_min=self.bounds[:,0], a_max=None)
            local_max   = np.clip(x_best +0.5*local_bound_size, a_min=None, a_max=self.bounds[:,1])
            bounds = np.array(list(zip(local_min, local_max)))
            acquisition_func_args = {'beta':0.01}
            self.X_pending, self.Y_pending_future= self.bo.loop( 
                n_loop=1,  # number of additional optimization interation
                func_obj = self.obj_w_plot,
                bounds = bounds,
                acquisition_func_args = acquisition_func_args,
                X_pending = self.X_pending, 
                Y_pending_future = self.Y_pending_future,
                batch_size = 1,
                C_penal = 0,
                C_favor = 0,
                polarity_penalty = 0,
                )
            
    def finalize(self):
        
        if self.dp is not None:
            self.dp.update(HTML('<center><h3>Finalize...</h3></center>'))
        else:
            self.dp=display(HTML('<center><h3>Finalize...</h3></center>'),display_id=True)
            
        self.bo.update_model(X_pending=self.X_pending,
                             Y_pending_future = self.Y_pending_future)
        
        self.dp.update(HTML('<center><h3>Done.</h3></center>'))
        self.dp = None
    
        
def runBO(
     func_obj,
     bounds,
     n_init,
     budget,
     isfunc_obj_batched = False,
     batch_size = 1,
     x0 = None,
     y0 = None,
     model=None,
#      load_log_fname='',
     ramping_rate = None,
     polarity_change_time = 15,
     polarity_penalty = None,
     acquisition_func = None,
     acquisition_func_args=None,
     acquisition_optimize_options = None,
     UCB_beta = None,
     fixed_values_for_each_dim=None,
     scipy_minimize_options=None,
     optmization_stopping_criteria = None,
     timeout = None,
     prior_mean_model=None,
     yvar=None,
     noise_constraint = None,
     path="./log/",
     tag="",
     write_log = False,
     callbacks = None,
#      plot_history = False,
#      ax = None,
    ):
    
    assert n_init >= batch_size and n_init > 2
    
    if isfunc_obj_batched:
        def func_obj_batched(x):
            return func_obj(x)
    else:
        def func_obj_batched(x):
            x = np.atleast_2d(x)
            y = np.zeros((len(x),1))
            for q in range(len(x)):
                y[q] = func_obj(x[q,:])
            return y
    
    bounds = np.atleast_2d(bounds)
    ndim, _ = bounds.shape
    if polarity_change_time == 0:
        polarity_penalty = 0 
        
    
    if x0 is None:
        assert y0 is None
        x0 = np.atleast_2d(0.5*(bounds[:,0]+bounds[:,1]))
#         print("bounds",bounds)
#         print("x0.shape",x0.shape)
    else:
        x0 = np.atleast_2d(x0)
        assert x0.shape[1] == ndim
        
    if y0 is None:
        n_y0 = 0
    else:
        y0 = np.array(y0).reshape(-1,1)
        assert x0.shape[0] == y0.shape[0]
        n_y0 = y0.shape[0]
        
    Y_pending_future = None
   
    if len(x0) < n_init:
 
        train_x = util.proximal_ordered_init_sampler(
            n_init-len(x0),
            bounds=bounds,
            x0=x0[-1:,:],
            ramping_rate=ramping_rate,
            polarity_change_time=polarity_change_time, 
            method='sobol',
            seed=None)
    
        n_remain =  min(n_init - len(x0),batch_size)
        X_pending = train_x[-n_remain:]
        if len(train_x)>n_remain:
            train_x = np.vstack((x0,train_x[:-n_remain]))
        else:
            train_x = x0
        
        train_y = np.array(func_obj_batched(train_x[n_y0:])).reshape(-1,1)
        if y0 is not None:
            train_y = np.vstack((y0,train_y))
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        Y_pending_future = executor.submit(func_obj_batched,X_pending)  # asynchronous parallel objective evaluation
        
    else:
        X_pending = x0[-1:,:]
        train_x = x0
        if y0 is None:
            train_y = np.array(func_obj_batched(train_x)).reshape(-1,1)
            assert len(train_y) == len(train_x)
        else:
            train_y = y0

    bo = BO(
            x = train_x, 
            y = train_y,
            bounds = bounds,
            batch_size = batch_size,
            acquisition_func = acquisition_func,
            acquisition_optimize_options = acquisition_optimize_options,
            scipy_minimize_options = scipy_minimize_options,
            prior_mean_model = prior_mean_model,
            path = path,
            tag= tag,
            write_log=write_log
           )
    if callbacks is not None:
        for f in callbacks:
            f()
      
    if Y_pending_future is None:
        X_pending = bo.query_candidates( batch_size = min(np.ceil(budget-len(bo.x)),batch_size),
                                         bounds = bounds,
                                         timeout = timeout,
                                         X_pending = X_pending,
                                         ramping_rate = ramping_rate,
                                         polarity_change_time = polarity_change_time,# this is used to impose polarity change penality
                                         write_log = write_log,
                                        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        Y_pending_future = executor.submit(func_obj_batched,X_pending)  # asynchronous parallel objective evaluation
        budget = budget - batch_size
        
    if budget >= len(bo.x) + batch_size:
        X_pending, Y_pending_future= bo.loop( 
                                        n_loop=int(np.ceil((budget-len(bo.x))/batch_size)),
                                        func_obj = func_obj_batched,
                                        isfunc_obj_batched = isfunc_obj_batched,
                                        X_pending = X_pending, 
                                        Y_pending_future = Y_pending_future,
                                        batch_size = batch_size,
                                        timeout = timeout,
                                        ramping_rate = ramping_rate,
                                        polarity_change_time = polarity_change_time, 
                                        callbacks = callbacks,
                                        write_log = write_log,
                                        )
    
#     bo.update_model(X_pending=X_pending,
#                     Y_pending_future = Y_pending_future)
    
    return bo, X_pending, Y_pending_future

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
                 acquisition_optimize_options = None,
                 scipy_minimize_options = None,
                 prior_mean_model = None,
                 path="./log/",
                 tag="",
                 write_log = False,
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
            self.ndim = len(bounds)
            self.bounds = np.array(bounds,dtype=np.float64)
            for i in range(self.ndim):
                assert self.bounds[i,0] < self.bounds[i,1]

        if x is None:
            assert y is None
            self.x = None
            self.y = None
        else:
            assert y is not None
            self.x = np.array(x,dtype=dtype)
            self.y = np.array(y,dtype=dtype).reshape(-1,1)
            self.x, self.y = util.unique_xy(self.x,self.y)
            b,d = x.shape
            if bounds is not None:
                assert d == self.ndim    
            self.ndim = d

        if yvar is None:
            self.yvar = None
        else:
            raise NotYetImplementedError("pyBO does not support 'yvar' yet. botorch can support this functionality")
            # self.yvar = np.array(yvar,dtype=np.float64)
        
        self.noise_constraint = noise_constraint
        if noise_constraint is not None:
            raise NotYetImplementedError("pyBO does not support 'noise_constraint' yet. botorch can support this functionality")
            
            
        self.batch_size = batch_size or 1
#         if batch_size != 1:
#             raise NotYetImplementedError("pyBO support batch_size=1 only yet. botorch can support this functionality")
                 
        if acquisition_func in ["EI","ExpectedImprovement"]:
            self.acquisition_func = acquisition.ExpectedImprovement(self.model)
        elif acquisition_func in ["UCB","UpperConfidenceBound"]  or acquisition_func is None:
            self.acquisition_func = acquisition.UpperConfidenceBound(self.model)
#         elif acquisition_func in ["KG","qKnowledgeGradient","KnowledgeGradient"]:
#             self.acquisition_func = qKnowledgeGradient
        elif callable(acquisition_func):
            self.acquisition_func = acquisition_func
        else:
            raise TypeError("acquisition_func input is not recognized. currently supports EI, UCB or callable")
        
        if acquisition_optimize_options is None:
            acquisition_optimize_options = {"num_restarts":100}
        elif not "num_restarts" in acquisition_optimize_options:
            acquisition_optimize_options["num_restarts"] = 100
        
        self.acquisition_optimize_options = acquisition_optimize_options
        self.scipy_minimize_options = scipy_minimize_options

        if path[-1]!="/":
            path += "/"
        self.path = path
        self.tag = tag

        self.history = []
        self.update_model(write_log=write_log)
            

    def update_model(self,
                     x1=None,y1=None,y1var=None,
                     x=None,y=None,yvar=None,
                     X_pending=None, Y_pending_future=None,
                     noise_constraint=None,
                     write_log=False,
                     append_hist=True,
                     debug = False):
#                          , write_gp_on_grid=False, grid_ponits_each_dim=25):
        '''
        x,y: optional, if want to replace the input/output data
        x1,y1: optional, if newly evaluated data available
        '''
        start_time = time.monotonic()
    
    
        if x is None:
            assert y is None
            x = self.x
            y = self.y
            yvar = self.yvar
            
        if y is not None:
            x = np.atleast_2d(x).astype(dtype)
            y = np.array(y,dtype=dtype).reshape(-1, 1)
            if yvar is not None:
                yvar = np.array(yvar,dtype=dtype).reshape(-1, 1)
                
        if X_pending is not None:
            assert Y_pending_future is not None
            while(not Y_pending_future.done()):
                time.sleep(1)
            x1 = X_pending
            y1 = [Y_pending_future.result()]

        if x1 is not None or y1 is not None:
            assert x1 is not None and y1 is not None
            x1 = np.atleast_2d(x1).astype(dtype)
            y1 = np.array(y1,dtype=dtype).reshape(-1, 1)
            if x is None:
                x = x1
                y = y1
            else:
                x = np.concatenate((x,x1), axis=0)
                y = np.concatenate((y,y1), axis=0)
            if yvar is not None:
                assert y1var is not None
                y1var = np.array(y1var,dtype=dtype).reshape(-1, 1)
                yvar =np.concatenate((yvar,y1var), axis=0)
                
        if x is None:
            print("data is empty. model fit will not proceed")
            return
        
#         if len(self.history)>0:
#             if self.history[-1]['x'].shape == x.shape:
#                 if np.all(self.history[-1]['x'] == x):
#                     print('model is most recent. no update needed')
#                     return
        
        self.x = x
        self.y = y
        self.yvar = yvar
        noise_constraint = noise_constraint or self.noise_constraint
        
        self.model.fit(x,y)
        
        if debug:
            print("--in pyBO update model--")
            print("  [debug]append_hist",append_hist)
            print("  [debug]self.history[-1]['acquisition_args']:")
            print(self.history[-1]['acquisition_args'])
  
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
            hist['model']=copy(self.model),
            hist['model_fit_time']=time.monotonic()-start_time,
        
        if write_log:
            self.write_log(path=self.path, tag=self.tag)
            
            
        if debug:
            print("--in pyBO update model--2nd")
            print("  [debug]self.history[-2]['acquisition_args']:")
            print(self.history[-2]['acquisition_args'])
            
                            
    def query_candidates(self,
                         batch_size = None,
                         bounds = None,
                         acquisition_func=None,
                         acquisition_func_args=None,
                         fixed_values_for_each_dim=None,
                         UCB_beta = None,
                         best_y = None,
                         acquisition_optimize_options = None,
                         scipy_minimize_options = None,
                         timeout = None,
                         X_pending = None,
                         Y_pending_future = None,
                         X_penal = None, 
                         L_penal = None,
                         C_penal = None,
                         X_favor = None,
                         L_favor = None,
                         C_favor = None,
                         polarity_penalty = None,
                         ramping_rate = None,
                         polarity_change_time = None,
                         optmization_stopping_criteria = None,
                         write_log =False,
                         debug = False,
                         ):
               
        start_time = time.monotonic()  
        batch_size = batch_size or self.batch_size or 1
    
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
            if UCB_beta is not None:
                beta = UCB_beta
            elif 'beta' in acquisition_func_args:
                beta = acquisition_func_args['beta']
            else:
                beta = 9.0
            acquisition_func_args['beta'] = beta
        
        if acquisition_optimize_options is None:
            acquisition_optimize_options = {"num_restarts":100}
        elif not "num_restarts" in acquisition_optimize_options:
            acquisition_optimize_options["num_restarts"] = 100
              
        if X_pending is not None:
            X_pending = np.atleast_2d(X_pending)
        
        x1 = np.zeros((batch_size,self.ndim),dtype=dtype)
        for q in range(batch_size):   
            def acqu(x):
                return acquisition_func(x,**acquisition_func_args)
            penal_favor_acqu_args = self._auto_penal_favor(X_penal,L_penal,C_penal,
                                                           X_favor,L_favor,C_favor,
                                                           X_pending,polarity_penalty,
                                                           bounds = bounds,
                                                           acquisition=acqu)
            
            if debug:
                print("--in pyBO acqu query--")
                print("  [debug]acquisition_func_args",acquisition_func_args)
                print("  [debug]penal_favor_acqu_args",penal_favor_acqu_args)
                
                
            for key, val in penal_favor_acqu_args.items():
                acquisition_func_args[key] = val
                
            def acqu(x):
                return acquisition_func(x,**acquisition_func_args)
                
            acqu_bounds = np.array(bounds)
            if fixed_values_for_each_dim is not None:
                def acqu(x,fixed_values_for_each_dim=fixed_values_for_each_dim):
                    return float(acquisition_func(self._insert_fixed_values(x,fixed_values_for_each_dim),
                                                  **acquisition_func_args))
                acqu_bounds = []
                for i,b in enumerate(bounds):
                    if i not in fixed_values_for_each_dim.keys():
                        acqu_bounds.append(b)
                acqu_bounds = np.array(acqu_bounds)
                
            if optmization_stopping_criteria is None:
                optmization_stopping_criteria = []
                if batch_size>1:
                    if timeout is None:
                        warn("'timeout' input is recommend for multi-batch candidate querry. By default timeout = 5 sec for each candidate search")
                        timeout = 5*batch_size  
                    optmization_stopping_criteria.append(util.timeout(timeout))
                if Y_pending_future is not None:
                    optmization_stopping_criteria.append(Y_pending_future.done)
                if len(optmization_stopping_criteria) == 0:
                    optmization_stopping_criteria = None

            acquisition_optimize_options['optmization_stopping_criteria'] = optmization_stopping_criteria
            if X_pending is not None:
                if X_pending.ndim != 2 or len(X_pending)<1:
                    print("in acqu query X_pending",X_pending)
                acquisition_optimize_options['x0'] = np.clip(X_pending[-1,:] + 0.02*np.random.randn()*(acqu_bounds[:,1]-acqu_bounds[:,0]),
                                                             a_min=acqu_bounds[:,0],
                                                             a_max=acqu_bounds[:,1],)
            result = util.maximize_with_restarts(acqu,bounds=acqu_bounds,
                                                 **acquisition_optimize_options,
                                                 **scipy_minimize_options)
            if not hasattr(result,"x"):
                if X_pending is not None:
                    warn('Scipy minimize could not find new candidate. Old candidate will be used instead')
                    result.x = X_pending[-1,:]
                else:
                    warn('Scipy minimize could not find new candidate. Random candidate will be used instead')
                    result.x = np.random.rand(len(acqu_bounds))*(acqu_bounds[:,1]-acqu_bounds[:,0]) +acqu_bounds[:,0] 
                    
#                 print("=== debug ===")
#                 print("result",result)
#                 print("acquisition_optimize_options",acquisition_optimize_options)
#                 print("scipy_minimize_options",scipy_minimize_options)
#                 print("acqu_bounds",acqu_bounds)
#                 if 'x0' in acquisition_optimize_options:
#                     print("acqu(acquisition_optimize_options['x0'])",acqu(acquisition_optimize_options['x0']))
            if fixed_values_for_each_dim is None:
                x1[q,:] = result.x.flatten()
            else:
                x1[q,:] = self._insert_fixed_values(result.x,fixed_values_for_each_dim).flatten()
            
            # polarity_penalty does not keep track of past candidate.
            # strict zero makes polarity_penalty mis-behave
            is_zero = x1[q,:]==0
            if X_pending is not None:
                x1[q,is_zero] = 1e-6*np.sign(X_pending[-1,is_zero])
                
            hist = self.history[-1] 
            hist['acquisition'].append(copy(acquisition_func))
            hist['bounds'].append(copy(bounds))
            hist['acquisition_args'].append(copy(acquisition_func_args))
            hist['x1'].append(copy(x1[q:q+1,:]))
            if write_log:
                self.write_log(path=self.path, tag=self.tag)
            hist['query_time'].append(time.monotonic()-start_time)
            
            
            if X_pending is None:
                X_pending = x1[:q+1,:]
            else:
                X_pending = np.vstack((X_pending,x1[q:q+1,:]))
                
#         return x1
#         fig,axs = plt.subplots(1,2,figsize=(8,3))
#         ax=axs[0]
#         ax.plot(x1[:,0],x1[:,1],alpha=0.5,c='r')
#         print("before x1.shape",x1.shape)
#         time.sleep(1)
#         ax.scatter(x1[:,0],x1[:,1],s=np.linspace(4,16,len(x1)))
#         ax.hlines(0,bounds[0,0],bounds[0,1],ls='--',color='k')
#         ax.vlines(0,bounds[1,0],bounds[1,1],ls='--',color='k')
#         ax.scatter(X_pending[:1,0],X_pending[:1,1],c='k')
#         ax.set_xlim(bounds[0,0],bounds[0,1])
#         ax.set_ylim(bounds[1,0],bounds[1,1])
#         ax=axs[1]
#         x1 = util.order_samples(x1,
#                                x0=X_pending[:1,:],
#                                ramping_rate=ramping_rate,
#                                polarity_change_time=polarity_change_time)
#         print("after x1.shape",x1.shape)
#         time.sleep(1)
#         ax.plot(x1[:,0],x1[:,1],alpha=0.5,c='r')
#         ax.scatter(x1[:,0],x1[:,1],s=np.linspace(4,16,len(x1)))
#         ax.hlines(0,bounds[0,0],bounds[0,1],ls='--',color='k')
#         ax.vlines(0,bounds[1,0],bounds[1,1],ls='--',color='k')
#         ax.scatter(X_pending[:1,0],X_pending[:1,1],c='k')
#         ax.set_xlim(bounds[0,0],bounds[0,1])
#         ax.set_ylim(bounds[1,0],bounds[1,1])
        x1 = np.unique(x1, axis=0)

        return  util.order_samples(x1,
                                   x0=X_pending[:1,:],
                                   ramping_rate=ramping_rate,
                                   polarity_change_time=polarity_change_time)
    
    
    def loop(self,
             n_loop,
             func_obj,
             X_pending,
             Y_pending_future,
             isfunc_obj_batched = False,
             batch_size = None,
             bounds = None,
             acquisition_func=None,
             acquisition_func_args=None,
             fixed_values_for_each_dim=None,
             UCB_beta = None,
#              best_y = None,
             acquisition_optimize_options = None,
             scipy_minimize_options = None,   
             timeout = None,
             X_penal = None,
             L_penal = None,
             C_penal = None,
             X_favor = None,
             L_favor = None,
             C_favor = None,
             polarity_penalty = None,
             ramping_rate = None,
             polarity_change_time = None,
             optmization_stopping_criteria = None,
             callbacks = None,
             write_log =False,
             debug = False,
             update_model = False,
            ):
        
        if isfunc_obj_batched:
            func_obj_batched = func_obj
        else:
            def func_obj_batched(x):
                x = np.atleast_2d(x)
    #             y = np.zeros(len(x))
                y = np.zeros((len(x),1))
                for q in range(len(x)):
                    y[q,:] = func_obj(x[q,:])
                return y       
 
#         for i_loop in util.progressbar(range(n_loop)):
        for i_loop in range(n_loop):
            X_pending_new = self.query_candidates(
                                 batch_size =batch_size,
                                 bounds = bounds,
                                 acquisition_func=acquisition_func,
                                 acquisition_func_args=acquisition_func_args,
                                 fixed_values_for_each_dim=fixed_values_for_each_dim,
                                 UCB_beta = UCB_beta,
                                 best_y = None,
                                 acquisition_optimize_options = acquisition_optimize_options,
                                 scipy_minimize_options = scipy_minimize_options,
                                 timeout = timeout,
                                 X_pending = X_pending,
                                 Y_pending_future = Y_pending_future,
                                 X_penal = X_penal, 
                                 L_penal = L_penal,
                                 C_penal = C_penal,
                                 X_favor = X_favor,
                                 L_favor = L_favor,
                                 C_favor = C_favor,
                                 polarity_penalty = polarity_penalty,
                                 ramping_rate=ramping_rate,
                                 polarity_change_time=polarity_change_time,
                                 optmization_stopping_criteria=optmization_stopping_criteria,
                                 write_log =write_log,
                                 debug = debug,)


            # parallel model update
            self.update_model(
                              X_pending = X_pending,
                              Y_pending_future = Y_pending_future,
                              write_log = write_log,
                              debug = debug,
                              )
            
            if callbacks is not None:
                for f in callbacks:
                    f()
                
                
            new_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            Y_pending_future_new = new_executor.submit(func_obj_batched,X_pending_new)

            X_pending = X_pending_new
            Y_pending_future = Y_pending_future_new
            
        if update_model:
            self.update_model(X_pending=X_pending,
                              Y_pending_future = Y_pending_future,
                              write_log = write_log)
        
        return X_pending, Y_pending_future
        

    
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
        
        tag = tag or ""
        if tag == "":
            tag = "pyBO_history_"+now
        
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
                      fname=None,
                      acquisition_optimize_options = None,
                      scipy_minimize_options=None,
                      prior_mean_model=None,
                      prior_mean_model_env=None,
                      prior_mean_model_kwargs={},
                      path="./log/",
                      tag=""):
        if fname is None:
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
        
        if acquisition_optimize_options is None:
            acquisition_optimize_options = {"num_restarts":100}
        elif "num_restarts" not in acquisition_optimize_options:
            acquisition_optimize_options['num_restarts'] = 100
            
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
                                 project_maximum = True,
                                 project_mode = None,
                                 fixed_values_for_each_dim = None, 
                                 overdrive = False,
                                 fig = None,
                                 ax = None,
                                 colorbar = True,
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
            fig, ax = plt.subplots(figsize=(3.5,3),dpi=96)
            
        util.plot_2D_projection(
                        func=func,
                        bounds=bounds,
                        dim_xaxis=dim_xaxis,
                        dim_yaxis=dim_yaxis,
                        grid_ponits_each_dim=grid_ponits_each_dim, 
                        project_maximum=project_maximum,
                        project_mode=project_mode,
                        fixed_values_for_each_dim=project_mode, 
                        overdrive=overdrive,
                        fig=fig,
                        ax=ax,
                        colorbar=colorbar,
                        dtype=dtype)
                         
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
                                 project_maximum = True,
                                 project_mode = None,
                                 fixed_values_for_each_dim = None, 
                                 overdrive = False,
                                 fig = None,
                                 ax = None,
                                 colorbar = True,
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
        if acquisition_func_args is None:
            acquisition_func_args = {}
        if n_query>0:
            x1 = self.history[epoch]['x1'][i_query]
            acquisition_func_args_ = copy(self.history[epoch]['acquisition_args'][i_query])
            for k,v in acquisition_func_args.items():
                acquisition_func_args_[k]=v
            acquisition_func_args = acquisition_func_args_
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
                acquisition_func = self.acquisition_func
#                 acquisition_func = 'UpperConfidenceBound'
        if type(acquisition_func) is str:
            model = self.history[epoch]['model']
            if model is None or type(model)==str:
                model = Model.GaussianProcess(covfunc.matern52())
                model.fit(x,y)
            if acquisition_func in ['EI','ExpectedImprovement']:
                acquisition_func = acquisition.ExpectedImprovement(model)
                
            elif acquisition_func in ["UCB","UpperConfidenceBound"]:
                acquisition_func = acquisition.UpperConfidenceBound(model)
                
                    
        if acquisition_func.name == "ExpectedImprovement":
            if 'best_y' not in acquisition_func_args:
                acquisition_func_args['best_y'] = np.max(y) 
        elif acquisition_func.name == "UpperConfidenceBound":
            if 'beta' not in acquisition_func_args:
                acquisition_func_args['beta'] = 9.0

        def func(x):
            return acquisition_func(x,**acquisition_func_args)
                

        if ax is None:
            fig, ax = plt.subplots(figsize=(3.5,3),dpi=96)
            
        cs = util.plot_2D_projection(
                        func=func,
                        bounds=bounds,
                        dim_xaxis=dim_xaxis,
                        dim_yaxis=dim_yaxis,
                        grid_ponits_each_dim=grid_ponits_each_dim, 
                        project_maximum=project_maximum,
                        project_mode=project_mode,
                        fixed_values_for_each_dim=fixed_values_for_each_dim, 
                        overdrive=overdrive,
                        fig=fig,
                        ax=ax,
                        colorbar=colorbar,
                        dtype=dtype)
                         
        ax.scatter(x [:,dim_xaxis],x [:,dim_yaxis],c="b", alpha=0.7,label = "training data")
        if X_penal is not None:
            ax.scatter(X_penal[:,dim_xaxis],X_penal[:,dim_yaxis],s=50, c="r", marker='x', label = "penal") 
        if X_favor is not None:
            ax.scatter(X_favor[:,dim_xaxis],X_favor[:,dim_yaxis],s=50, c="g", marker='+', label = "favor") 
        if x1 is not None:
            ax.scatter(x1[:,dim_xaxis],x1[:,dim_yaxis],c="r", alpha=0.7,label = "candidate")
            #print('x1:',x1)
        ax.legend()
        return cs
                    
            
    def _auto_penal_favor(self,
                          X_penal, 
                          L_penal,
                          C_penal,
                          X_favor,
                          L_favor,
                          C_favor,
                          X_pending,
                          polarity_penalty,
                          bounds = None,
                          acquisition = None):
                          
        x = copy(self.x)
        _,xdim = x.shape
        if bounds is None:
            inbounds = [True]*_
        else:
            bounds = np.array(bounds)
            inbounds = np.logical_and( np.all(bounds[:,0][None,:] <= x,axis=1),  np.all(x<=bounds[:,1][None,:],axis=1))
#             print('inbounds',inbounds)
            x = x[inbounds]
        
        if acquisition is None:
            y = self.y[inbounds]
        else:
            y = acquisition(x)
            
        if X_penal is None:
            if X_pending is not None:
                X_penal = X_pending
        if X_favor is None:
            if X_pending is not None:
                X_favor = X_pending[-1:,:]
            else:
                X_favor = None
                
        nsample = min(int(0.5*len(x))+2*self.ndim, len(x)-1)  
        if nsample < 2:
            return {
                "X_penal":None,
                "L_penal":None,
                "C_penal":None,
                "X_favor":None,
                "L_favor":None,
                "C_favor":None,
                "X_pending":None,
                "polarity_penalty":None
               }
            
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
                    tmp = np.partition(np.abs(X_penal[i:i+1,:]-x),nsample,axis=0)[1:nsample+1,:]
                    L[i,:] = tmp.mean(axis=0) + tmp.std(axis=0)
                    # find neighbor based on the length scale
                    imask = np.argpartition(np.sum( ((X_penal[i:i+1,:]-x)/(L[i:i+1,:]+1e-3))**2,axis=1),nsample)[:nsample+1]
#                     print('x[imask]',x[imask])
#                     print('x',x)
#                     print('y',y)
#                     print('y[imask]',y[imask])
                    C[i,0] = y[imask].max()-y[imask].min()
                if L_penal is None:
                    L_penal = 0.4*L
                if C_penal is None:
                    C_penal = C
#                 if L_penal is None:
#                     L_penal = 0.5*L
#                 if C_penal is None:
#                     C_penal = 0.4*C
                    
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
                    tmp = np.partition(np.abs(X_favor[i:i+1,:]-x),nsample,axis=0)[1:nsample+1,:]
                    L[i,:] = tmp.mean(axis=0) + tmp.std(axis=0)
                    # find neighbor based on the length scale
                    imask = np.argpartition(np.sum( ((X_favor[i:i+1,:]-x)/(L[i:i+1,:]+1e-3))**2,axis=1),nsample)[:nsample+1]
                    C[i,0] = y[imask].max()-y[imask].min()
#                 if L_favor is None:
#                     L_favor = 5*L
#                 if C_favor is None:
#                     C_favor = 0.1*C
                if L_favor is None:
                    L_favor = 4*L
                if C_favor is None:
                    C_favor = 0.1*C
                    
        if X_pending is not None:
            if y is not None:
                polarity_penalty = 0.2*(np.max(y) - np.min(y))
            else:
                polarity_penalty = 0
                    
#         print({
#                 "X_penal":X_penal,
#                 "L_penal":L_penal,
#                 "C_penal":C_penal,
#                 "X_favor":X_favor,
#                 "L_favor":L_favor,
#                 "C_favor":C_favor,
#                 "X_pending":X_pending,
#                 "polarity_penalty":polarity_penalty
#                })

                            
        return {
                "X_penal":X_penal,
                "L_penal":L_penal,
                "C_penal":C_penal,
                "X_favor":X_favor,
                "L_favor":L_favor,
                "C_favor":C_favor,
                "X_pending":X_pending,
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
            
