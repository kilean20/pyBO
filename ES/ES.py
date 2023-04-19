import numpy as np
from typing import List, Union, Optional, Callable, Tuple
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from copy import deepcopy as copy

class ES():
    def __init__(self,
                 loss_func: Callable,
                 aES: List[float],
                 x0: List[float],
                 y0: Optional[float] = None,
                 x_bounds: Optional[List[Tuple[float]]] = None,
                 wES: Optional[float] = None,
                 kES: Optional[float] = None,
                 minimize: Optional[bool] = True,
                 history_buffer_size: Optional[int] = 100000,
                 ):
        '''
        aES [array of shape (dim,)]: decision parameter oscillation size
        x0 [array of shape (dim,)]: initial decision parameter
        y0 [float or array of shape (1,)]: initial loss. If None, loss_func(x0) will be evaluated.
        wES [array of shape (dim,)]: decision parameter oscillation frequency in unit of 1
        kES [array of shape (dim,)]: ES gain parameter
        '''
        self.loss_func = loss_func

        # dim = number of decision parameter
        self.dim = len(aES)
        self.aES = np.array(aES)
        assert self.aES.ndim == 1
        
        if wES is None:
            self.wES = 2*np.pi*(0.5*(np.arange(self.dim)+0.5)/self.dim+0.5)/10    # frequency
        else:
            self.wES = np.array(wES)
        assert self.wES.ndim == 1
        
        self.x = x0
        if y0 is None:
            self.y = self.loss_func(self.x)
        else:
            self.y = y0
        self.t = 0
        self.x_bounds = x_bounds 
        if self.x_bounds is not None:
            self.x_bounds = np.array(self.x_bounds)
            assert np.all(self.x_bounds[:,0]<=self.x) and np.all(self.x<=self.x_bounds[:,1])
            
        
        if type(kES) is float or type(kES) is int:
            self.kES = kES*np.ones(self.dim)
        elif kES is None:
            self.kES = 0.2*self.wES
        else:
            self.kES = np.array(kES)
        assert self.kES.ndim == 1
        
        self.minimize = minimize
                   
        self.history = {
                        't':[self.t],
                        'y':[self.y],
                       }   
        for i in range(self.dim):
            self.history['aES'+str(i)]=[self.aES[i]]
            self.history['kES'+str(i)]=[self.kES[i]]
            
        self.debug = []
#         self._no_auto_aES = False

            
        self.history_buffer_size = history_buffer_size


    def __call__(self,
                 n_iter=1,
                 aES = None,
                 kES = None,
                 auto_kES = False,
                 auto_kES_kwargs={'phase_fraction':1.0},
                 auto_aES = False,
                 auto_aES_kwargs={'budget':None},
                ):
        
        if auto_kES:
            if auto_kES_kwargs is None:
                auto_kES_kwargs = {}
            self.auto_kES(**auto_kES_kwargs)
        if auto_aES:
            if auto_aES_kwargs is None:
                auto_aES_kwargs = {}
            self.auto_aES(**auto_aES_kwargs)
        
            
        for n in range(n_iter):
            
            if aES is None:
                aES_ = self.aES
            else:
                aES_ = aES
            if kES is None:
                kES_ = self.kES
            else:
                kES_ = kES
                
            if self.minimize:
                self.x += (aES_*self.wES)**0.5*np.sin(self.t*self.wES +kES_*self.y)
            else:
                self.x += (aES_*self.wES)**0.5*np.sin(self.t*self.wES -kES_*self.y)
            if self.x_bounds is not None:
                self.x = np.clip(self.x, a_min=self.x_bounds[:,0], a_max=self.x_bounds[:,1])
            self.y = self.loss_func(self.x) 

            self.t += 1
            self.history['t'].append(self.t)
            self.history['y'].append(self.y)    
            for i in range(self.dim):
                self.history['aES'+str(i)].append(aES_[i])
                self.history['kES'+str(i)].append(kES_[i])
#                 self.history['dft'+str(i)].append(dfts[i])
                
#             for i in range(int(256/2+1)):
#                 self.history['fft'+str(i)]=ffts[i]
                
            if len(self.history['t']) > self.history_buffer_size:
                print('ES history data size is over the history_buffer_size. Removing 10% of the old data')
                iremove = int(0.1*self.history_buffer_size)
                for key,val in self.history.items():
                    self.history[key] = val[iremove:]
                    
            if auto_kES:
                self.auto_kES(**auto_kES_kwargs)
            if auto_aES:
                self.auto_aES(**auto_aES_kwargs)

            
    def auto_kES(self, phase_fraction=1.0):
        '''
        automatically adjust kES, the hyper-parameter of gain
        '''
        if len(self.history['y'])<128:
            return
            
        if type(phase_fraction) is float:
            y = self.history['y'][-128:]
            p = np.polyfit(np.arange(0,len(y)), y, 1)
            std = np.std(y)
            variation = max(np.abs(p[0]),std)
            if np.abs(p[0])>std:
                print('!!!')
#             variation = np.std(self.history['y'][-128:])
            new_kES = phase_fraction*self.wES/(variation+1e-3)
            self.kES += np.clip(new_kES-self.kES, a_min=-0.02*self.kES, a_max=0.02*self.kES)

                
    def auto_aES(self, budget=None):
#         if self._no_auto_aES:
#                 return
        budget = budget or min(64*self.dim,1024)
        if len(self.history['y']) > budget:
            budget = min(len(self.history['y']), 64*self.dim)
        y = np.array(self.history['y'][-budget:]) 
        y -= y.mean()
        T = np.arange(budget)
        def dft_neg_amplitude(nu_):
            return -(np.sum(y*np.sin(2*np.pi*nu_*T))**2 + np.sum(y*np.cos(2*np.pi*nu_*T))**2)**0.5/budget
        
        if hasattr(self,'wES_detune'):
            for i,w in enumerate(self.wES_detune/(2*np.pi)):
                result = minimize(dft_neg_amplitude,w,bounds=( (w-1/(20*self.dim), w-1/(20*self.dim)),))
                if result.success:
                    self.wES_detune[i] = result.x[0]
        else:
#             key = input('ES will run with current hyper-parameters for '+str(budget)+' iteration to optimize the hyper-parameter aES. Yes/No')
#             if key in ["No","no"]:
#                 self._no_auto_aES = True
#                 return   
            print('ES will run with current hyper-parameters for '+str(budget)+' iteration to optimize the hyper-parameter aES')
            self(budget-1,kES=np.zeros(self.dim),auto_kES=False)
            self.wES_detune = copy(self.wES)
        
        ffts = np.abs(np.fft.rfft(y)).astype(np.float64)/budget
        mask = [True]*len(ffts)
        for i in np.round(self.wES_detune/(2*np.pi)*2*len(ffts)).astype(int):
            mask[i] = False
        if np.sum(mask)>2:
            ref = ffts[mask].mean() 
        else:
            ref = np.min(ffts)
        
        for i,nu in enumerate(self.wES_detune):
            self.aES[i] *= np.clip(ref/(-dft_neg_amplitude(nu/(2*np.pi))+1e-6),a_min=0.98,a_max=1.02)        

        
            
            