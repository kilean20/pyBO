import warnings
import sys
import time

import numpy as np
from scipy import optimize
from scipy.optimize import OptimizeResult
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

Auto = 'Auto'


def defaultKeyVal(d,k,v):
  if k in d.keys():
    return d[k]
  else:
    return v
    

class dictClass(dict):
  """ 
  This class is essentially a subclass of dict
  with attribute accessors, one can see which attributes are available
  using the `keys()` method.
  """
  def __dir__(self):
      return self.keys()
    
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)
  if dict==None:
    __setattr__ = {}.__setitem__
    __delattr__ = {}.__delitem__
  else:
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

  def __repr__(self):
    if self.keys():
      L = list(self.keys())
      L = [str(L[i]) for i in range(len(L))]
      m = max(map(len, L)) + 1
      f = ''
      for k, v in self.items():
        if isinstance(v,dict):
          f = f + '\n'+str(k).rjust(m) + ': ' + repr(k) + ' class'
      return f
    else:
      return self.__class__.__name__ + "()"

  def find_key(self,val):
    if val==None :
      return
    for k in self.keys():
      if self[k]==val:
        return 
    

class OptimizationTimeoutError(Exception):
    r"""Exception raised when optimization times out."""

    def __init__(
        self, /, *args: Any, current_x: np.ndarray, runtime: float
    ) -> None:
        r"""
        Args:
            current_x: A numpy array representing the current iterate.
            runtime: The total runtime in seconds after which the optimization
                timed out.
        """
        self.current_x = current_x
        self.runtime = runtime
        
        
class NotYetImplementedError(Exception):
    r"""Exception raised when optimization times out."""

    def __init__(self) -> None:
        pass


def minimize_with_timeout(
    fun: Callable,
    x0: np.ndarray,
    method: Optional[str] = None,
    jac: Optional[Union[str, Callable, bool]] = None,
    hess: Optional[Union[str, Callable, optimize.HessianUpdateStrategy]] = None,
    hessp: Optional[Callable] = None,
    bounds: Optional[Union[Sequence[Tuple[float, float]], optimize.Bounds]] = None,
    constraints=(),  # Typing this properly is a s**t job
    tol: Optional[float] = None,
    callback: Optional[Callable] = None,
    options: Optional[Dict[str, Any]] = None,
    timeout_sec: Optional[float] = None,
) -> optimize.OptimizeResult:
    r"""Wrapper around scipy.optimize.minimize to support timeout.
    This method calls scipy.optimize.minimize with all arguments forwarded
    verbatim. The only difference is that if provided a `timeout_sec` argument,
    it will automatically stop the optimziation after the timeout is reached.
    Internally, this is achieved by automatically constructing a wrapper callback
    method that is injected to the scipy.optimize.minimize call and that keeps
    track of the runtime and the optimization variables at the current iteration.
    """
    if timeout_sec:

        start_time = time.monotonic()
        callback_data = {"num_iterations": 0}  # update from withing callback below

        def timeout_callback(xk: np.ndarray) -> bool:
            runtime = time.monotonic() - start_time
            callback_data["num_iterations"] += 1
            if runtime > timeout_sec:
                raise OptimizationTimeoutError(current_x=xk, runtime=runtime)
            return False

        if callback is None:
            wrapped_callback = timeout_callback

        elif callable(method):
            raise NotImplementedError(
                "Custom callable not supported for `method` argument."
            )

        elif method == "trust-constr":  # special signature

            def wrapped_callback(
                xk: np.ndarray, state: optimize.OptimizeResult
            ) -> bool:
                # order here is important to make sure base callback gets executed
                return callback(xk, state) or timeout_callback(xk=xk)

        else:

            def wrapped_callback(xk: np.ndarray) -> None:
                timeout_callback(xk=xk)
                callback(xk)

    else:
        wrapped_callback = callback

    try:
        return optimize.minimize(
            fun=fun,
            x0=x0,
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=wrapped_callback,
            options=options,
        )
    except OptimizationTimeoutError as e:
        msg = f"Optimization timed out after {e.runtime} seconds."
        current_fun, *_ = fun(e.current_x, *args)

        return optimize.OptimizeResult(
            fun=current_fun,
            x=e.current_x,
            nit=callback_data["num_iterations"],
            success=False,  # same as when maxiter is reached
            status=1,  # same as when L-BFGS-B reaches maxiter
            message=msg,
        )

    
     
                
def progressbar(it, prefix="", size=40, out=sys.stdout): # Python3.6+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {int(j/count*100)}%/{100}%", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)
    
    
    
def minimize_with_restarts(
    fun, 
    bounds, 
    x0 = None,
    num_restarts=100, 
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    constraints=None, 
    tol=None,
    callback=None,
    options=None,
    timeout_sec=None,
    optmization_stopping_criteria=None,
    ):

    
    if num_restarts is None:
        num_restarts=1       
    if num_restarts<1:
        num_restarts=1
    
    start_time = time.monotonic()
        
    bounds = np.array(bounds)
    xdim = len(bounds)
    
    candidates = {'success':{'x':[],'fun':[]},
                  'fail':{'x':[],'fun':[]}}
    for i_restart in range(num_restarts):
        if len(candidates['success']['fun'])>2 and optmization_stopping_criteria is not None:
            if optmization_stopping_criteria():
                break
            
        if timeout_sec is not None:
            timeout_sec_ = timeout_sec - (time.monotonic() - start_time)
        else:
            timeout_sec_ = None
        
        if x0 is not None and i_restart==0:
            x0 = x0
        else:
            x0 = np.random.rand(xdim)*(bounds[:,1]-bounds[:,0])+bounds[:,0]
        result = minimize_with_timeout(
                    fun,
                    x0,
                    method,
                    jac,
                    hess,
                    hessp,
                    bounds,
                    constraints, 
                    tol,
                    callback,
                    options,
                    timeout_sec_,)
        
        if result.success:
            assert type(result.x) == np.ndarray
            assert type(result.fun) == float
            candidates['success']['x'].append(result.x)
            candidates['success']['fun'].append(result.fun)
        else:
            if type(result.x) == np.ndarray and type(result.fun) == float:
                candidates['fail']['x'].append(result.x)
                candidates['fail']['fun'].append(result.fun)               
    
    success = True
    if len(candidates['success']['fun'])>=1:
        i = np.argmin(candidates['success']['fun'])
        x = candidates['success']['x'][i]
        fun = candidates['success']['fun'][i]
    elif len(candidates['fail']['fun'])>=1:
        i = np.argmin(candidates['fail']['fun'])
        x = candidates['fail']['x'][i]
        fun = candidates['fail']['fun'][i]
        success = False
    else:
        success = False
        return optimize.OptimizeResult(
            time=time.monotonic()-start_time,
            success=success)
    
    return optimize.OptimizeResult(
            fun=fun,
            x=x,
            time=time.monotonic()-start_time,
            success=success)


def maximize_with_restarts(
    fun, 
    bounds, 
    x0 = None,
    num_restarts=100, 
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    constraints=None, 
    tol=None,
    callback=None,
    options=None,
    timeout_sec=None,
    optmization_stopping_criteria=None):
    
    
    def neg_fun(x):
        return -fun(x)
    
    if jac is not None:
        def neg_jac(x):
            return -jac(x)
    else:
        neg_jac = None
    
    return minimize_with_restarts(
        neg_fun, 
        bounds, 
        x0,
        num_restarts, 
        method, 
        neg_jac, 
        hess, 
        hessp, 
        constraints, 
        tol, 
        callback, 
        options,
        timeout_sec,
        optmization_stopping_criteria)



def plot_2D_projection(
                        func,
                        bounds,
                        dim_xaxis = 0,
                        dim_yaxis = 1,
                        grid_ponits_each_dim = 25, 
                        project_minimum = False,
                        project_maximum = False,
                        project_mean = False,
                        fixed_values_for_each_dim = None, 
                        overdrive = False,
                        fig = None,
                        ax = None,
                        colarbar = True,
                        dtype = np.float32 ):
        '''
        fixed_values_for_each_dim: dict of key: dimension, val: value to fix for that dimension
        '''
        
        n_fixed = 0
        if fixed_values_for_each_dim is not None:
            n_fixed = len(fixed_values_for_each_dim.keys())
            assert dim_xaxis not in fixed_values_for_each_dim.keys()
            assert dim_yaxis not in fixed_values_for_each_dim.keys()
        else:
            fixed_values_for_each_dim = {}
            
        dim = len(bounds)    
        if dim > 2+n_fixed:
            assert project_minimum + project_maximum + project_mean == 1
        
        batch_size = 1
        for n in range(dim-2-n_fixed):
            batch_size*=grid_ponits_each_dim
            if batch_size*dim > 2e4:
                if overdrive or not (project_minimum or project_mean):
                    batch_size = int(batch_size/grid_ponits_each_dim)
                    print("starting projection plot...")
                    break
                else:
                    raise RuntimeError("Aborting: due to high-dimensionality and large number of grid point, minimum or mean projection plot may take long time. Try to reduce 'grid_ponits_each_dim' or turn on 'overdrive' if long time waiting is OK'")
        n_batch = int(grid_ponits_each_dim**(dim-n_fixed-2)/batch_size)
        linegrid = np.linspace(0,1,grid_ponits_each_dim)
        x_grid = np.zeros((grid_ponits_each_dim*grid_ponits_each_dim,dim))
        y_grid = np.zeros((grid_ponits_each_dim*grid_ponits_each_dim))
        
        n = 0
        for i in progressbar(range(grid_ponits_each_dim)):
            bounds_xaxis = bounds[dim_xaxis,:]
            for j in range(grid_ponits_each_dim):
                bounds_yaxis = bounds[dim_yaxis,:]
                x_grid[n,dim_xaxis] = linegrid[i]*(bounds_xaxis[1]-bounds_xaxis[0])+bounds_xaxis[0]
                x_grid[n,dim_yaxis] = linegrid[j]*(bounds_yaxis[1]-bounds_yaxis[0])+bounds_yaxis[0]
                if (project_minimum or project_maximum or project_mean) and dim > 2+n_fixed:
                    inner_grid = []
                    for d in range(dim):
                        if d == dim_xaxis:
                            inner_grid.append([x_grid[n,dim_xaxis]])
                        elif d == dim_yaxis:
                            inner_grid.append([x_grid[n,dim_yaxis]])
                        elif d in fixed_values_for_each_dim.keys():
                            inner_grid.append([fixed_values_for_each_dim[d]])
                        else:
                            inner_grid.append(np.linspace(bounds[d,0],
                                                          bounds[d,1],
                                                          grid_ponits_each_dim))
                    inner_grid = np.meshgrid(*inner_grid)
                    inner_grid = np.array(list(list(x.flat) for x in inner_grid),dtype=dtype).T
                    
                    y = []
                    for b in range(n_batch):
                        i1 = b*batch_size
                        i2 = i1 + batch_size
                        x_batch = inner_grid[i1:i2]
                        y.append(func(x_batch))
                    y = np.concatenate(y, axis=0)

                    if project_minimum or project_maximum:
                        if project_minimum:
                            arg_project = np.nanargmin
                        elif project_maximum:
                            arg_project = np.nanargmax
                        iproj = arg_project(y)
                        y_grid[n] = y[iproj]
                        x_grid[n,:] = inner_grid[iproj]
                    elif project_mean:
                        y_grid[n] = np.nanmean(y_mean)
                n+=1
                
        if dim==2+n_fixed:
            if fixed_values_for_each_dim is not None:
                for dim,val in fixed_values_for_each_dim.items():
                    x_grid[:,dim]  = val   
            y_grid  = func(x_grid)
            
        iNaN = np.any([np.isnan(x_grid).any(axis=1),np.isnan(y_grid)],axis=0)
        x_grid = x_grid[~iNaN,:]
        y_grid = y_grid[~iNaN]

                           
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.5,3))
        cs = ax.tricontourf(x_grid[:,dim_xaxis], x_grid[:,dim_yaxis], y_grid, levels=64, cmap="viridis");
        fig.colorbar(cs,ax=ax,shrink=0.95)
        

        
        
            
# def _init_population_lhs(nsample, ndim):
#     """
#     Initializes the population with Latin Hypercube Sampling.
#     Latin Hypercube Sampling ensures that each parameter is uniformly
#     sampled over its range.
#     rng : random generator
#     """

#     segsize = 1.0 / nsample
#     samples = (segsize * np.random.uniform(size=(nsample,ndim))
#     # Offset each segment to cover the entire parameter range [0, 1)
#                + np.linspace(0., 1., nsample,
#                              endpoint=False)[:, np.newaxis])
#     population = np.zeros_like(samples)

#     # Initialize population of candidate solutions by permutation of the
#     # random samples.
#     for j in range(ndim):
#         order = np.random.permutation(range(nsample))
#         population[:, j] = samples[order, j]

#     return population


def _init_population_qmc(n, d, qmc_engine='sobol',seed=None):
    """Initializes the population with a QMC method.
    QMC methods ensures that each parameter is uniformly
    sampled over its range.
    Parameters
    ----------
    qmc_engine : str
        The QMC method to use for initialization. Can be one of
        ``latinhypercube``, ``sobol`` or ``halton``.
    """
    from scipy.stats import qmc

    # Create an array for population of candidate solutions.
    if qmc_engine == 'latinhypercube':
        sampler = qmc.LatinHypercube(d=d, seed=seed)
    elif qmc_engine == 'sobol':        
        sampler = qmc.Sobol(d=d, seed=seed)
    elif qmc_engine == 'halton':
        sampler = qmc.Halton(d=d, seed=seed)
    else:
        raise ValueError("qmc_engine",qmc_engine,"is not recognized.")

    return sampler.random(n=n)


            
def proximal_ordered_init_sampler(n,
                                  bounds,
                                  x0,
                                  ramping_rate,
                                  polarity_change_time=10,
                                  method='sobol',seed=None):
    if n==0:
        return None
    bounds = np.array(bounds,dtype=np.float64)
    d = len(bounds)
    x0 = np.atleast_2d(x0)
    _,xd = x0.shape
    assert xd==d
    samples = list(_init_population_qmc(n,d,method,seed)*(bounds[:,1]-bounds[:,0]) + bounds[:,0])
    
    
    ordered_samples = []
    x0_ = x0
    while len(ordered_samples)<n:
        pop = np.array(samples)
        distance = polarity_change_time*np.any(np.sign(pop) != np.sign(x0_), axis=1) + np.max( np.abs(pop-x0_)/ramping_rate, axis=1)
        ordered_samples.append(samples.pop(np.argmin(distance)))
        x0_ = ordered_samples[-1]
        
    return np.array(ordered_samples)   