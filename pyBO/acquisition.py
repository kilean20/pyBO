from typing import Callable, Optional
from scipy.stats import norm, t
import numpy as np
import warnings
from typing import Optional, Union
eps = 1e-6


def penalize_X(
    X: np.ndarray,
    X_penal: Optional[np.ndarray] = None,
    L_penal: Optional[Union[np.ndarray, float]] = 0.1,
    C_penal: Optional[float] = 0.2,
    ) -> np.ndarray:
    
    X = np.atleast_2d(X)
    b,d = X.shape
    f = np.zeros(b)
    if X_penal is not None:
        X_penal = np.atleast_2d(X_penal)
        _,d = X_penal.shape
        for i in range(b):
            f[i] = - np.mean(C_penal*np.exp(-np.mean((X[i:i+1,-d:]-X_penal)**2/L_penal**2, axis=1)))
    return f


def favor_X(         
    X: np.ndarray,
    X_favor: Optional[np.ndarray] = None,
    L_favor: Optional[Union[np.ndarray, float]] = 0.1,
    C_favor: Optional[float] = 0.2,
    ) -> np.ndarray:
    
    X = np.atleast_2d(X)
    b,d = X.shape
    f = np.zeros(b)
    if X_favor is not None:
        if f is None:
            f = np.zeros(b)
        X_favor = np.atleast_2d(X_favor)
        _,d = X_favor.shape
        for i in range(b):
            f[i] =   np.mean(C_favor*np.exp(-np.mean((X[i:i+1,-d:]-X_favor)**2/L_favor**2, axis=1)))
    return f
    

def penalize_polarity_change(
    X: np.ndarray,
    X_current: np.ndarray,
    polarity_penalty = 0.,
    ) -> np.ndarray:
    
    X = np.atleast_2d(X)
    if polarity_penalty == 0 or polarity_penalty is None or X_current is None :
        return 0.
    else:
        X_current = np.atleast_2d(X_current)
        _,d = X_current.shape
        return polarity_penalty*np.any(np.logical_and(np.sign(X) != np.sign(X_current),X_current!=0), axis=1)
    

class AcquisitionFunction:
    r"""Abstract base class for acquisition functions.
    """
    def __init__(self,model=None):
        r"""Constructor for the AcquisitionFunction base class.
        Args:
            model: A fitted surrogate model.
        """
#         super().__init__()
        self.model = model
        # if model is None:
        #     warnings.warn('AcquisitionFunction is initialized without model. Must be specified before call.')    
        self.name = ''
        
    def penal_or_favor(self,
        X: np.ndarray, 
        X_penal: Optional[np.ndarray] = None,
        L_penal: Optional[Union[np.ndarray, float]] = 0.1,
        C_penal: Optional[float] = 0.2,
        X_favor: Optional[np.ndarray] = None,
        L_favor: Optional[Union[np.ndarray, float]] = 0.1,
        C_favor: Optional[float] = 0.2,
        X_current: Optional[np.ndarray] = None,
        polarity_penalty: Optional[float] = 1.0,
        **kwargs
        ) -> np.ndarray:
        f = 0.
        if X_penal is not None:
            f+= penalize_X(X,X_penal,L_penal,C_penal)
        if X_favor is not None:
            f+= favor_X(X,X_favor,L_favor,C_favor)
        if X_current is not None:
            f-= penalize_polarity_change(X,X_current,polarity_penalty)
        return f
        
class ExpectedImprovement(AcquisitionFunction):
    r"""Logarithm of single-outcome Expected Improvement (analytic).
    """
    def __init__(
        self,
        model = None,
#         best_y = None,
    ):
        r"""Logarithm of single-outcome Expected Improvement (analytic).
        Args:
            model: A fitted surrogate model
            best_y: a scalar representing the best function value observed so far (assumed noiseless).
        """
        super().__init__(model=model)
        self.name = 'ExpectedImprovement'


    def __call__(self, 
                    X: np.ndarray, 
                    best_y: float,
                    X_penal: Optional[np.ndarray] = None,
                    L_penal: Optional[Union[np.ndarray, float]] = 0.1,
                    C_penal: Optional[float] = 0.2,
                    X_favor: Optional[np.ndarray] = None,
                    L_favor: Optional[Union[np.ndarray, float]] = 0.1,
                    C_favor: Optional[float] = 0.2,
                    X_current: Optional[np.ndarray] = None,
                    polarity_penalty: Optional[float] = 1.0,
                    **kwargs) -> np.ndarray:
#         self.best_y = best_y or self.best_y
#         if self.best_y is None:
#             raise TypeError("best_y could not be inferred. ExpectedImprovement requires best_y")
        mean, var = self.model(X,return_var=True)
        with np.errstate(divide='warn'):
            z = np.clip( (mean - best_y) / (var**0.5), -6., 6.).astype(np.float64)
        if np.any(np.isnan(z)):
            print("mean,var,best_y",mean,var,best_y)
            import time
            time.sleep(2)
            raise RuntimeError("NaN!!",z)
        f = (mean - best_y) * norm.cdf(z) + var**0.5 * norm.pdf(z)
                 
        return f + self.penal_or_favor(X,X_penal,L_penal,C_penal,X_favor,L_favor,C_favor,X_current,polarity_penalty)
    
    
class UpperConfidenceBound(AcquisitionFunction):
    r"""Logarithm of single-outcome Expected Improvement (analytic).
    """
    def __init__(
        self,
        model = None,
        beta = None,
    ):
        r"""Logarithm of single-outcome Expected Improvement (analytic).
        Args:
            model: A fitted surrogate model
        """
        super().__init__(model=model)
        self.name = 'UpperConfidenceBound'
        self.beta = beta


    def __call__(self, 
                    X: np.ndarray, 
                    beta: Optional[float] = None,
                    X_penal: Optional[np.ndarray] = None,
                    L_penal: Optional[Union[np.ndarray, float]] = 0.1,
                    C_penal: Optional[float] = 0.2,
                    X_favor: Optional[np.ndarray] = None,
                    L_favor: Optional[Union[np.ndarray, float]] = 0.1,
                    C_favor: Optional[float] = 0.2,
                    X_current: Optional[np.ndarray] = None,
                    polarity_penalty: Optional[float] = 1.0,
                    **kwargs ) -> np.ndarray:
#         self.beta = beta or self.beta
#         if beta is None:
#             raise TypeError("beta could not be inferred. UpperConfidenceBound requires beta")
        beta = beta or self.beta
        if beta is None:
            beta = 0
        mean, var = self.model(X,return_var=True)
        f = mean + (beta*var)**0.5
        
        return f+ self.penal_or_favor(X,X_penal,L_penal,C_penal,X_favor,L_favor,C_favor,X_current,polarity_penalty)

    

    

      
            
        
    
    