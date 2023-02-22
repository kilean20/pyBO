from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np
from scipy.linalg import cholesky, solve
from collections import OrderedDict
from scipy.optimize import minimize
from typing import Any
eps = 1e-6

class Model(ABC):
    def __init__(self, data_standarize=True):
        r"""Abstract base class for BoNumpy models.
        """  
        self.x_mean = 0
        self.x_std = 1
        self.y_mean = 0
        self.y_std = 1
        self.data_standarize = data_standarize

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> None:
        pass  
    
    @abstractmethod
    def __call__(self, 
                 Xstar: np.ndarray,
                 return_var=True,
                )-> np.ndarray:
        pass  
    
    
    
class GaussianProcess(Model):
    def __init__(self, covfunc, optimize=False, usegrads=False, data_standarize=True, prior_mean_model=None):
        super().__init__(data_standarize)
        """
        Gaussian Process regressor class. Based on Rasmussen & Williams [1]_ algorithm 2.1.

        Parameters
        ----------
        covfunc: instance from a class of covfunc module
            Covariance function. An instance from a class in the `covfunc` module.
        optimize: bool:
            Whether to perform covariance function hyperparameter optimization.
        usegrads: bool
            Whether to use gradient information on hyperparameter optimization. Only used
            if `optimize=True`.

        Attributes
        ----------
        covfunc: object
            Internal covariance function.
        optimize: bool
            User chosen optimization configuration.
        usegrads: bool
            Gradient behavior
        mprior: float
            Explicit value for the mean function of the prior Gaussian Process.

        Notes
        -----
        [1] Rasmussen, C. E., & Williams, C. K. I. (2004). Gaussian processes for machine learning.
        International journal of neural systems (Vol. 14). http://doi.org/10.1142/S0129065704001899
        """
        self.covfunc = covfunc
        self.optimize = optimize
        self.usegrads = usegrads
        self.prior_mean_model = prior_mean_model
        self.name = 'GaussianProcess'


    def getcovparams(self):
        """
        Returns current covariance function hyperparameters

        Returns
        -------
        dict
            Dictionary containing covariance function hyperparameters
        """
        d = {}
        for param in self.covfunc.parameters:
            d[param] = self.covfunc.__dict__[param]
        return d

    def fit(self, x, y):
        """
        Fits a Gaussian Process regressor

        Parameters
        ----------
        x: np.ndarray, shape=(nsamples, nfeatures)
            Training instances to fit the GP.
        y: np.ndarray, shape=(nsamples,)
            Corresponding continuous target values to x.

        """
        y = np.array(y).reshape(-1,1)
        if self.prior_mean_model is not None:
            y = y-self.prior_mean_model(x).reshape(-1,1)
        if self.data_standarize:
            self.x_mean = np.mean(x)
            self.x_std = np.std(x)
            self.x = (x - self.x_mean)/self.x_std
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
            self.y = (y - self.y_mean)/self.y_std
        else:
            self.x = x
            self.y = y
        self.nsamples = self.x.shape[0]
        if self.optimize:
            grads = None
            if self.usegrads:
                grads = self._grad
            self.optHyp(param_key=self.covfunc.parameters, param_bounds=self.covfunc.bounds, grads=grads)

        self.K = self.covfunc.K(self.x, self.x)
        try:
            self.L = cholesky(self.K).T
        except:
            k = np.eye(len(self.K))
            dk = 2e-6 # 0.01*np.sum(k*self.K)/len(self.K)
            while(True):
                print('!!!! not invertible adding constant diangonal noise.....')
                try:
                    self.K += k*dk
                    self.L = cholesky(self.K).T
                    break
                except:
                    pass
        self.alpha = solve(self.L.T, solve(self.L, self.y.flatten()))
        self.logp = -.5 * np.dot(self.y.flatten(), self.alpha) - np.sum(np.log(np.diag(self.L))) - self.nsamples / 2 * np.log(
            2 * np.pi)

    def param_grad(self, k_param):
        """
        Returns gradient over hyperparameters. It is recommended to use `self._grad` instead.

        Parameters
        ----------
        k_param: dict
            Dictionary with keys being hyperparameters and values their queried values.

        Returns
        -------
        np.ndarray
            Gradient corresponding to each hyperparameters. Order given by `k_param.keys()`
        """
        k_param_key = list(k_param.keys())
        covfunc = self.covfunc.__class__(**k_param, bounds=self.covfunc.bounds)
        K = covfunc.K(self.x, self.x)
        L = cholesky(K).T
        alpha = solve(L.T, solve(L, self.y))
        inner = np.dot(np.atleast_2d(alpha).T, np.atleast_2d(alpha)) - np.linalg.inv(K)
        grads = []
        for param in k_param_key:
            gradK = covfunc.gradK(self.x, self.x, param=param)
            gradK = .5 * np.trace(np.dot(inner, gradK))
            grads.append(gradK)
        return np.array(grads)

    def _lmlik(self, param_vector, param_key):
        """
        Returns marginal negative log-likelihood for given covariance hyperparameters.

        Parameters
        ----------
        param_vector: list
            List of values corresponding to hyperparameters to query.
        param_key: list
            List of hyperparameter strings corresponding to `param_vector`.

        Returns
        -------
        float
            Negative log-marginal likelihood for chosen hyperparameters.

        """
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        self.covfunc = self.covfunc.__class__(**k_param, bounds=self.covfunc.bounds)

        # This fixes recursion
        original_opt = self.optimize
        original_grad = self.usegrads
        self.optimize = False
        self.usegrads = False

        self.fit(self.x, self.y)

        self.optimize = original_opt
        self.usegrads = original_grad
        return (- self.logp)

    def _grad(self, param_vector, param_key):
        """
        Returns gradient for each hyperparameter, evaluated at a given point.

        Parameters
        ----------
        param_vector: list
            List of values corresponding to hyperparameters to query.
        param_key: list
            List of hyperparameter strings corresponding to `param_vector`.

        Returns
        -------
        np.ndarray
            Gradient for each evaluated hyperparameter.

        """
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        return - self.param_grad(k_param)

    def optHyp(self, param_key, param_bounds, grads=None, n_trials=20):
        """
        Optimizes the negative marginal log-likelihood for given hyperparameters and bounds.
        This is an empirical Bayes approach (or Type II maximum-likelihood).

        Parameters
        ----------
        param_key: list
            List of hyperparameters to optimize.
        param_bounds: list
            List containing tuples defining bounds for each hyperparameter to optimize over.

        """
        xs = [[1, 1, 1]]
        fs = [self._lmlik(xs[0], param_key)]
        for trial in range(n_trials):
            x0 = []
            for param, bound in zip(param_key, param_bounds):
                x0.append(np.random.uniform(bound[0], bound[1], 1)[0])
            if grads is None:
                res = minimize(self._lmlik, x0=x0, args=(param_key), method='L-BFGS-B', bounds=param_bounds)
            else:
                res = minimize(self._lmlik, x0=x0, args=(param_key), method='L-BFGS-B', bounds=param_bounds, jac=grads)
            xs.append(res.x)
            fs.append(res.fun)

        argmin = np.argmin(fs)
        opt_param = xs[argmin]
        k_param = OrderedDict()
        for k, x in zip(param_key, opt_param):
            k_param[k] = x
        self.covfunc = self.covfunc.__class__(**k_param, bounds=self.covfunc.bounds)

    def __call__(self, Xstar, return_var=True):
        """
        Returns mean and covariances for the posterior Gaussian Process.

        Parameters
        ----------
        Xstar: np.ndarray, shape=((nsamples, nfeatures))
            Testing instances to predict.
        return_std: bool
            Whether to return the standard deviation of the posterior process. Otherwise,
            it returns the whole covariance matrix of the posterior process.

        Returns
        -------
        np.ndarray
            Mean of the posterior process for testing instances.
        np.ndarray
            Covariance of the posterior process for testing instances.
        """
        Xstar = np.atleast_2d(Xstar)
        if self.prior_mean_model is not None:
            y_prior = self.prior_mean_model(Xstar).flatten()
        if self.data_standarize:
            Xstar = (Xstar-self.x_mean)/self.x_std
        kstar = self.covfunc.K(self.x, Xstar).T
        y = np.dot(kstar, self.alpha)
        if self.data_standarize:
            y = y*self.y_std + self.y_mean
        if self.prior_mean_model is not None:
            y += y_prior
        if return_var:
            v = solve(self.L, kstar.T)
            fcov = self.covfunc.K(Xstar, Xstar) - np.dot(v.T, v)
            fcov = np.diag(fcov)
            if self.data_standarize:
                fcov = fcov*self.y_std**2
            return y, np.clip(fcov, a_min=eps, a_max=None)
        else:
            return y
    