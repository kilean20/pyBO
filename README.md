# Overview

pyBO is a Python package for asynchronous and cost-aware Bayesian Optimization (BO), specifically designed to maximize beam-time utilization. It requires only numpy, scipy, and matplotlib, avoiding policy complications related to third-party package installations (like PyTorch) in secure control networks.

## Features

- **Asynchronous BO**: Efficiently manages machine evaluations and computation for optimal beam-time use.
- **Cost-Aware Optimization**: Minimizes costs related to control ramping.
- **Local Optimization**: Implements a moving window around the best-evaluated point for refined local searches. This is crucial to achieve good solutions within limited beam-time.
- **Automatic Hyperparameter Tuning**: Dynamically adjusts the "beta" parameter for Upper Confidence Bound (UCB) to enhance performance.
- and many other

## Example Usage

```python
from pyBO import pyBO

# Initialize the BO controller
ctrBO = pyBO.bo_controller(obj, local_optimization=False)

# Initial budget
ctrBO.init(n_init_budget)

# Global optimization
ctrBO.optimize_global(n_global_opt_budget, beta_scheduler='auto')

# Local optimization
ctrBO.optimize_local(n_local_opt_budget, beta_scheduler='auto')

# Fine-tuning
ctrBO.fine_tune(n_finetune_budget)

# Finalize the optimization
ctrBO.finalize()

# Close all plot callbacks
for f in ctrBO.plot_callbacks:
    f.close()

# Get the best results
x_best, y_best_old = ctrBO.bo.best_sofar()
y_best_new = obj(x_best)
