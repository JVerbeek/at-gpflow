from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from typing import Generator, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import mcmc
import pandas as pd
import gpflow
from gpflow.config import default_float
from gpflow.utilities import print_summary

import pandas as pd

gpflow.default_jitter()

__all__ = [
    "create_data", "create_trcd_model", "HMCParameters", "optimize_with_scipy_optimizer", "create_standard_mcmc",
    "create_nuts_mcmc", "handle_pool"
]

Data = Tuple[tf.Tensor, tf.Tensor]
Data_p = Tuple[tf.Tensor,tf.Tensor]
Initial_D = Tuple[tf.Tensor]
Initial_S = Tuple[tf.Tensor]
Initial_lengthscale = Tuple[tf.Tensor]
Initial_variance = Tuple[tf.Tensor]
FullData = Data
Observations = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


def dfloat(value):  # default float
    return tf.cast(value, default_float())


@contextmanager
def handle_pool(pool: Pool) -> Generator[Pool, None, None]:
    try:
        yield pool
    finally:
        pool.close()
        pool.join()

#################################################################################
# Functions for filtering genes
#################################################################################
def fit_rbf(data, init_lengthscale, init_variance):
    #alpha, beta = compute_prior_hyperparameters(10.0, 5.0)
    k = gpflow.kernels.RBF()
    m = gpflow.models.GPR(data, kernel=k)
    m.likelihood.variance.assign(0.01)
    m.kernel.lengthscales.assign(init_lengthscale)
    m.kernel.variance.assign(init_variance)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss,
                            m.trainable_variables,
                            options=dict(maxiter=100))
    return m, np.asarray(m.training_loss())

def fit_rbf2(data, init_lengthscale, init_variance):
    k = gpflow.kernels.RBF()
    m = gpflow.models.GPR(data, kernel=k,mean_function=gpflow.mean_functions.Constant(c=None))
    m.likelihood.variance.assign(0.01)
    m.kernel.lengthscales.assign(init_lengthscale)
    m.kernel.variance.assign(init_variance)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss,
                            m.trainable_variables,
                            options=dict(maxiter=100))
    return m, np.asarray(m.training_loss())

def fit_noise(data, init_variance):
    #k = gpflow.kernels.White()
    k = gpflow.kernels.RBF()
    m = gpflow.models.GPR(data, kernel=k, mean_function=gpflow.mean_functions.Constant(c=None))
    m.likelihood.variance.assign(0.01)
    m.kernel.variance.assign(init_variance)
    m.kernel.lengthscales.assign(1000000.0)
    gpflow.utilities.set_trainable(m.kernel.lengthscales, False)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss,
                            m.trainable_variables,
                            options=dict(maxiter=100))
    return m, np.asarray(m.training_loss())
