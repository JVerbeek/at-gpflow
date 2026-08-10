import gpflow as gpf
import tensorflow as tf
import numpy as np
import sys
sys.path.append("/home/janneke/src/at-gpflow")
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
import matplotlib.pyplot as plt
from atlikelihood import TransferLikelihood
from sklearn.metrics import mean_squared_error
from atmodel import ConditionalMOGP, SparseCMOGP
from atlikelihood import TransferLikelihood

def optimize(m):
    opt = gpf.optimizers.Scipy()
    res = opt.minimize(m.training_loss, m.trainable_variables, 
    
    track_loss_history=True, options={"disp": 50}, method="L-BFGS-B")

def get_kernel():
    return gpf.kernels.Matern32()

repetitions = 20
proportions = np.arange(0, 1, 0.1)

results_svgp = np.zeros((len(proportions), repetitions))
results_cmogp = np.zeros((len(proportions), repetitions))

for i, prop in enumerate(np.arange(0, 1, 0.1)):
    print("*"*10, "proportion", prop, "*"*10)
    for j in range(repetitions):
        print("REPETITION", j)
        Xs = np.linspace(0, 50, 500).reshape(-1, 1)
        Xt = np.linspace(0, 50, 500).reshape(-1, 1)
        f1 = np.random.multivariate_normal(np.zeros_like(Xs.flatten()), gpf.kernels.RBF(lengthscales=10)(Xs))
        f2 = np.random.multivariate_normal(np.zeros_like(Xt.flatten()), gpf.kernels.RBF(lengthscales=3)(Xt))

        ys = (prop * f1 + (1-prop) * f2 + np.random.normal(0, 0.1, len(Xs))).reshape(-1, 1)
        yt = ((1-prop)* f1 + prop * f2 + np.random.normal(0, 0.1, len(Xt))).reshape(-1, 1)
        yt = [y for i, y in enumerate(yt) if i % 10 == 0]
        Xt = [x for i, x in enumerate(Xt) if i % 10 == 0]
        yt_train = yt[:int(len(Xt)*0.75)]
        Xt_train = Xt[:int(len(Xt)*0.75)]
        yt_test = yt[int(len(Xt)*0.75):]
        Xt_test = Xt[int(len(Xt)*0.75):]

        X = np.vstack((np.hstack((Xs, np.zeros_like(Xs))), np.hstack((Xt, np.ones_like(Xt)))))
        y = np.vstack((np.hstack((ys, np.zeros_like(ys))), np.hstack((yt, np.ones_like(yt)))))

        output_dim = 2  # Number of outputs
        rank = 1  # Rank of W

        # Base kernel
        k = get_kernel() 

        # Coregion kernel
        coreg = gpf.kernels.Coregion(
            output_dim=output_dim, rank=rank, active_dims=[1]
        )


        ivs = np.linspace(0, max(X[:,0]), 50).reshape(-1, 1)
        iv_ind = np.concatenate((np.ones((25,1 )), np.zeros((25, 1))))
        shuffle = np.random.permutation(np.arange(len(ivs)))
        ivs = ivs[shuffle]
        ivs = np.hstack((ivs, iv_ind))
        kern = k * coreg 
        m = SparseCMOGP((X, y), jitter=1e-5, inducing_variable=ivs, kernel=kern, likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))
        #gpf.set_trainable(m.kernel.kernels[0].variance, True)
        #gpf.set_trainable(m.likelihood.source.variance, False)

        optimize(m)

        Xtst = np.hstack((Xt_test, np.ones_like(Xt_test)))
        fmean_tst, fvar_tst = m.predict_f(Xtst)

        cmogp_mse = mean_squared_error(yt_test, fmean_tst)
        results_cmogp[i, j] = cmogp_mse
        # This likelihood switches between Gaussian noise with different variances for each f_i:
        from robust_svgp import LMCInducingPointsBase
        output_dim = 2  # Number of outputs
        rank = 1  # Rank of W

        # Base kernel
        k = get_kernel()

        # Coregion kernel
        coreg = gpf.kernels.Coregion(
            output_dim=output_dim, rank=rank, active_dims=[1]
        )

        kern = k * coreg

        lik = TransferLikelihood(
            source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()
        )

        ivs = np.linspace(0, max(X[:,0]), 50).reshape(-1, 1)
        iv_ind = np.concatenate((np.ones((25,1 )), np.zeros((25, 1))))
        ivs = ivs[shuffle]
        ivs = np.hstack((ivs, iv_ind))
        # now build the GP model as normal

        m = gpf.models.SVGP(kernel=kern, inducing_variable=LMCInducingPointsBase(ivs), num_data=len(X), likelihood=lik)
        # fit the covariance function parameters
        gpf.optimizers.Scipy().minimize(
            m.training_loss_closure((X, y)), 
            m.trainable_variables,
            method="L-BFGS-B",
        )

        Xall = Xs.reshape(-1, 1)
        Xplot = np.hstack((Xall, np.ones_like(Xall)))
        Xtst = np.hstack((Xt_test, np.ones_like(Xt_test)))
        fmean, fvar = m.predict_f(Xplot)
        fmean_tst, fvar_tst = m.predict_f(Xtst)

        svgp_mse = mean_squared_error(yt_test, fmean_tst)
        results_svgp[i, j] = svgp_mse
        
        print("CMOGP:", cmogp_mse)
        print("SVGP:", svgp_mse)
        np.savez("results_at_home_matern32", svgp=results_svgp, cmogp=results_cmogp)