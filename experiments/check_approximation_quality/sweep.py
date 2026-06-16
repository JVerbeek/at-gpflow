import numpy as np 
import matplotlib.pyplot as plt 
import gpflow as gpf
import tensorflow as tf
import sys
import time
sys.path.append("/home/janneke/repos/at-gpflow/")
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from atmodel import ConditionalMOGP, SparseCMOGP
from atlikelihood import TransferLikelihood

def optimize(m):
    opt = gpf.optimizers.Scipy()
    res = opt.minimize(m.training_loss, m.trainable_variables, options={"disp": 50})

tries = 10

source_points = 400
props = np.arange(0.05, 2, 0.05)

vgp_mse = np.zeros((len(props), tries))
cmogp_mse = np.zeros((len(props), tries))
vgp_times = np.zeros((len(props), tries))
cmogp_times = np.zeros((len(props), tries))

for i, target_proportion in enumerate(props):
    for j in range(tries):
        k = gpf.kernels.RBF(lengthscales=3, variance=1) #+ gpf.kernels.Linear(0.001)
        NSP = source_points
        NTP = int(source_points * target_proportion)
        Xall = np.linspace(0, 50, NSP)
        Yall  = np.random.multivariate_normal(np.zeros_like(Xall), k(Xall.reshape(-1, 1)))
        ind_1 = np.arange(0, NSP, 1)
        ind_2 = np.linspace(0, 50, NTP).astype(int)
        TEST_INDEX = int(0.8*len(ind_2))

        ind_train, ind_test = ind_2[:TEST_INDEX], ind_2[TEST_INDEX:]
        X1, y1 = Xall[ind_1], Yall[ind_1] #+ np.random.multivariate_normal(np.zeros_like(Xall[ind_1]), gpf.kernels.Cosine(variance=1, lengthscales=10)(Xall[ind_1].reshape(-1, 1)))
        X2 = Xall[ind_train]
        y2 = Yall[ind_train]

        X2 = X2 - X2[0]
        scalar = -1
        y2 = (scalar * y2 + np.random.normal(0, 0.3, len(y2))).reshape(-1, 1)
        Xtest, ytest = Xall[ind_test], Yall[ind_test]
        ytest =  (ytest * scalar + np.random.normal(0, 0.3, len(ytest))).reshape(-1, 1)
        ytest = ytest
        Xtest = Xtest

        y1 = y1.reshape(-1, 1) + np.random.normal(0, 0.3, len(X1)).reshape(-1, 1)
        y2 = y2.reshape(-1, 1) 
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        X = np.vstack((np.hstack((X1, np.zeros_like(X1))), np.hstack((X2, np.ones_like(X2)))))
        y = np.vstack((np.hstack((y1, np.zeros_like(y1))), np.hstack((y2, np.ones_like(y2)))))
        X = tf.cast(X, np.float64)
        y = tf.cast(y, np.float64)
        
        ############ cMOGP ##############
        output_dim = 2  # Number of outputs
        rank = 1  # Rank of W

        # Base kernel
        k = gpf.kernels.Matern32(active_dims=[0])

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
        cmogp  = ConditionalMOGP((X, y), kernel=kern, likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))

        gpf.utilities.print_summary(cmogp)
        t = time.time()
        optimize(cmogp)
        dt = t - time.time()
        gpf.utilities.print_summary(cmogp)
        Xall = Xall.reshape(-1, 1)
        Xtst = Xtest.reshape(-1, 1)
        Xplot = np.hstack((Xall, np.ones_like(Xall)))
        Xtst = np.hstack((Xtst, np.ones_like(Xtst)))
        fmean_tst, fvar_tst = cmogp.predict_f(Xtst)
        cmogp_fmse = mean_squared_error(ytest, fmean_tst)

        cmogp_mse[i, j] = cmogp_fmse
        cmogp_times[i, j] = dt

        ############ SVGP ##############
        output_dim = 2  # Number of outputs
        rank = 1  # Rank of W
        # Base kernel
        k = gpf.kernels.Matern32(active_dims=[0])

        # Coregion kernel
        coreg = gpf.kernels.Coregion(
            output_dim=output_dim, rank=rank, active_dims=[1]
        )

        kern = k * coreg

        lik = gpf.likelihoods.SwitchedLikelihood(
            [gpf.likelihoods.Gaussian(), gpf.likelihoods.Gaussian()]
        )

        ivs = np.linspace(0, max(X[:,0]), 50).reshape(-1, 1)
        iv_ind = np.concatenate((np.ones((25,1 )), np.zeros((25, 1))))
        ivs = ivs[shuffle]
        ivs = np.hstack((ivs, iv_ind))
        X = np.vstack((np.hstack((X1, np.zeros_like(X1))), np.hstack((X2, np.ones_like(X2)))))
        y = np.vstack((np.hstack((y1, np.zeros_like(y1))), np.hstack((y2, np.ones_like(y2)))))
        # now build the GP model as normal
        svgp = gpf.models.SVGP(kernel=kern, likelihood=lik, num_data=len(X), inducing_variable=ivs)

        gpf.utilities.print_summary(svgp)
        # fit the covariance function parameters
        vt = time.time()
        gpf.optimizers.Scipy().minimize(
            svgp.training_loss_closure((X, y)),
            svgp.trainable_variables,
            method="L-BFGS-B",
        )
        vdt = vt - time.time()
        gpf.utilities.print_summary(svgp)
        Xall = Xall.reshape(-1, 1)
        Xtst = Xtest.reshape(-1, 1)
        Xplot = np.hstack((Xall, np.ones_like(Xall)))
        Xtst = np.hstack((Xtst, np.ones_like(Xtst)))
        fmean_tst, fvar_tst = svgp.predict_f(Xtst)
        vgp_fmse = mean_squared_error(ytest, fmean_tst)
        vgp_mse[i, j] = vgp_fmse
        vgp_times[i, j] = vdt

np.savez("/home/janneke/repos/at-gpflow/experiments/check_approximation_quality/results_sweep.npz", vgp_mse=vgp_mse, cmogp_mse=cmogp_mse)
res = np.load("/home/janneke/repos/at-gpflow/experiments/check_approximation_quality/results_sweep.npz")

print(res["cmogp_mse"])
print(res["vgp_mse"])

