import gpflow as gpf
import tensorflow as tf
import numpy as np
tf.random.set_seed(42)
np.random.seed(42)
import sys
from pathlib import Path
sys.path.append(str(Path.home() / "src" / "at-gpflow"))
print(sys.path)
from sklearn.metrics import mean_squared_error
from robust_svgp import LMCInducingPointsBase
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
import matplotlib.pyplot as plt

from atmodel import ConditionalMOGP, SparseCMOGP, SparseCMOGP_QR
from atlikelihood import TransferLikelihood

def optimize(m):
    opt = gpf.optimizers.Scipy()
    res = opt.minimize(m.training_loss, m.trainable_variables, track_loss_history=True, options={"disp": 50}, method="L-BFGS-B")
    # plt.plot(res["loss_history"])
    # plt.show()

def get_kernel():
    return gpf.kernels.Matern32(active_dims=[0])



scmogp_mse = np.zeros((10, 2))  # full, sparse
svgp_mse = np.zeros((10, 2))  # full, sparse

for i in range(10):
    print("*"*20, i, "*"*20)
    Xs = np.linspace(0, 20, 200).reshape(-1, 1)
    Xt = np.linspace(0, 20, 200).reshape(-1, 1)
    f1 = np.random.multivariate_normal(np.zeros_like(Xs.flatten()), gpf.kernels.Matern32(lengthscales=5, variance=1)(Xs))
    f2 = np.random.multivariate_normal(np.zeros_like(Xs.flatten()), gpf.kernels.Matern32(lengthscales=1, variance=1)(Xs))

    test_size = int(int(len(Xt)) * 0.1)
    start = int(int(len(Xt)) * 0.45)
    ys = (f1 + np.random.normal(0, 0.05, len(Xs))).reshape(-1, 1)
    yt = (f1 + f2 + np.random.normal(0, 0.05, len(Xt))).reshape(-1, 1)
    yt_train_full = np.concatenate((yt[:start], yt[start+test_size:]))
    Xt_train_full = np.concatenate((Xt[:start], Xt[start+test_size:]))
    yt_test_full = yt[start:start + test_size]
    Xt_test_full = Xt[start:start + test_size]

    X_full = np.vstack((np.hstack((Xs, np.zeros_like(Xs))), np.hstack((Xt_train_full, np.ones_like(Xt_train_full)))))
    y_full = np.vstack((np.hstack((ys, np.zeros_like(ys))), np.hstack((yt_train_full, np.ones_like(yt_train_full)))))

    yt_ds = [y for i, y in enumerate(yt) if i % 4 == 0]
    Xt_ds = [x for i, x in enumerate(Xt) if i % 4 == 0]

    test_size = int(int(len(Xt_ds)) * 0.1)
    start = int(int(len(Xt_ds)) * 0.45)

    yt_train_ds = np.concatenate((yt_ds[:start], yt_ds[start+test_size:]))
    Xt_train_ds = np.concatenate((Xt_ds[:start], Xt_ds[start+test_size:]))
    yt_test_ds= yt_ds[start:start + test_size]
    Xt_test_ds = Xt_ds[start:start + test_size]

    X_ds = np.vstack((np.hstack((Xs, np.zeros_like(Xs))), np.hstack((Xt_train_ds, np.ones_like(Xt_train_ds)))))
    y_ds = np.vstack((np.hstack((ys, np.zeros_like(ys))), np.hstack((yt_train_ds, np.ones_like(yt_train_ds)))))


    fig, (ax1, ax2) = plt.subplots(1, 2)
    for j, (X, y, Xtest, ytest) in enumerate([(X_full, y_full, Xt_test_full, yt_test_full), (X_ds, y_ds, Xt_test_ds, yt_test_ds)]):
        output_dim = 2  # Number of outputs
        rank = 1  # Rank of W
        target_index = 1

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

        nIVS = 100 * output_dim
        ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
        iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
        iv_ind = np.concatenate(iv_ind) 
        shuffle = np.random.permutation(np.arange(len(ivs)))
        ivs = ivs[shuffle]
        ivs = np.hstack((ivs, iv_ind))

        l1 = gpf.likelihoods.Gaussian()
        l2 = gpf.likelihoods.Gaussian()
        lik = gpf.likelihoods.SwitchedLikelihood(
            [l1 if i != target_index else l2 for i in range(output_dim)]
        )
        # now build the GP model as normal
        model2 =  gpf.models.SVGP(kernel=kern, likelihood=lik, num_data=len(X), inducing_variable=LMCInducingPointsBase(ivs))

        # fit the covariance function parameters
        gpf.optimizers.Scipy().minimize(
            model2.training_loss_closure((X, y)),
            model2.trainable_variables,
            method="L-BFGS-B",
        )

        # base kernel
        k = get_kernel() 

        # coregion kernel
        coreg = gpf.kernels.Coregion(
            output_dim=output_dim, rank=rank, active_dims=[1]
        )

        kern = k * coreg 

        ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
        iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
        iv_ind = np.concatenate(iv_ind)  
        ivs = ivs[shuffle]
        ivs = np.hstack((ivs, iv_ind))
        
        model1 = SparseCMOGP((X, y), conditioning_index=target_index, exact_target=False, kernel=kern, jitter=1e-5, inducing_variable=LMCInducingPointsBase(ivs), likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))

        optimize(model1) 
        
        fmean_test, fvar_test = model1.predict_f(np.hstack((Xtest, np.ones_like(Xtest))))
        mse = mean_squared_error(ytest, fmean_test[:,0])
        print(model1, mse)
        scmogp_mse[i, j] = mse
        fmean_test, fvar_test = model2.predict_f(np.hstack((Xtest, np.ones_like(Xtest))))
        mse = mean_squared_error(ytest, fmean_test[:,0])
        print(model2, mse)
        svgp_mse[i, j] = mse

np.savez("toy-example-interpolation", svgp=svgp_mse, scmogp=scmogp_mse)