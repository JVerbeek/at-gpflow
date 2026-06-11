import numpy as np
import gpflow as gpf 
import matplotlib.pyplot as plt
import time 
import sys
sys.path.append("/home/janneke/repos/at-gpflow/")
from atmodel import SparseCMOGP
from atlikelihood import TransferLikelihood
n_steps = 8
n_tries = 10

times = np.zeros((n_steps, n_tries)) 
for j, NDP in enumerate(np.logspace(6, 11, n_steps, base=2)):
    continue
    print(int(NDP))
    gen_k = gpf.kernels.RBF(lengthscales=5, variance=1) #+ gpf.kernels.Linear(0.001)
    Xall = np.linspace(0, 50, int(NDP))
    Yall  = np.random.multivariate_normal(np.zeros_like(Xall), gen_k(Xall.reshape(-1, 1)))
    ind_1 = np.arange(0, int(NDP), 1)
    ind_2 = np.linspace(0, 50, 50).astype(int)

    TEST_INDEX = int(0.8*len(ind_2))
    ind_train, ind_test = ind_2[:TEST_INDEX], ind_2[TEST_INDEX:]
    X1, y1 = Xall[ind_1], Yall[ind_1] #+ np.random.multivariate_normal(np.zeros_like(Xall[ind_1]), gpf.kernels.Cosine(variance=1, lengthscales=10)(Xall[ind_1].reshape(-1, 1)))
    X2 = Xall[ind_train]
    y2 = Yall[ind_train]

    X2 = X2 - X2[0]
    scalar = -1
    y2 = scalar * y2
    y2 = y2
    Xtest, ytest = Xall[ind_test], Yall[ind_test]
    ytest = scalar * ytest.reshape(-1, 1) + np.random.normal(0, 0.1, len(Xtest)).reshape(-1, 1)

    y1 = y1.reshape(-1, 1) + np.random.normal(0, 0.5, len(X1)).reshape(-1, 1)
    y2 = y2.reshape(-1, 1) + np.random.normal(0, 0.1, len(X2)).reshape(-1, 1) 
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)
    y2 = y2 

    X = np.vstack((np.hstack((X1, np.zeros_like(X1))), np.hstack((X2, np.ones_like(X2)))))
    y = np.vstack((np.hstack((y1, np.zeros_like(y1))), np.hstack((y2, np.ones_like(y2)))))

    for i in range(n_tries):

        output_dim = 2  # Number of outputs
        rank = 1  # Rank of W

        # Base kernel
        k = gpf.kernels.Matern32(active_dims=[0])

        # Coregion kernel
        coreg = gpf.kernels.Coregion(
            output_dim=output_dim, rank=rank, active_dims=[1]
        )
        kern = k * coreg
        nIVS = 50
        ivs = np.linspace(0, max(X[:,0]), nIVS).reshape(-1, 1)
        iv_ind = np.concatenate((np.ones((int(nIVS/2),1 )), np.zeros((int(nIVS/2), 1))))  # I guess the IPs for the target don't matter here, but this makes the comparison fair.
        shuffle = np.random.permutation(np.arange(len(ivs)))
        ivs = ivs[shuffle]
        ivs = np.hstack((ivs, iv_ind))
        t_opt = time.time()
        m = SparseCMOGP((X, y), exact_target=False, kernel=kern, jitter=1e-5, inducing_variable=ivs, likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))
        opt = gpf.optimizers.Scipy()
        opt.minimize(m.training_loss, m.trainable_variables, options={"disp": 50, "maxiter": 50})
        dt_opt = time.time() - t_opt
        times[j, i] = dt_opt

ticks = np.linspace(6, 11, 8).astype(int)
times = [ 1.14012425,  1.06276362,  1.12836003,  1.23768082, 1.56313865,  6.22101068, 12.519521,   26.18981016]
plt.plot(ticks, times)
plt.xlabel("n_points")
plt.xticks(ticks, [f"$2^{{ {t} }}$" for t in ticks])
plt.ylabel("$\Delta$t, 50 steps")
plt.show()