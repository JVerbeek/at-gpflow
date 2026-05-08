import numpy as np  
import gpflow
import matplotlib.pyplot as plt 
import scipy.stats as ss
import tqdm as tqdm
import time

import sys
sys.path.append("..")

np.random.seed(124)
ndp = 200
repetitions = 5

cors = []
t1 = time.time()
for i in tqdm.tqdm(range(repetitions)):
    X1 = np.linspace(0, ndp, ndp)[:,None]  # Observed locations for first output
    X2_full = np.linspace(0, ndp, ndp)[:,None]  # Observed locations for second output

    Xall = np.linspace(0, ndp+10, ndp+10).reshape(-1, 1)
    k = gpflow.kernels.RBF(active_dims=[0], lengthscales=5, variance=5)
    Yall  = np.random.multivariate_normal(np.zeros_like(Xall[:,0]), k(Xall))
    Y1 = Yall[3:ndp+3][:,None] + np.random.normal(0, 0.1, size=len(X1))[:,None]
    Y2_full = Y1 * -1

    ind = np.sort(np.random.choice(np.arange(ndp), int(ndp/10)))
    X2 = X2_full[ind]
    Y2 = Y2_full[ind]


    # Augment the input with ones or zeros to indicate the required output dimension
    X_augmented = np.vstack(
        (np.hstack((X1, np.zeros_like(X1))), np.hstack((X2, np.ones_like(X2))))
    )

    # Augment the Y data with ones or zeros that specify a likelihood from the list of likelihoods
    Y_augmented = np.vstack(
        (np.hstack((Y1, np.zeros_like(Y1))), np.hstack((Y2, np.ones_like(Y2))))
    )


    output_dim = 2  # Number of outputs
    rank = 1  # Rank of W

    # Base kernel
    k = gpflow.kernels.RBF()

    # Coregion kernel
    coreg = gpflow.kernels.Coregion(
        output_dim=output_dim, rank=rank, active_dims=[1]
    )

    kern = k * coreg
    #kern2 = TransferKernel(0.5, gpflow.kernels.RBF(), source_length=int(len(Y_augmented) - np.sum(Y_augmented[:,1])))

    # This likelihood switches between Gaussian noise with different variances for each f_i:
    lik = gpflow.likelihoods.SwitchedLikelihood(likelihood_list=[gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian()])
    # now build the GP model as normal
    #lik = gpflow.likelihoods.Gaussian()

    m = gpflow.models.VGP((X_augmented, Y_augmented), kernel=kern, likelihood=lik)
    # fit the covariance function parameters
    res = gpflow.optimizers.Scipy().minimize(
        m.training_loss,
        m.trainable_variables,
        method="L-BFGS-B",
        track_loss_history=True
    )

    cov = m.kernel.kernels[1].output_covariance().numpy()
    cor = cov / cov.max()

    cors.append(cor)
print(time.time() - t1)
cors = np.array(cors)
print(cors.mean(axis=0), cors.std(axis=0))

