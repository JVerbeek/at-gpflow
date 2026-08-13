import sys
sys.path.append("/home/janneke/src/at-gpflow")
from constrained_kernel import ConstrainedCoregion
from atmodel import ConditionalMOGP, SparseCMOGP
from atlikelihood import TransferLikelihood
from robust_svgp import LMCInducingPointsBase
import matplotlib.pyplot as plt 
import numpy as np 
import gpflow as gpf 
import tensorflow as tf

# Generate data with known correlations
k = gpf.kernels.RBF(lengthscales=3, variance=1) #+ gpf.kernels.Linear(0.001)
NDP = 200
Xall = np.linspace(0, 50, NDP)
Yall  = np.random.multivariate_normal(np.zeros_like(Xall), k(Xall.reshape(-1, 1)))
ind_1 = np.arange(0, NDP, 1)
ind_2 = np.linspace(0, NDP-1, int(0.1*NDP)).astype(int)
print(len(ind_2))
TEST_INDEX = int(0.8*len(ind_2))
ind_train, ind_test = ind_2[:TEST_INDEX], ind_2[TEST_INDEX:]
X1, y1 = Xall[ind_1], Yall[ind_1] #+ np.random.multivariate_normal(np.zeros_like(Xall[ind_1]), gpf.kernels.Cosine(variance=1, lengthscales=10)(Xall[ind_1].reshape(-1, 1)))
X2 = Xall[ind_train]
y2 = Yall[ind_train]

X2 = X2 - X2[0]
scalar = -0.2
y2 = (scalar * y2 + np.random.normal(0, 0.1, len(y2))).reshape(-1, 1)
Xtest, ytest = Xall[ind_test], Yall[ind_test]
ytest =  (ytest * scalar + np.random.normal(0, 0.1, len(ytest))).reshape(-1, 1)
ytest = ytest
Xtest = Xtest

y1 = y1.reshape(-1, 1) + np.random.normal(0, 0.1, len(X1)).reshape(-1, 1)
y2 = y2.reshape(-1, 1) 
X1 = X1.reshape(-1, 1)
X2 = X2.reshape(-1, 1)
plt.plot(X1, y1, marker=".")
plt.plot(X2, y2, marker=".", color="red")
plt.plot(Xtest, ytest, marker=".")
X = np.vstack((np.hstack((X1, np.zeros_like(X1))), np.hstack((X2, np.ones_like(X2)))))
y = np.vstack((np.hstack((y1, np.zeros_like(y1))), np.hstack((y2, np.ones_like(y2)))))
plt.show()
X = tf.cast(X, np.float64)
y = tf.cast(y, np.float64)

nIVS = 50
shuffle = np.random.permutation(np.arange(nIVS))
# Constrain correlation of CMOGP
## Constraining the correlation basically just means that the W matrix should have 1s on the diagonal.
## So I build that.
ivs = np.linspace(0, max(X[:,0]), nIVS).reshape(-1, 1)
iv_ind = np.concatenate((np.ones((int(len(ivs)/2),1 )), np.zeros((int(len(ivs)/2), 1))))
ivs = ivs[shuffle]
ivs = np.hstack((ivs, iv_ind))
k = gpf.kernels.RBF() * gpf.kernels.Coregion(output_dim=2, rank=1, active_dims=[1])
model = SparseCMOGP(data=(X, y), kernel= k, jitter=0.00001, inducing_variable=ivs, likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))
gpf.utilities.print_summary(model)
gpf.optimizers.Scipy().minimize(model.training_loss, model.trainable_variables)
gpf.utilities.print_summary(model)
## Run.

# Constrain correlation of VGP
## Uses same kernel, but different (approximate) objective.
## Run.
ivs = np.linspace(0, max(X[:,0]), nIVS).reshape(-1, 1)
iv_ind = np.concatenate((np.ones((int(len(ivs)/2),1 )), np.zeros((int(len(ivs)/2), 1))))
ivs = ivs[shuffle]
ivs = LMCInducingPointsBase(np.hstack((ivs, iv_ind)))
k = gpf.kernels.RBF() * gpf.kernels.Coregion(output_dim=2, rank=1, active_dims=[1])
m = gpf.models.SVGP(kernel=k, likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()), num_data=len(X), inducing_variable=ivs)

gpf.utilities.print_summary(m)
# fit the covariance function parameters
res = gpf.optimizers.Scipy().minimize(
    m.training_loss_closure((X, y)),
    m.trainable_variables,
    track_loss_history=True
)
gpf.utilities.print_summary(m)

# Record: W or B matrix
# Record: MSE

from sklearn.metrics import mean_squared_error
Xall = Xall.reshape(-1, 1)
Xtst = Xtest.reshape(-1, 1)
Xplot = np.hstack((Xall, np.ones_like(Xall)))
Xtst = np.hstack((Xtst, np.ones_like(Xtst)))
fmean, fvar = model.predict_f(Xplot)
fmean_tst, fvar_tst = model.predict_f(Xtst)
plt.plot(Xall, fmean, color="C0")
plt.plot(X1, y1, "kx")
plt.plot(X2, y2, "rx")
plt.plot(Xtest, ytest, "bx")
plt.fill_between(
    Xall[:, 0],
    (fmean - 2 * np.sqrt(fvar))[:, 0],
    (fmean + 2 * np.sqrt(fvar))[:, 0],
    color="C0",
    alpha=0.4,
)

cmogp_mse = mean_squared_error(ytest, fmean_tst)

fmean, fvar = m.predict_f(Xplot)
fmean_tst, fvar_tst = m.predict_f(Xtst)
plt.plot(Xall, fmean, color="C1")
plt.plot(X1, y1, "kx")
plt.plot(X2, y2, "rx")
plt.plot(Xtest, ytest, "bx")
plt.fill_between(
    Xall[:, 0],
    (fmean - 2 * np.sqrt(fvar))[:, 0],
    (fmean + 2 * np.sqrt(fvar))[:, 0],
    color="C1",
    alpha=0.4,
)
plt.show()

vgp_mse = mean_squared_error(ytest, fmean_tst)

print(cmogp_mse, vgp_mse)
W1 = model.kernel.kernels[1].W.numpy()
W2 = m.kernel.kernels[1].W.numpy()
B1 = W1 @ W1.T + np.diag(model.kernel.kernels[1].kappa)
B2 = W2 @ W2.T + np.diag(m.kernel.kernels[1].kappa)

print(B1, B2)

norm1 = np.diag(1/np.sqrt(np.diag(B1)))
print(norm1)
print(norm1 @ B1 @ norm1)

norm2 = np.diag(1/np.sqrt(np.diag(B2)))
print(norm2)
print(norm2 @ B2 @ norm2)

corr_coeff_1 = B1[0,1] / (np.sqrt(B1[0,0]) * np.sqrt(B1[1,1]))
print(f"Correlation coefficient B1: {corr_coeff_1}")

corr_coeff_2 = B2[0,1] / (np.sqrt(B2[0,0]) * np.sqrt(B2[1,1]))
print(f"Correlation coefficient B2: {corr_coeff_2}")
