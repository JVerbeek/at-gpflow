"""Simple fit of the conditional MOGP on the snow data

Sources: three daily SNOTEL pillows (output indices 0, 1, 2). Target: Shrine Pass snow course (index 3).
Winters 2011-2020, November to May only, pillows thinned to every 2nd day. One winter of course
readings is held out to show the prediction between visits.
    python fit_conditional.py          # hold out water year 2019
    python fit_conditional.py 2016
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import gpflow as gpf
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

at_dir = Path(__file__).resolve().parents[2]  # repo root, where atmodel.py lives
sys.path.append(str(at_dir))
from atmodel import ConditionalMOGP, SparseCMOGP
from atlikelihood import TransferLikelihood
from robust_svgp import LMCInducingPointsBase

# ----------------------------------------------------------------------------- data
d = np.load(Path(__file__).parent / "data" / "nearby.npz")
HOLDOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
WINTERS = range(2011, 2021)


def month(t):
    return (np.floor((t - np.floor(t)) * 12).astype(int) % 12) + 1


def water_year(t):
    return np.where(month(t) >= 10, np.floor(t) + 1, np.floor(t)).astype(int)


def in_season(t):
    return np.isin(month(t), [11, 12, 1, 2, 3, 4, 5]) & np.isin(water_year(t), list(WINTERS))


data_X, data_Y = [], []
n_src = len(d["source_labels"])

for i in range(n_src):
    t, y = d[f"source{i}_t"], d[f"source{i}_y"]
    t, y = t[in_season(t)][::2], y[in_season(t)][::2]
    data_X.append(t.reshape(-1, 1))
    data_Y.append(y.reshape(-1, 1))
t, y = d["target_t"], d["target_y"]
t, y = t[in_season(t)], y[in_season(t)]
test = water_year(t) == HOLDOUT
Xtest, ytest = t[test].reshape(-1, 1), y[test].reshape(-1, 1)
data_X.append(t[~test].reshape(-1, 1))
data_Y.append(y[~test].reshape(-1, 1))

# standardize each output and build the stacked (value, index) arrays
means, stds = [Y.mean() for Y in data_Y], [Y.std() for Y in data_Y]
X = np.vstack([np.hstack((Xi, i * np.ones_like(Xi))) for i, Xi in enumerate(data_X)])
y = np.vstack([np.hstack(((Yi - m) / s, i * np.ones_like(Yi))) for i, (Yi, m, s) in enumerate(zip(data_Y, means, stds))])
print(f"{len(X)} training points, target index {n_src}, held-out winter {HOLDOUT} with {len(Xtest)} readings")

def optimize(m):
    opt = gpf.optimizers.Scipy()
    res = opt.minimize(m.training_loss, m.trainable_variables, track_loss_history=True, options={"disp": 50})

# ----------------------------------------------------------------------------- model: full conditional
# output_dim = n_src + 1  # Number of outputs
# rank = 1  # Rank of W
# condition_index = n_src  # the snow course

# # Base kernel
# k = gpf.kernels.Matern32(active_dims=[0], lengthscales=0.1)

# # Coregion kernel
# coreg = gpf.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])

# kern = k * coreg

# model1 = ConditionalMOGP((X, y), kernel=kern, conditioning_index=condition_index,
#                          likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))
# optimize(model1)
# gpf.utilities.print_summary(model1)

# ----------------------------------------------------------------------------- model: sparse conditional
output_dim = n_src + 1  # Number of outputs
rank = 1  # Rank of W
condition_index = n_src  # the snow course

# Base kernel
k = gpf.kernels.Matern32(active_dims=[0], lengthscales=0.1)

# Coregion kernel
coreg = gpf.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])

kern = k * coreg

nIVS = 25 * output_dim
ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
iv_ind = np.concatenate(iv_ind) 
shuffle = np.random.permutation(np.arange(len(ivs)))
ivs = ivs[shuffle]
ivs = np.hstack((ivs, iv_ind))

model2 = SparseCMOGP((X, y), kernel=kern, jitter=1e-5, conditioning_index=condition_index, inducing_variable=LMCInducingPointsBase(ivs),
                         likelihood=TransferLikelihood(source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()))
optimize(model2)
gpf.utilities.print_summary(model2)


# ----------------------------------------------------------------------------- model: sparse conditional
# Base kernel
k = gpf.kernels.Matern32(active_dims=[0])

# Coregion kernel
coreg = gpf.kernels.Coregion(
    output_dim=output_dim, rank=rank, active_dims=[1]
)

ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
iv_ind = np.concatenate(iv_ind)  # I guess the IPs for the target don't matter here, but this makes the comparison fair.
ivs = ivs[shuffle]
ivs = np.hstack((ivs, iv_ind))

l1 = gpf.likelihoods.Gaussian()
l2 = gpf.likelihoods.Gaussian()
lik = gpf.likelihoods.SwitchedLikelihood(
    [l1 if i != condition_index else l2 for i in range(output_dim)]
)
# now build the GP model as normal
model3 =  gpf.models.SVGP(kernel=kern, likelihood=lik, num_data=len(X), inducing_variable=LMCInducingPointsBase(ivs))


gpf.utilities.print_summary(model3)
# fit the covariance function parameters
gpf.optimizers.Scipy().minimize(
    model3.training_loss_closure((X, y)),
    model3.trainable_variables,
    method="L-BFGS-B",
)
gpf.utilities.print_summary(model3)
# ----------------------------------------------------------------------------- model: sgpr
# Base kernel
k = gpf.kernels.Matern32(active_dims=[0])

# Coregion kernel
coreg = gpf.kernels.Coregion(
    output_dim=output_dim, rank=rank, active_dims=[1]
)

ivs = np.linspace(np.min(X[:,0]), np.max(X[:,0]), nIVS).reshape(-1, 1)
iv_ind = [j * np.ones((int(nIVS/output_dim), 1)) for j in range(output_dim)]
iv_ind = np.concatenate(iv_ind)  # I guess the IPs for the target don't matter here, but this makes the comparison fair.
ivs = ivs[shuffle]
ivs = np.hstack((ivs, iv_ind))

l1 = gpf.likelihoods.Gaussian()
l2 = gpf.likelihoods.Gaussian()
lik = gpf.likelihoods.SwitchedLikelihood(
    [l1 if i != condition_index else l2 for i in range(output_dim)]
)
# now build the GP model as normal
model4 =  gpf.models.SGPR((X, y), kernel=kern, likelihood=l1, inducing_variable=LMCInducingPointsBase(ivs))


gpf.utilities.print_summary(model4)
# fit the covariance function parameters
gpf.optimizers.Scipy().minimize(
    model4.training_loss_closure((X, y)),
    model4.trainable_variables,
    method="L-BFGS-B",
)
gpf.utilities.print_summary(model4)
# ----------------------------------------------------------------------------- predictions
from sklearn.metrics import mean_squared_error
from matplotlib import color_sequences
colors = color_sequences["Set2"]
lo, hi = HOLDOUT - 1 + 10 / 12, HOLDOUT + 6 / 12  # plot the held-out winter only
Xtst = Xtest.reshape(-1, 1)

for name, model in zip(["sCMOGP", "SVGP", "SGPR"], [model2, model3, model4]):
    for index in range(output_dim):
        Ax, Ay = X[X[:, 1] == index], data_Y[index]
        Xplot = np.hstack((np.linspace(lo, hi, 300)[:, None], index * np.ones((300, 1))))
        fmean, fvar = model.predict_f(Xplot)
        fmean = fmean * stds[index] + means[index]  # back to inches
        fvar = fvar * stds[index] ** 2
        plt.figure(figsize=(12, 3))
        if index == condition_index:
            fmean_test, fvar_test = model.predict_f(np.hstack((Xtst, index * np.ones((len(Xtst), 1)))))
            fmean_test = fmean_test * stds[index] + means[index]
            plt.plot(Xtst, ytest, "r.", ms=12, label="target, held out")
        label = str(d["source_labels"][index]) if index < n_src else str(d["target_label"])
        plt.plot(Xplot[:, 0], fmean, color=colors[index], label=f"{label} predictions")
        m = (Ax[:, 0] >= lo) & (Ax[:, 0] <= hi)
        plt.plot(Ax[m, 0], Ay[m, 0], color=colors[index], marker=".", alpha=0.5,
                lw=0 if index == condition_index else 1, label=label)
        plt.fill_between(
            Xplot[:, 0],
            (fmean - 2 * np.sqrt(fvar))[:, 0],
            (fmean + 2 * np.sqrt(fvar))[:, 0],
            color=colors[index],
            alpha=0.4,
        )
        plt.xlim(lo, hi); plt.ylabel("SWE [in]"); plt.legend(fontsize=8, loc="upper left")
        plt.savefig(Path(__file__).parent / "figures" / f"fit_conditional_{HOLDOUT}_output{index}_{name}.png", dpi=120)
    plt.show()

    ours_mse = mean_squared_error(ytest, fmean_test)
    print(f"{name} held-out readings:", ytest[:, 0].round(1))
    print(f"{name} predicted:        ", fmean_test.numpy()[:, 0].round(1))
    print(f"{name} rmse:", np.sqrt(ours_mse).round(2), "in")
