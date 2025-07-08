import gpflow as gpf
from atkernel import TransferKernel
import tensorflow as tf
import numpy as np
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
import matplotlib.pyplot as plt
from atlikelihood import TransferLikelihood
gpf.config.set_default_jitter(0.001)

class AdaptiveTransferGPR(gpf.models.GPModel, InternalDataTrainingLossMixin):
    def __init__(self, data_source, data_target, base_kernel):
        self.kernel = TransferKernel(1, 1, base_kernel)
        self.data_source = gpf.models.util.data_input_to_tensor(data_source)
        self.data_target = gpf.models.util.data_input_to_tensor(data_source)
        self.mean_function = gpf.mean_functions.Zero()
        super().__init__(
            kernel=self.kernel,
            likelihood=TransferLikelihood(
                source=gpf.likelihoods.Gaussian(variance=0.01), target=gpf.likelihoods.Gaussian(variance=0.01)
            ),
            mean_function=self.mean_function,
            num_latent_gps=1
        )
        
    def maximum_log_likelihood_objective(self):
        return self.adaptive_log_marginal_likelihood()
    
    def adaptive_log_marginal_likelihood(self):
        Sx, Sy = self.data_source
        Tx, Ty = self.data_target
        
        Kss = self.kernel.kernel(Sx, Sx) + self.likelihood.source.variance * tf.eye(tf.shape(Sx)[0], dtype=Sx.dtype)
        Kst = self.kernel.interdomain(Sx, Tx)
        Kts = tf.linalg.matrix_transpose(Kst)
        Ktt = self.kernel.kernel(Tx, Tx) + self.likelihood.target.variance * tf.eye(tf.shape(Tx)[0], dtype=Tx.dtype)
        
        Lss = tf.linalg.cholesky(Kss)
        
        A = tf.linalg.triangular_solve(Lss, Sy, lower=True)
        V = tf.linalg.triangular_solve(Lss, Kst, lower=True)
        
        mu_t = tf.matmul(V, A, transpose_a=True)
        C_t = Ktt - tf.matmul(Kts, V, transpose_a=True)
        
        L_t = tf.linalg.cholesky(C_t)
        
        logdet_t = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_t)))
        
        delta = Ty - mu_t
        alpha_t = tf.linalg.triangular_solve(L_t, delta, lower=True)
        
        n_target = tf.cast(tf.shape(Tx)[0], Sx.dtype)
        lml = -0.5 * (tf.reduce_sum(tf.square(alpha_t)) + 
                    n_target * tf.cast(tf.math.log(2*np.pi), Sx.dtype) + 
                    2 * logdet_t)
        
        return tf.squeeze(lml)
        

    def predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False):
        r"""
        Allegedly, the GP prediction stays the same, so instead of creating an inference shaped footgun, use GPFlow methods.
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data
        points.
        """
        Sx, Sy = self.data_source
        Tx, Ty = self.data_target
        X = tf.concat((Sx, Tx), 0)
        y = tf.concat((Ty, Sy), 0)
        err = y - self.mean_function(X)
        
        # Construct Knn
        knn = self.kernel.kernel(Xnew, full_cov=full_cov)
        
        # Construct the kernel matrix Kmm
        Css = self.kernel.kernel(Sx, Sx)
        Cst = self.kernel.interdomain(Sx, Tx)
        Ctt = self.kernel.kernel(Tx, Tx)
        Kmm = tf.concat((tf.concat((Css, Cst), 0), tf.concat((tf.linalg.matrix_transpose(Cst), Ctt), 0)), 1)
        kmm_plus_s = Kmm + tf.concat((self.likelihood.source.variance_at(Sx), self.likelihood.target.variance_at(Tx)), 0)
        
        # Construct Kmn
        Ks = self.kernel.interdomain(Xnew, Sx)
        Kt = self.kernel.kernel(Xnew, Tx)
        kmn = tf.concat((Ks, Kt), 0)

        conditional = gpf.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var
        
        
Sx = np.linspace(0, 10, 100).reshape(-1, 1)
Sy = (np.sin(Sx * 2) + np.random.normal(0, 0.01, size=100).reshape(-1, 1)) / 0.01
Tx = np.linspace(0, 10, 10).reshape(-1, 1)
Ty = (np.sin(Tx * 2) + np.random.normal(0, 0.01, size=10).reshape(-1, 1)) / 0.01
plt.plot(Sx, Sy)
plt.plot(Tx, Ty)
plt.show()
at_gpr = AdaptiveTransferGPR((Sx, Sy), (Tx, Ty), gpf.kernels.RBF())

print("Training loss value:", at_gpr.training_loss().numpy())
at_gpr.training_loss()
opt = gpf.optimizers.Scipy()
opt.minimize(at_gpr.training_loss, at_gpr.trainable_variables)
gpf.utilities.print_summary(at_gpr)
at_gpr.kernel.kernel.lengthscales.assign(0.02)
Kst = at_gpr.kernel.interdomain(Sx, Tx)
print(at_gpr.training_loss())

Xplot = np.linspace(0, 15, 100).reshape(-1, 1).astype(float)

f_mean, f_var = at_gpr.predict_f(Xplot)
y_mean, y_var = at_gpr.predict_y(Xplot)
print(f_mean.shape, f_var.shape)

f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)

plt.plot(Sx, Sy, "kx", mew=2, label="input data")
plt.plot(Tx, Ty, "kx", mew=2, label="input data")
plt.plot(Xplot, f_mean, "-", color="C0", label="mean")
plt.plot(Xplot, f_lower, "--", color="C0", label="f 95% confidence")
plt.plot(Xplot, f_upper, "--", color="C0")
plt.fill_between(
    Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.1
)
# plt.plot(Xplot, y_lower, ".", color="C0", label="Y 95% confidence")
# plt.plot(Xplot, y_upper, ".", color="C0")
# plt.fill_between(
#     Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
# )
plt.legend()
plt.show()
