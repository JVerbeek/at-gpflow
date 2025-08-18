import gpflow as gpf
from atkernel import TransferKernel
import tensorflow as tf
import numpy as np
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
import matplotlib.pyplot as plt
from atlikelihood import TransferLikelihood
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.utilities import to_default_float, assert_params_false, add_likelihood_noise_cov
gpf.config.set_default_jitter(0.1)
from gpflow.config import default_jitter, default_float
from gpflow.utilities import to_default_float
from gpflow.models.util import InducingPointsLike, data_input_to_tensor, inducingpoint_wrapper
from gpflow.inducing_variables import InducingPoints
from typing import NamedTuple
class AdaptiveTransferSGPR(gpf.models.GPModel, InternalDataTrainingLossMixin):
    class CommonTensors(NamedTuple):
            sigma_sq: tf.Tensor
            sigma: tf.Tensor
            A: tf.Tensor
            B: tf.Tensor
            LB: tf.Tensor
            AAT: tf.Tensor
            L: tf.Tensor
            
    def __init__(self, data_source, data_target, base_kernel, mu, b, inducing_variable: InducingPointsLike,):
        self.kernel = TransferKernel(mu, b,  base_kernel)
        self.data_source = gpf.models.util.data_input_to_tensor(data_source)
        self.data_target = gpf.models.util.data_input_to_tensor(data_target)
        self.data = tf.concat((data_source[0], data_target[0]), 0), tf.concat((data_source[1], data_target[1]), 0)
        self.mean_function = gpf.mean_functions.Zero()        
        self.inducing_variable: InducingPoints = inducingpoint_wrapper(inducing_variable)
        
        super().__init__(
            kernel=self.kernel,
            likelihood=TransferLikelihood(
                source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()
            ),
            mean_function=self.mean_function,
            num_latent_gps=1
        )

    # type-ignore is because of changed method signature:
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.elbo()
    
    def get_lmb(self):
        return self.kernel.get_lmb()
    
    def _common_calculation(self) -> "SGPR.CommonTensors":
        """
        Matrices used in log-det calculation

        :return:
            * :math:`σ²`,
            * :math:`σ`,
            * :math:`A = L⁻¹K_{uf}/σ`, where :math:`LLᵀ = Kᵤᵤ`,
            * :math:`B = AAT+I`,
            * :math:`LB` where :math`LBLBᵀ = B`,
            * :math:`AAT = AAᵀ`,
        """
        Sx, _ = self.data_source  # [N]
        Tx, _ = self.data_target  # [N]
        x = tf.concat((Sx, Tx), 0)
        iv = self.inducing_variable

        sigma_sq = tf.squeeze(self.likelihood.variance_at(x), axis=-1)  # [N]
        sigma = tf.sqrt(sigma_sq)  # [N]

        kuf = Kuf(iv, self.kernel, x)  # [M, N]
        kuu = Kuu(iv, self.kernel, jitter=gpf.config.default_jitter())  # [M, M]
        L = tf.linalg.cholesky(kuu)  # [M, M]

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf / sigma, lower=True)
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = gpf.utilities.add_noise_cov(AAT, tf.cast(1.0, AAT.dtype))
        LB = tf.linalg.cholesky(B)

        return self.CommonTensors(sigma_sq, sigma, A, B, LB, AAT, L)


    def logdet_term(self, common: "SGPR.CommonTensors") -> tf.Tensor:
        r"""
        Bound from Jensen's Inequality:

        .. math::
            \log |K + σ²I| <= \log |Q + σ²I| + N * \log (1 + \textrm{tr}(K - Q)/(σ²N))

        :param common: A named tuple containing matrices that will be used
        :return: log_det, lower bound on :math:`-.5 * \textrm{output_dim} * \log |K + σ²I|`
        """
        sigma_sq = common.sigma_sq
        LB = common.LB
        AAT = common.AAT

        x, y = self.data 
        outdim = to_default_float(tf.shape(y)[1])
        kdiag = self.kernel(x, full_cov=False)

        # tr(K) / σ²
        trace_k = tf.reduce_sum(kdiag / sigma_sq)
        # tr(Q) / σ²
        trace_q = tf.reduce_sum(tf.linalg.diag_part(AAT))
        # tr(K - Q) / σ²
        trace = trace_k - trace_q

        # 0.5 * log(det(B))
        half_logdet_b = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        # sum log(σ²)
        log_sigma_sq = tf.reduce_sum(tf.math.log(sigma_sq))

        logdet_k = -outdim * (half_logdet_b + 0.5 * log_sigma_sq + 0.5 * trace)
        return logdet_k

    def quad_term(self, common: "SGPR.CommonTensors") -> tf.Tensor:
        """
        :param common: A named tuple containing matrices that will be used
        :return: Lower bound on -.5 yᵀ(K + σ²I)⁻¹y
        """
        sigma = common.sigma
        A = common.A
        LB = common.LB
        x, y = self.data
        err = (y - self.mean_function(x)) / sigma[..., None]

        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True)

        # σ⁻² yᵀy
        err_inner_prod = tf.reduce_sum(tf.square(err))
        c_inner_prod = tf.reduce_sum(tf.square(c))

        quad = -0.5 * (err_inner_prod - c_inner_prod)
        return quad
    
    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        common = self._common_calculation()
        output_shape = tf.shape(self.data[-1])
        num_data = to_default_float(output_shape[0])
        output_dim = to_default_float(output_shape[1])
        const = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        logdet = self.logdet_term(common)
        quad = self.quad_term(common)
        return const + logdet + quad

    def predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        # could copy into posterior into a fused version

        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        X_data, Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        err = Y_data - self.mean_function(X_data)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)

        sigma_sq = tf.squeeze(self.likelihood.variance_at(X_data), axis=-1)
        sigma = tf.sqrt(sigma_sq)

        L = tf.linalg.cholesky(kuu)  # cache alpha, qinv
        A = tf.linalg.triangular_solve(L, kuf / sigma, lower=True)
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(
            num_inducing, dtype=default_float()
        )  # cache qinv
        LB = tf.linalg.cholesky(B)  # cache alpha
        Aerr = tf.linalg.matmul(A, err / sigma[..., None])
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True)
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean + self.mean_function(Xnew), var


    def compute_qu(self):
        """
        Computes the mean and variance of q(u) = N(mu, cov), the variational distribution on
        inducing outputs.

        SVGP with this q(u) should predict identically to SGPR.

        :return: mu, cov
        """
        X_data, Y_data = self.data

        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())

        var = tf.squeeze(self.likelihood.variance_at(X_data), axis=-1)
        std = tf.sqrt(var)
        scaled_kuf = kuf / std
        sig = kuu + tf.matmul(scaled_kuf, scaled_kuf, transpose_b=True)
        sig_sqrt = tf.linalg.cholesky(sig)

        sig_sqrt_kuu = tf.linalg.triangular_solve(sig_sqrt, kuu)

        cov = tf.linalg.matmul(sig_sqrt_kuu, sig_sqrt_kuu, transpose_a=True)
        err = Y_data - self.mean_function(X_data)
        scaled_err = err / std[..., None]
        mu = tf.linalg.matmul(
            sig_sqrt_kuu,
            tf.linalg.triangular_solve(sig_sqrt, tf.linalg.matmul(scaled_kuf, scaled_err)),
            transpose_a=True,
        )

        return mu, cov


    
class AdaptiveTransferGPR(gpf.models.GPModel, InternalDataTrainingLossMixin):
    def __init__(self, data_source, data_target, base_kernel):
        self.kernel = TransferKernel(4, 0.2,  base_kernel)
        self.data_source = gpf.models.util.data_input_to_tensor(data_source)
        self.data_target = gpf.models.util.data_input_to_tensor(data_target)
        self.mean_function = gpf.mean_functions.Zero()
        super().__init__(
            kernel=self.kernel,
            likelihood=TransferLikelihood(
                source=gpf.likelihoods.Gaussian(), target=gpf.likelihoods.Gaussian()
            ),
            mean_function=self.mean_function,
            num_latent_gps=1
        )
        
    def maximum_log_likelihood_objective(self):
        return self.adaptive_log_marginal_likelihood()
    
    def adaptive_log_marginal_likelihood(self):
        Sx, Sy = self.data_source
        Tx, Ty = self.data_target
        
        Kss = self.kernel.kernel(Sx, Sx) + tf.linalg.diag(tf.squeeze(self.likelihood.source.variance_at(Sx)))
        Kst = self.kernel.interdomain(Sx, Tx)
        Kts = tf.linalg.matrix_transpose(Kst)
        Ktt = self.kernel.kernel(Tx, Tx) + tf.linalg.diag(tf.squeeze(self.likelihood.target.variance_at(Tx)))
        
        Lss = tf.linalg.cholesky(Kss)
        
        A = tf.linalg.triangular_solve(Lss, Sy, lower=True)
        V = tf.linalg.triangular_solve(Lss, Kst, lower=True)
        
        mu_t = tf.matmul(V, A, transpose_a=True)
        C_t = Ktt - tf.matmul(Kts, V)
        L_t = tf.linalg.cholesky(C_t)
        logdet_t = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_t)))
        
        delta = Ty - mu_t
        alpha_t = tf.linalg.triangular_solve(L_t, delta, lower=True)
        
        n_target = tf.cast(tf.shape(Tx)[0], Sx.dtype)
        
        lml = -0.5 * (logdet_t + tf.reduce_sum(tf.square(alpha_t)) + n_target * tf.cast(tf.math.log(2*np.pi), Sx.dtype))
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
        y = tf.concat((Sy, Ty), 0)
        err = y - self.mean_function(X)
        
        # Construct Knn
        knn = self.kernel.kernel(Xnew, full_cov=full_cov) + tf.squeeze(self.likelihood.target.variance_at(Xnew))
        

        # Construct the kernel matrix Kmm
        Css = self.kernel.kernel(Sx, Sx)
        Cst = self.kernel.interdomain(Sx, Tx)
        Ctt = self.kernel.kernel(Tx, Tx)
        Kmm = tf.concat((tf.concat((Css, tf.linalg.matrix_transpose(Cst)), 0), tf.concat((Cst, Ctt), 0)), 1)
        noise = tf.squeeze(tf.linalg.diag(tf.concat((self.likelihood.source.variance_at(Sx), self.likelihood.target.variance_at(Tx)), 0)))
        kmm_plus_s = Kmm + tf.linalg.diag(noise)
        
        # Construct Kmn
        Ks = self.kernel.interdomain(Xnew, Sx)
        Kt = self.kernel.kernel(Xnew, Tx)
        kmn = tf.linalg.matrix_transpose(tf.concat((Ks, Kt), 1))

        conditional = gpf.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var
    
def tests():    
    # # Tests      
    Sx = np.linspace(0, 10, 100).reshape(-1, 1)
    Sy = (np.sin(Sx * 1) + np.random.normal(0, 0.1, size=100).reshape(-1, 1)) / 0.1
    Tx = np.linspace(0.5, 10.5, 100).reshape(-1, 1)

    f = 1 * np.exp(Tx*0.2)  # Change 0.2 to something else to get a more/less similar target
    Ty = (np.sin(Tx) + np.random.normal(0, 0.1, size=100).reshape(-1, 1)) / 0.1
    plt.plot(Sx, Sy)
    plt.plot(Tx, Ty)
    plt.show()
    at_gpr = AdaptiveTransferSGPR((Sx, Sy), (Tx, Ty), gpf.kernels.RBF(), inducing_variable=np.linspace(0, 10, 10).reshape(-1, 1))

    print("Training loss value before training:", at_gpr.training_loss().numpy())
    opt = gpf.optimizers.Scipy()
    opt.minimize(at_gpr.training_loss, at_gpr.trainable_variables)
    gpf.utilities.print_summary(at_gpr)
    print("Training loss value after training:", at_gpr.training_loss().numpy())

    # print("Lambda is:", 2 * ((1/(1 + at_gpr.kernel.mu)) ** at_gpr.kernel.b) - 1)


    Xplot = np.linspace(0, 10, 100).reshape(-1, 1).astype(float)
    # plt.imshow(at_gpr.kernel(Xplot))
    # plt.show()
    f_mean, f_var = at_gpr.predict_f(Xplot)
    y_mean, y_var = at_gpr.predict_y(Xplot)
    print(f_mean)

    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)
    y_lower = y_mean - 1.96 * np.sqrt(y_var)
    y_upper = y_mean + 1.96 * np.sqrt(y_var)

    plt.scatter(at_gpr.inducing_variable.Z, np.zeros_like(at_gpr.inducing_variable.Z), marker="^", color="green", label="inducing variable")
    plt.plot(Sx, Sy, "rx", mew=2, label="source data")
    plt.plot(Tx, Ty, "kx", mew=2, label="target data")
    plt.plot(Xplot, f_mean, "-", color="C0", label="mean")
    plt.plot(Xplot, f_lower, "--", color="C0", label="f 95% confidence")
    plt.plot(Xplot, f_upper, "--", color="C0")
    plt.fill_between(
        Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.1
    )
    plt.plot(Xplot, y_lower, ".", color="C0", label="Y 95% confidence")
    plt.plot(Xplot, y_upper, ".", color="C0")
    plt.fill_between(
        Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
    )
    plt.legend()
    plt.show()
