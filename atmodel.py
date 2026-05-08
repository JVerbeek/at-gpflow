import gpflow as gpf
from atkernel import TransferKernel
import tensorflow as tf
import numpy as np
import sys
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
import matplotlib.pyplot as plt
from atlikelihood import TransferLikelihood

from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.utilities import to_default_float, assert_params_false, add_likelihood_noise_cov
gpf.config.set_default_jitter(0.001)
from gpflow.config import default_jitter, default_float
from gpflow.utilities import to_default_float
from gpflow.models.util import InducingPointsLike, data_input_to_tensor, inducingpoint_wrapper
from gpflow.inducing_variables import InducingPoints
from typing import NamedTuple



class ConditionalMOGP(gpf.models.GPModel, InternalDataTrainingLossMixin):
    def __init__(self, data, kernel, likelihood, mean_function=None, num_latent_gps=2):
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.data = data   
        self.kernel = kernel
        self.likelihood = likelihood
        self.mean_function = gpf.mean_functions.Zero()
        self.conditioning_indices = [0]

    def conditional_likelihood(self, *args, **kwargs) -> tf.Tensor:
        def get_condition_number(M, name=""):
            s = tf.linalg.svd(M, compute_uv=False)
            tf.print(f"condition number {name}", s[0]/s[-1], s[0], s[-1])
            return 

        # Rebuild: look at all unique indices. 
        # Then, figure out what set to condition on
        # Then, gather all conditioning variables as source, rest as target.

        Xs, Ys = self.data
        Kall = self.kernel(Xs)

        # Determine what is source and what is target (currently use only one target var)
        As = (Ys[:,1]  == 0)
        Bs = (Ys[:,1]  == 1)

        Ax, Ay = tf.reshape(Xs[:,0][As], (-1, 1)), tf.reshape(Ys[:,0][As], (-1, 1))
        Bx, By = tf.reshape(Xs[:,0][Bs], (-1, 1)), tf.reshape(Ys[:,0][Bs], (-1, 1))

        indices_A = tf.reshape(tf.where(As), [-1])
        indices_B = tf.reshape(tf.where(Bs), [-1])
        Kaa = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_A, axis=1) + tf.linalg.diag(tf.squeeze(self.likelihood.source.variance_at(Ax)))
        Kab = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_B, axis=1)
        Kbb = tf.gather(tf.gather(Kall, indices_B, axis=0), indices_B, axis=1) + tf.linalg.diag(tf.squeeze(self.likelihood.target.variance_at(Bx)))
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======

>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        Kba = tf.transpose(Kab)

        # Assume kernel * coregion
        Lss = tf.linalg.cholesky(Kaa)
<<<<<<< HEAD
<<<<<<< HEAD
 
        A = tf.linalg.cholesky_solve(Lss, Ay)
        V = tf.linalg.cholesky_solve(Lss, Kab)
=======
        
        A = tf.linalg.triangular_solve(Lss, Ay, lower=True)
        V = tf.linalg.triangular_solve(Lss, Kab, lower=True)
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
        
        A = tf.linalg.triangular_solve(Lss, Ay, lower=True)
        V = tf.linalg.triangular_solve(Lss, Kab, lower=True)
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        
        mu_t = tf.matmul(V, A, transpose_a=True)
        C_t = Kbb - tf.matmul(Kba, V)
        L_t = tf.linalg.cholesky(C_t)
<<<<<<< HEAD
<<<<<<< HEAD

        logdet_t = tf.linalg.logdet(C_t)
        
        delta = By - mu_t
        alpha_t = tf.linalg.triangular_solve(L_t, delta)
=======
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        get_condition_number(C_t)

        logdet_t = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_t)))
        
        delta = By - mu_t
        alpha_t = tf.linalg.triangular_solve(L_t, delta, lower=True)
<<<<<<< HEAD
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        
        n_source = tf.cast(tf.shape(Bx)[0], Bx.dtype)
        
        lml = -0.5 * (logdet_t + tf.reduce_sum(tf.square(alpha_t)) + n_source * tf.cast(tf.math.log(2*np.pi), Ax.dtype))
        return tf.squeeze(lml)

    def maximum_log_likelihood_objective(self):
        return self.conditional_likelihood()


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

        Xs, Ys = self.data
        Kall = self.kernel(Xs)
<<<<<<< HEAD
<<<<<<< HEAD
        err = (Ys - self.mean_function(Xs))[:,0][:,None]
      
        As = (Ys[:,1] == 0)
        Bs = (Ys[:,1] == 1)

        knn = self.kernel(Xnew, full_cov=full_cov)
=======
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        tf.print("kall", Kall.shape)
        err = (Ys - self.mean_function(Xs))[:,0][:,None]
      
        tf.print("Err", err.shape)
        As = (Ys[:,1] == 0)
        Bs = (Ys[:,1] == 1)

        knn = self.kernel(Xnew, full_cov=full_cov) + tf.squeeze(self.likelihood.target.variance_at(Xnew))
<<<<<<< HEAD
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc

        Ax, Ay = tf.reshape(Xs[:,0][As], (-1, 1)), tf.reshape(Ys[:,0][As], (-1, 1))
        Bx, By = tf.reshape(Xs[:,0][Bs], (-1, 1)), tf.reshape(Ys[:,0][Bs], (-1, 1))

        indices_A = tf.reshape(tf.where(As), [-1])
        indices_B = tf.reshape(tf.where(Bs), [-1])
<<<<<<< HEAD
<<<<<<< HEAD
       
        # Compute exact kernel parts
        Kbb = tf.gather(tf.gather(Kall, indices_B, axis=0), indices_B, axis=1) 
        Kaa = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_A, axis=1) 
        Kab = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_B, axis=1)
        Kba = tf.transpose(Kab)
        Kmm = tf.concat((tf.concat((Kaa, Kba), 0), tf.concat((Kab, Kbb), 0)), 1)
        noise = tf.squeeze(tf.linalg.diag(tf.concat((self.likelihood.source.variance_at(Ax), self.likelihood.target.variance_at(Bx)), 0)))
        kmm_plus_s = Kmm + tf.linalg.diag(noise)

        #tf.print(kmm_plus_s.shape)
        # Construct Kmn
        Knm = self.kernel(Xs, Xnew)
        Lkmm = tf.linalg.cholesky(kmm_plus_s)
        KnminvLmm = tf.linalg.triangular_solve(Lkmm, Knm) #Lkmm-1 Knm
        tf.print(KnminvLmm.shape)
        Lkmmy = tf.linalg.triangular_solve(Lkmm, err) # Lkmm-1 err
      
        cond = tf.transpose(KnminvLmm) @ KnminvLmm
        f_mean_zero = tf.transpose(KnminvLmm) @ Lkmmy
        f_var = tf.expand_dims(tf.linalg.diag_part(knn - cond), 1)
        fvar = knn - tf.reduce_sum(tf.square(Lkmmy), -2)  # [..., N]
        fvar = tf.expand_dims(fvar, -2)
        #tf.print("fm0", f_mean_zero.shape)
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, fvar
=======
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc

        Caa = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_A, axis=1)
        Cab = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_B, axis=1)
        Cbb = tf.gather(tf.gather(Kall, indices_B, axis=0), indices_B, axis=1)
        Cba = tf.transpose(Cab)
        
        # Construct the kernel matrix Kmm
        Kmm = tf.concat((tf.concat((Caa, tf.linalg.matrix_transpose(Cab)), 0), tf.concat((Cab, Cbb), 0)), 1)
        noise = tf.squeeze(tf.linalg.diag(tf.concat((self.likelihood.source.variance_at(Ax), self.likelihood.target.variance_at(Bx)), 0)))
        kmm_plus_s = Kmm + tf.linalg.diag(noise)
        
        # Construct Kmn
        kmn = self.kernel(Xs, Xnew)
        tf.print("kmn", kmn.shape)

        conditional = gpf.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        tf.print("fm0", f_mean_zero.shape)
        f_mean = f_mean_zero + self.mean_function(Xnew[:,0][:,None])
        return f_mean, f_var
<<<<<<< HEAD
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc

class AdaptiveTransferGPR(gpf.models.GPModel, InternalDataTrainingLossMixin):
    def __init__(self, data_source, data_target, kernel):
        self.kernel = kernel
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
        
    def get_mean_function(self):   # Some of this should probably be in a precompute
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
        return mu_t
        
    def maximum_log_likelihood_objective(self):
        return self.adaptive_log_marginal_likelihood()
    
    def adaptive_log_marginal_likelihood(self, decompose=False):
        Sx, Sy = self.data_source
        Tx, Ty = self.data_target
        
        Kss = self.kernel.kernel(Sx, Sx) * self.kernel.get_B()[0, 0] + tf.linalg.diag(tf.squeeze(self.likelihood.source.variance_at(Sx)))
        Kst = self.kernel.interdomain(Sx, Tx)
        Kts = tf.linalg.matrix_transpose(Kst)
        Ktt = self.kernel.kernel(Tx, Tx) * self.kernel.get_B()[1, 1] + tf.linalg.diag(tf.squeeze(self.likelihood.target.variance_at(Tx)))

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
        if decompose:
            return -0.5 * logdet_t, -0.5 * tf.reduce_sum(tf.square(alpha_t)), -0.5 * n_target * tf.cast(tf.math.log(2*np.pi), Sx.dtype)
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
        Css = self.kernel.kernel(Sx, Sx) * self.kernel.get_B()[0, 0]
        Cst = self.kernel.interdomain(Sx, Tx)
        Ctt = self.kernel.kernel(Tx, Tx) * self.kernel.get_B()[1, 1]
        Kmm = tf.concat((tf.concat((Css, tf.linalg.matrix_transpose(Cst)), 0), tf.concat((Cst, Ctt), 0)), 1)
        noise = tf.squeeze(tf.linalg.diag(tf.concat((self.likelihood.source.variance_at(Sx), self.likelihood.target.variance_at(Tx)), 0)))
        kmm_plus_s = Kmm + tf.linalg.diag(noise)
        
        # Construct Kmn
        Ks = self.kernel.interdomain(Xnew, Sx)
        Kt = self.kernel.kernel(Xnew, Tx) * self.kernel.get_B()[1, 1]
        kmn = tf.linalg.matrix_transpose(tf.concat((Ks, Kt), 1))

        conditional = gpf.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var
    
def tests():    
    # # Tests      
    Sx = np.linspace(0, 1, 100).reshape(-1, 1)
    Sy = (np.sin(Sx * 5) + np.random.normal(0, 0.1, size=100).reshape(-1, 1)) / 0.1
    Tx = np.linspace(0.5, 1.5, 100).reshape(-1, 1)

    f = 1 * np.exp(Tx*0.2)  # Change 0.2 to something else to get a more/less similar target
    Ty = (np.sin(Tx) + np.random.normal(0, 0.1, size=100).reshape(-1, 1)) / 0.1
    plt.plot(Sx, Sy)
    plt.plot(Tx, Ty)
    plt.show()
    at_gpr = AdaptiveTransferGPR((Sx, Sy), (Tx, Ty), gpf.kernels.RBF())

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


import gpflow as gpf
from atkernel import TransferKernel
import tensorflow as tf
import numpy as np
import sys
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
import matplotlib.pyplot as plt
from atlikelihood import TransferLikelihood

from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.utilities import to_default_float, assert_params_false, add_likelihood_noise_cov
from gpflow.config import default_jitter, default_float
from gpflow.utilities import to_default_float
from gpflow.models.util import InducingPointsLike, data_input_to_tensor, inducingpoint_wrapper
from gpflow.inducing_variables import InducingPoints
from typing import NamedTuple
<<<<<<< HEAD
<<<<<<< HEAD
import time 

class SparseCMOGP(gpf.models.GPModel, InternalDataTrainingLossMixin):
    def __init__(self, data, kernel, likelihood, mean_function=None, num_latent_gps=2, inducing_variable=[], exact_target=False, jitter=1e-6):
=======

class SparseCMOGP(gpf.models.GPModel, InternalDataTrainingLossMixin):
    def __init__(self, data, kernel, likelihood, mean_function=None, num_latent_gps=2, inducing_variable=[]):
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======

class SparseCMOGP(gpf.models.GPModel, InternalDataTrainingLossMixin):
    def __init__(self, data, kernel, likelihood, mean_function=None, num_latent_gps=2, inducing_variable=[]):
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.data = data   
        self.kernel = kernel
        self.likelihood = likelihood
<<<<<<< HEAD
<<<<<<< HEAD
        self.inducing_variable = InducingPoints(inducing_variable[:,0].reshape(-1, 1))
        self.inducing_indices = inducing_variable[:,1].reshape(-1, 1)
        self.mean_function = gpf.mean_functions.Zero()
        self.conditioning_indices = [0]
        self.opt_logs = []
        self.exact_target = exact_target
        self.jitter = tf.cast(jitter, tf.float64)

        
    def conditional_likelihood(self, *args, **kwargs) -> tf.Tensor:
=======
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        self.inducing_variable = InducingPoints(inducing_variable)
        self.mean_function = gpf.mean_functions.Zero()
        self.conditioning_indices = [0]
        self.opt_logs = []

        
    def conditional_likelihood(self, *args, **kwargs) -> tf.Tensor:

<<<<<<< HEAD
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        # Rebuild: look at all unique indices. 
        # Then, figure out what set to condition on
        # Then, gather all conditioning variables as source, rest as target.
        def get_condition_number(M, name=""):
            s = tf.linalg.svd(M, compute_uv=False)
            tf.print(f"condition number {name}", s[0]/s[-1], s[0], s[-1])
            return 

        Xs, Ys = self.data
<<<<<<< HEAD
<<<<<<< HEAD
        # Kall = self.kernel(Xs)
=======
        Kall = self.kernel(Xs)
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
        Kall = self.kernel(Xs)
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        Xind = self.inducing_variable.Z
        #tf.print(Kall.shape)

        # Determine what is source and what is target (currently use only one target var)
        As = (Ys[:,1]  == 0)
        Bs = (Ys[:,1]  == 1)

        Ax, Ay = tf.reshape(Xs[:,0][As], (-1, 1)), tf.reshape(Ys[:,0][As], (-1, 1))
        Bx, By = tf.reshape(Xs[:,0][Bs], (-1, 1)), tf.reshape(Ys[:,0][Bs], (-1, 1))

        indices_A = tf.reshape(tf.where(As), [-1])
        indices_B = tf.reshape(tf.where(Bs), [-1])

        # Exact matrices
<<<<<<< HEAD
<<<<<<< HEAD
        Kbb = self.kernel(Xs[Bs])# + tf.linalg.diag(tf.squeeze(self.likelihood.target.variance_at(Bx))) 
        Kaa = self.kernel(Xs[As])
        # Kab = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_B, axis=1)
        # Kba = tf.transpose(Kab)

        # Approximate matrices
        inducing_variable = tf.concat((self.inducing_variable.Z, self.inducing_indices), -1)
        Kmm = self.kernel(inducing_variable) + tf.eye(len(inducing_variable), dtype=tf.float64) * self.jitter
        Kma = self.kernel(inducing_variable, Xs[As])
        Kmb = self.kernel(inducing_variable, Xs[Bs])
        Kam = tf.transpose(Kma)
    
=======
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        Kbb = tf.gather(tf.gather(Kall, indices_B, axis=0), indices_B, axis=1)# + tf.linalg.diag(tf.squeeze(self.likelihood.target.variance_at(Bx))) 
        Kaa = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_A, axis=1) 
        Kab = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_B, axis=1)
        Kba = tf.transpose(Kab)

        # Approximate matrices
        Kma = self.kernel(self.inducing_variable.Z, Xs[As])
        Kmb = self.kernel(self.inducing_variable.Z, Xs[Bs])
        Kam = tf.transpose(Kma)
        Kmm = self.kernel(self.inducing_variable.Z) + np.eye(len(Xind), dtype=np.float64) * 1e-6
<<<<<<< HEAD
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        L_Kmm = tf.linalg.cholesky(Kmm)
        Lmm_inv_kma = tf.linalg.triangular_solve(L_Kmm, Kma)
        Lmm_inv_kmb = tf.linalg.triangular_solve(L_Kmm, Kmb)

        # Full approximations of the exact matrices
        Qaa = tf.matmul(tf.transpose(Lmm_inv_kma), Lmm_inv_kma) 
        Qab = tf.matmul(tf.transpose(Lmm_inv_kma), Lmm_inv_kmb)
        Qba = tf.transpose(Qab)
        
<<<<<<< HEAD
<<<<<<< HEAD
        #LQaa = tf.linalg.cholesky(Qaa + tf.eye(len(Qaa), dtype=tf.float64) * self.jitter) 
=======
        #LQaa = tf.linalg.cholesky(Qaa + tf.eye(len(Qaa), dtype=tf.float64) * 1e-6) 
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
        #LQaa = tf.linalg.cholesky(Qaa + tf.eye(len(Qaa), dtype=tf.float64) * 1e-6) 
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        #LQaa_Kab = tf.linalg.triangular_solve(LQaa, Qab)
        
        # Woodbury
        Lambda = tf.linalg.diag(tf.linalg.diag_part(Kaa - Qaa))
<<<<<<< HEAD
<<<<<<< HEAD
        tf.print(Lambda.shape)
        Linv = tf.linalg.diag(1./tf.linalg.diag_part(Lambda))
        tf.print(Linv.shape)
        right_part = tf.linalg.triangular_solve(tf.linalg.cholesky(Kmm + Kma @ Linv @ Kam), Kma @ Linv)
        Qaa_inv = Linv - tf.matmul(right_part, right_part, transpose_a=True)
        Qbb_given_aa = Qba @ Qaa_inv @ Qab

        #Qbb_given_aa = tf.matmul(LQaa_Kab, LQaa_Kab, transpose_a=True)
=======
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        Linv = tf.linalg.inv(Lambda)
        Kminv = tf.linalg.inv(Kmm)
        right_part = tf.linalg.triangular_solve(tf.linalg.cholesky(Kmm + Kma @ Linv @ Kam), Kma @ Linv)
        Qaa_inv = Linv - tf.matmul(right_part, right_part, transpose_a=True)
        Qbb_approx = Qba @ Qaa_inv @ Qab

        #Qbb_approx = tf.matmul(LQaa_Kab, LQaa_Kab, transpose_a=True)
<<<<<<< HEAD
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        Qbb = tf.matmul(tf.transpose(Lmm_inv_kmb), Lmm_inv_kmb)

        # FITC-type diagonal

        # Compute conditional mu
        Qaa_L = tf.linalg.cholesky(Qaa + Lambda + tf.linalg.diag(tf.squeeze(self.likelihood.source.variance_at(Ax)))) 
        gamma = tf.linalg.cholesky_solve(Qaa_L, Ay)
        mu_t = tf.matmul(Qba, gamma)
        
        # Compute conditional variance
<<<<<<< HEAD
<<<<<<< HEAD
        if self.exact_target:
            C_t = Kbb - Qbb_given_aa + tf.linalg.diag(tf.squeeze(self.likelihood.target.variance_at(Bx))) 
        else:
            Lambda_target =  tf.linalg.diag(tf.linalg.diag_part(Kbb - Qbb))
            C_t = Qbb - Qbb_given_aa + Lambda_target + tf.linalg.diag(tf.squeeze(self.likelihood.target.variance_at(Bx))) 

        delta = By - mu_t
        L_t = tf.linalg.cholesky(C_t  + tf.eye(tf.shape(C_t)[0], dtype=tf.float64) * self.jitter)
        alpha_t = tf.linalg.triangular_solve(L_t, delta)
        n_source = tf.cast(tf.shape(Bx)[0], Bx.dtype)
        logdet_t = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_t)))
        quad = tf.reduce_sum(alpha_t ** 2)
        lml = -0.5 * (logdet_t + quad + n_source * tf.cast(tf.math.log(2*tf.constant(np.pi)), Ax.dtype))
=======
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        Lambda_target =  tf.linalg.diag(tf.linalg.diag_part(Qbb - Qbb_approx))
        C_t = Kbb - Qbb_approx + Lambda_target + tf.linalg.diag(tf.squeeze(self.likelihood.target.variance_at(Bx))) 

        delta = By - mu_t
        L_t = tf.linalg.cholesky(C_t  + tf.eye(tf.shape(C_t)[0], dtype=tf.float64) * 1e-6)
        alpha_t = tf.linalg.cholesky_solve(L_t, delta)
        n_source = tf.cast(tf.shape(Bx)[0], Bx.dtype)
        logdet_t = tf.linalg.logdet(C_t)
        quad = tf.reduce_sum(alpha_t ** 2)
        lml = -0.5 * (logdet_t + quad + n_source * tf.cast(tf.math.log(2*np.pi), Ax.dtype))
        #tf.print(quad, logdet_t)
<<<<<<< HEAD
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        return tf.squeeze(lml)

    def maximum_log_likelihood_objective(self):
        return self.conditional_likelihood()


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

        Xs, Ys = self.data
        Xind = self.inducing_variable.Z
<<<<<<< HEAD
<<<<<<< HEAD
        # Kall = self.kernel(Xs)
=======
        Kall = self.kernel(Xs)
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
        Kall = self.kernel(Xs)
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        err = (Ys - self.mean_function(Xs))[:,0][:,None]
      
        As = (Ys[:,1] == 0)
        Bs = (Ys[:,1] == 1)

        knn = self.kernel(Xnew, full_cov=full_cov) + tf.squeeze(self.likelihood.target.variance_at(Xnew))

        Ax, Ay = tf.reshape(Xs[:,0][As], (-1, 1)), tf.reshape(Ys[:,0][As], (-1, 1))
        Bx, By = tf.reshape(Xs[:,0][Bs], (-1, 1)), tf.reshape(Ys[:,0][Bs], (-1, 1))

        indices_A = tf.reshape(tf.where(As), [-1])
        indices_B = tf.reshape(tf.where(Bs), [-1])
       
        # Compute exact kernel parts
<<<<<<< HEAD
<<<<<<< HEAD
        Kbb = self.kernel(Xs[Bs])
        Kaa = self.kernel(Xs[As])
        # Kaa = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_A, axis=1) 
        # Kab = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_B, axis=1)
        # Kba = tf.transpose(Kab)

        # Compute inducing points x rest data 
        inducing_variable = tf.concat((self.inducing_variable.Z, self.inducing_indices), -1)
        Kma = self.kernel(inducing_variable, Xs[As])
        Kmb = self.kernel(inducing_variable, Xs[Bs])
        Kam = tf.transpose(Kma)
        Kmm = self.kernel(inducing_variable) + np.eye(len(Xind.numpy()), dtype=np.float64) * self.jitter
=======
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        Kbb = tf.gather(tf.gather(Kall, indices_B, axis=0), indices_B, axis=1) 
        Kaa = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_A, axis=1) 
        Kab = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_B, axis=1)
        Kba = tf.transpose(Kab)

        # Compute inducing points x rest data 
        Kma = self.kernel(self.inducing_variable.Z, Xs[As])
        Kmb = self.kernel(self.inducing_variable.Z, Xs[Bs])
        Kam = tf.transpose(Kma)
        Kmm = self.kernel(self.inducing_variable.Z) + np.eye(len(Xind.numpy()), dtype=np.float64) * 1e-6
<<<<<<< HEAD
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        L_Kmm = tf.linalg.cholesky(Kmm)
        Lmm_inv_kma = tf.linalg.triangular_solve(L_Kmm, Kma)
        Lmm_inv_kmb = tf.linalg.triangular_solve(L_Kmm, Kmb)

        # Use above to compute approximations and diagonal
        Qaa = tf.matmul(tf.transpose(Lmm_inv_kma), Lmm_inv_kma)
        Qab = tf.matmul(tf.transpose(Lmm_inv_kma), Lmm_inv_kmb)
        Qba = tf.transpose(Qab)
        Qbb = tf.matmul(tf.transpose(Lmm_inv_kmb), Lmm_inv_kmb)
        Lambda = tf.linalg.diag(tf.linalg.diag_part(Kaa - Qaa))
        Lambda_t = tf.linalg.diag(tf.linalg.diag_part(Kbb - Qbb))

        #tf.print(self.inducing_variable.Z)
<<<<<<< HEAD
<<<<<<< HEAD
        if self.exact_target:
            K_fitc = tf.concat((tf.concat((Qaa + Lambda, Qab), 1), tf.concat((Qba, Kbb), 1)), 0)
        else:
            K_fitc = tf.concat((tf.concat((Qaa + Lambda, Qab), 1), tf.concat((Qba, Qbb + Lambda_t), 1)), 0)
        L_Kfitc = tf.linalg.cholesky(K_fitc + tf.eye(len(Xs), dtype=tf.float64) * self.jitter)

        #tf.print(kmm_plus_s.shape)
        # Construct Kmn
        Knm = self.kernel(Xs, inducing_variable) @ tf.linalg.inv(Kmm) @ self.kernel(inducing_variable, Xnew)
=======
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        K_fitc = tf.concat((tf.concat((Qaa + Lambda, Qab), 1), tf.concat((Qba, Kbb), 1)), 0)
        L_Kfitc = tf.linalg.cholesky(K_fitc + tf.eye(len(Xs), dtype=tf.float64) * 1e-5)

        #tf.print(kmm_plus_s.shape)
        # Construct Kmn
        Knm = self.kernel(Xs, self.inducing_variable.Z) @ tf.linalg.inv(Kmm) @ self.kernel(self.inducing_variable.Z, Xnew)
<<<<<<< HEAD
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
=======
>>>>>>> a157a4f2fbd074ec698829b9a8b75f1fb0231ccc
        LfitcKnm = tf.linalg.triangular_solve(L_Kfitc, Knm)
        Kfitcy = tf.linalg.cholesky_solve(L_Kfitc, err)
      
        cond = tf.transpose(LfitcKnm) @ LfitcKnm  
        f_mean_zero = tf.transpose(Knm) @ Kfitcy
        f_var = tf.expand_dims(tf.linalg.diag_part(knn - cond), 1)
        fvar = knn - tf.reduce_sum(tf.square(LfitcKnm), -2)  # [..., N]
        fvar = tf.expand_dims(fvar, -2)
        #tf.print("fm0", f_mean_zero.shape)
        f_mean = f_mean_zero + self.mean_function(Xnew[:,0][:,None])
        return f_mean, f_var

def optimize(m):
    opt = gpf.optimizers.Scipy()
    res = opt.minimize(m.training_loss, m.trainable_variables, track_loss_history=True, options={"disp": 50})
    plt.plot(res["loss_history"])
    plt.show()


if __name__ == "__main__":
    tests()
