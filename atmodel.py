import gpflow as gpf
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
            """Debug tool for if matrices turn non-singular"""
            s = tf.linalg.svd(M, compute_uv=False)
            tf.print(f"condition number {name}", s[0]/s[-1], s[0], s[-1])
            return 

        Xs, Ys = self.data
        Kall = self.kernel(Xs)

        # Determine what is source and what is target (currently use only one target var)
        As = (Ys[:,1]  == 0)
        Bs = (Ys[:,1]  == 1)

        Ax, Ay = tf.reshape(Xs[:,0][As], (-1, 1)), tf.reshape(Ys[:,0][As], (-1, 1))
        Bx, By = tf.reshape(Xs[:,0][Bs], (-1, 1)), tf.reshape(Ys[:,0][Bs], (-1, 1))

        indices_A = tf.reshape(tf.where(As), [-1])
        indices_B = tf.reshape(tf.where(Bs), [-1])

        # Compute all sub-matrices by collecting the kernel values for source and target. 
        Kaa = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_A, axis=1) + tf.linalg.diag(tf.squeeze(self.likelihood.source.variance_at(Ax)))
        Kab = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_B, axis=1)
        Kbb = tf.gather(tf.gather(Kall, indices_B, axis=0), indices_B, axis=1) + tf.linalg.diag(tf.squeeze(self.likelihood.target.variance_at(Bx)))
        Kba = tf.transpose(Kab)

        # Cholesky pain starts here
        Laa = tf.linalg.cholesky(Kaa) 
        A = tf.linalg.triangular_solve(Laa, Ay) # Laa * x = Ay, so then x = inv(Laa) * Ay
        V = tf.linalg.triangular_solve(Laa, Kab) # Laa * x = Kab, so then x = inv(Laa) * Kab
        
        mu_t = tf.matmul(V, A, transpose_a=True)  
        # Math:
        # transpose(inv(Laa) * Kab) = Kba * inv(Laa).T
        # mu_t = Kba * inv(Laa).T * inv(Laa) * Ay 
        #        <=> Kba * inv(LaaLaa.T) * Ay

        C_t = Kbb - tf.matmul(V, V, transpose_a=True)
        # Math:
        # We want to compute Kbb - Kba inv(Kaa) Kab, so since
        # Kba inv(Kaa) Kab <=> Kba * inv(Laa).T * inv(Laa) * Kab <=> Kba * (Laa * Laa.T) * Kab

        L_t = tf.linalg.cholesky(C_t)

        # product of cholesky diagonals is determinant, therefore sum of log cholesky is also determinant
        logdet_t = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_t)))
        
        delta = By - mu_t
        alpha_t = tf.linalg.triangular_solve(L_t, delta) 
        # Math
        # L_t * x = delta, then alpha_t = inv(L_t)*(By - mu_t)
        
        n_target = tf.cast(tf.shape(Bx)[0], Bx.dtype)  # We compute for the target (given source)
        quad_part = tf.matmul(alpha_t, alpha_t, transpose_a=True)
        
        lml = -0.5 * (logdet_t + quad_part + n_target * tf.cast(tf.math.log(2*np.pi), Ax.dtype))
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
        err = (Ys - self.mean_function(Xs))[:,0][:,None]
      
        As = (Ys[:,1] == 0)
        Bs = (Ys[:,1] == 1)

        knn = self.kernel(Xnew, full_cov=full_cov)

        Ax, Ay = tf.reshape(Xs[:,0][As], (-1, 1)), tf.reshape(Ys[:,0][As], (-1, 1))
        Bx, By = tf.reshape(Xs[:,0][Bs], (-1, 1)), tf.reshape(Ys[:,0][Bs], (-1, 1))

        indices_A = tf.reshape(tf.where(As), [-1])
        indices_B = tf.reshape(tf.where(Bs), [-1])
       
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
        fvar = tf.expand_dims(tf.linalg.diag_part(knn - cond), 1)
        #tf.print("fm0", f_mean_zero.shape)
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, fvar


class SparseCMOGP(gpf.models.GPModel, InternalDataTrainingLossMixin):
    def __init__(self, data, kernel, likelihood, mean_function=None, num_latent_gps=2, inducing_variable=[], exact_target=False, jitter=1e-6):
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.data = data   
        self.kernel = kernel
        self.likelihood = likelihood
        self.inducing_variable = InducingPoints(inducing_variable[:,0].reshape(-1, 1))
        self.inducing_indices = inducing_variable[:,1].reshape(-1, 1)
        self.mean_function = gpf.mean_functions.Zero()
        self.conditioning_indices = [0]
        self.opt_logs = []
        self.exact_target = exact_target
        self.jitter = tf.cast(jitter, tf.float64)

        
    def conditional_likelihood(self, *args, decompose_likelihood=False, **kwargs) -> tf.Tensor:
        # Rebuild: look at all unique indices. 
        # Then, figure out what set to condition on
        # Then, gather all conditioning variables as source, rest as target.
        def get_condition_number(M, name=""):
            s = tf.linalg.svd(M, compute_uv=False)
            tf.print(f"condition number {name}", s[0]/s[-1], s[0], s[-1])
            return 

        Xs, Ys = self.data
        # Kall = self.kernel(Xs)
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
        Kbb = self.kernel(Xs[Bs]) 
        Kaa = self.kernel(Xs[As])

        # Approximate matrices
        inducing_variable = tf.concat((self.inducing_variable.Z, self.inducing_indices), -1)
        Kmm = self.kernel(inducing_variable) + tf.eye(len(inducing_variable), dtype=tf.float64) * self.jitter
        Kma = self.kernel(inducing_variable, Xs[As])
        Kmb = self.kernel(inducing_variable, Xs[Bs])
        Kam = tf.transpose(Kma)
    
        # Steps to avoid explicit inverse of Kmm
        L_Kmm = tf.linalg.cholesky(Kmm) 
        Lmm_inv_kma = tf.linalg.triangular_solve(L_Kmm, Kma) # L_Kmm x = Kma, x = L_Kmm^-1 Kma
        Lmm_inv_kmb = tf.linalg.triangular_solve(L_Kmm, Kmb) # L_Kmm x = Kmb, x = L_Kmm^-1 Kmb

        # Full approximations of the exact matrices
        Qaa = tf.matmul(tf.transpose(Lmm_inv_kma), Lmm_inv_kma) 
        Qab = tf.matmul(tf.transpose(Lmm_inv_kma), Lmm_inv_kmb)
        Qba = tf.transpose(Qab)

        # Alternative way of computing inv(Qaa) implicitly, should be a bit slower.
        #LQaa = tf.linalg.cholesky(Qaa + tf.eye(len(Qaa), dtype=tf.float64) * self.jitter) 
        #LQaa_Kab = tf.linalg.triangular_solve(LQaa, Qab)
        #Qbb_given_aa = tf.matmul(LQaa_Kab, LQaa_Kab, transpose_a=True)

        Lambda = tf.linalg.diag(tf.linalg.diag_part(Kaa - Qaa))

        # Want to compute: inv((Lambda + Kam inv(Kmm) Kma))
        # Woodbury: (A+UCV)^-1 = inv(A) - inv(A) U inv((inv(C) + V inv(A) U)) V inv(A)
        # A = Lambda 
        # U = V.T = Kam 
        # C = inv(Kmm), inv(C) = Kmm
        #Linv = tf.linalg.diag(1./tf.linalg.diag_part(Lambda))  # Lambda is diagonal, so this is the fast inverse.
        D_a = tf.linalg.diag_part(Lambda) + tf.squeeze(self.likelihood.source.variance_at(Ax))
        Linv = tf.linalg.diag(1./D_a)
        woodbury_middle = tf.linalg.cholesky(Kmm + Kma @ Linv @ Kam) # This needs to be inverted, dimension is m by m. 
        right_part = tf.linalg.triangular_solve(woodbury_middle, Kma @ Linv) # LWM * x = Kma @ Linv, so x = inv(LWM) @ Kma @ Linv
        Qaa_inv = Linv - tf.matmul(right_part, right_part, transpose_a=True) # Full Woodbury. 
        Qbb_given_aa = Qba @ Qaa_inv @ Qab
        Qbb = tf.matmul(tf.transpose(Lmm_inv_kmb), Lmm_inv_kmb)


        # Compute conditional mu similar to exact version
        Qaa_L = tf.linalg.cholesky(Qaa + Lambda + tf.linalg.diag(tf.squeeze(self.likelihood.source.variance_at(Ax))))
        V = tf.linalg.triangular_solve(Qaa_L, Qab) # Qaa_L x = Qba, x = inv(Qaa_L) Qab
        A = tf.linalg.triangular_solve(Qaa_L, Ay) # Qaa_L x = Ay, x = inv(Qaa_L) Ay 
        mu_t = tf.matmul(V, A, transpose_a=True)
        
        # Compute conditional variance *bb - Qba inv(Qaa) Qab
        if self.exact_target: # Exact target matrix Kbb, approximate Kba inv(Kaa) Kab <~> Qba inv(Qaa) Qab
            C_t = Kbb - Qbb_given_aa + tf.linalg.diag(tf.squeeze(self.likelihood.target.variance_at(Bx))) 
        else: # Approximate everything.
            Lambda_target =  tf.linalg.diag(tf.linalg.diag_part(Kbb - Qbb))
            C_t = Qbb - Qbb_given_aa + Lambda_target + tf.linalg.diag(tf.squeeze(self.likelihood.target.variance_at(Bx))) 

        delta = By - mu_t
        L_t = tf.linalg.cholesky(C_t  + tf.eye(tf.shape(C_t)[0], dtype=tf.float64) * self.jitter)
        alpha_t = tf.linalg.triangular_solve(L_t, delta)
        n_source = tf.cast(tf.shape(Bx)[0], Bx.dtype)
        logdet_t = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_t)))

        quad = tf.matmul(alpha_t, alpha_t, transpose_a=True)
        lml = -0.5 * (logdet_t + quad + n_source * tf.cast(tf.math.log(2*tf.constant(np.pi)), Ax.dtype))

        if not decompose_likelihood:
            return tf.squeeze(lml)
        else:
            return logdet_t.numpy().flatten(), quad.numpy().flatten(), (n_source * tf.cast(tf.math.log(2*tf.constant(np.pi)), Ax.dtype)).numpy().flatten()

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

        err = (Ys - self.mean_function(Xs))[:,0][:,None]
      
        As = (Ys[:,1] == 0)
        Bs = (Ys[:,1] == 1)

        knn = self.kernel(Xnew, full_cov=full_cov) #+ tf.squeeze(self.likelihood.target.variance_at(Xnew))

        Ax, Ay = tf.reshape(Xs[:,0][As], (-1, 1)), tf.reshape(Ys[:,0][As], (-1, 1))
        Bx, By = tf.reshape(Xs[:,0][Bs], (-1, 1)), tf.reshape(Ys[:,0][Bs], (-1, 1))

        indices_A = tf.reshape(tf.where(As), [-1])
        indices_B = tf.reshape(tf.where(Bs), [-1])
       
        # Compute exact kernel parts
        Kbb = self.kernel(Xs[Bs])
        Kaa = self.kernel(Xs[As])

        # Compute inducing points x rest data 
        inducing_variable = tf.concat((self.inducing_variable.Z, self.inducing_indices), -1)
        Kma = self.kernel(inducing_variable, Xs[As])
        Kmb = self.kernel(inducing_variable, Xs[Bs])
        Kam = tf.transpose(Kma)
        Kmm = self.kernel(inducing_variable) + np.eye(len(Xind.numpy()), dtype=np.float64) * self.jitter
        L_Kmm = tf.linalg.cholesky(Kmm)
        Lmm_inv_kma = tf.linalg.triangular_solve(L_Kmm, Kma)  # Ax = b, x= A-1b
        Lmm_inv_kmb = tf.linalg.triangular_solve(L_Kmm, Kmb)

        # Use above to compute approximations and diagonal
        Qaa = tf.matmul(tf.transpose(Lmm_inv_kma), Lmm_inv_kma)
        Qab = tf.matmul(tf.transpose(Lmm_inv_kma), Lmm_inv_kmb)
        Qba = tf.transpose(Qab)
        Qbb = tf.matmul(tf.transpose(Lmm_inv_kmb), Lmm_inv_kmb)
        Lambda = tf.linalg.diag(tf.linalg.diag_part(Kaa - Qaa))
        Lambda_t = tf.linalg.diag(tf.linalg.diag_part(Kbb - Qbb))

        #tf.print(self.inducing_variable.Z)
        if self.exact_target:
            K_fitc = tf.concat((tf.concat((Qaa + Lambda, Qba), 0), tf.concat((Qab, Kbb), 0)), 1)
        else:
            K_fitc = tf.concat((tf.concat((Qaa + Lambda, Qba), 0), tf.concat((Qab, Qbb + Lambda_t), 0)), 1)
        noise_ab = tf.concat((self.likelihood.source.variance_at(Ax), self.likelihood.target.variance_at(Bx)), 0)[:, 0]
        L_Kfitc = tf.linalg.cholesky(K_fitc + tf.linalg.diag(noise_ab) + tf.eye(len(Xs), dtype=tf.float64) * self.jitter)
        #L_Kfitc = tf.linalg.cholesky(K_fitc + tf.eye(len(Xs), dtype=tf.float64) * self.jitter)

        #tf.print(kmm_plus_s.shape)
        # Construct Kmn
        Knm = self.kernel(Xs, inducing_variable) @ tf.linalg.inv(Kmm) @ self.kernel(inducing_variable, Xnew)
        LfitcKnm = tf.linalg.triangular_solve(L_Kfitc, Knm)
        Kfitcy = tf.linalg.cholesky_solve(L_Kfitc, err)
      
        cond = tf.transpose(LfitcKnm) @ LfitcKnm  
        f_mean_zero = tf.transpose(Knm) @ Kfitcy
        f_var = tf.expand_dims(tf.linalg.diag_part(knn - cond), 1)
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
