import numpy as np
import tensorflow_probability as tfp
import pandas as pd
from bisect import bisect_left
from scipy import special
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.linalg import inv
import tensorflow as tf
import scipy.linalg
import gpflow
from scipy.stats.distributions import chi2
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from gpflow.utilities import print_summary, positive
np.set_printoptions(suppress=True)
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from gpflow import set_trainable
#import seaborn as sns
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
f64 = gpflow.utilities.to_default_float


# sq exp
class K_full_model_sqexp(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_11 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_11 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        self.variance_22 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_22 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

    #@params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        n = int(X.shape[0]/2)
        m = int(X2.shape[0]/2)

        # Block for experiment 1
        X_matrix_da_exp1 = tf.tile(tf.reshape(X[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp1 = tf.tile(tf.reshape(X2[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp1 = tf.transpose(X_matrix_tr_da_exp1)

        diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1
        K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)

        # Block for experiment 2

        X_matrix_da_exp2 = tf.tile(tf.reshape(X[n:2*n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp2 = tf.tile(tf.reshape(X2[m:2*m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp2 = tf.transpose(X_matrix_tr_da_exp2)

        diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2
        K_dada_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_22)**2)

        K_xx = K_dada_exp1
        K_ff = K_dada_exp2

        # Combine four blocks together

        K_xf = np.zeros((n,m))
        K_fx = np.zeros((n,m))

        #K_xf =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp1))))
        #K_fx =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp2))))

        #diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        #print('K_xf',K_xf)

        K_upper_row = tf.concat([K_xx, K_xf], axis=1)
        K_lower_row = tf.concat([K_fx, K_ff], axis=1)
        K = tf.concat([K_upper_row, K_lower_row], axis=0)
        #return(K_dada * K_dbdb + K_dbda * K_dadb)
        return K

    #@params_as_tensors
    def K_diag(self, X):
        n = int(X.shape[0]/2)
        diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_22)), (-1,))

        block_upper = diag_da_exp1
        block_lower = diag_da_exp2

        diag = tf.reshape(tf.stack([block_upper, block_lower]), (-1,))
        return diag



#Covariance model 1
class K_full_model_1(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_11 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_11 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

    #@params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        #n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        #m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #half_shape=tf.cast(tf.shape(X)[0]/2, dtype=tf.int32)
        #half_shape = np.asscalar(np.array(tf.shape(X)[0])/2).astype(int)
        #print('half_shape',half_shape)

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asarray(tf.cast(tf.shape(X)[0]/2, dtype=tf.int32))
        #m = np.asarray(tf.cast(tf.shape(X2)[0]/2, dtype=tf.int32))

        n = int(X.shape[0]/2)
        m = int(X2.shape[0]/2)
        sqrt5 = np.sqrt(5.0)

        # Block for experiment 1
        X_matrix_da_exp1 = tf.tile(tf.reshape(X[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp1 = tf.tile(tf.reshape(X2[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp1 = tf.transpose(X_matrix_tr_da_exp1)

        diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1#/self.lengthscale_11
        #K_dada_exp1 = self.variance_11*(1.0+sqrt5*diff_X_da_exp1+(2/3)*tf.math.square(diff_X_da_exp1))*tf.math.exp(-sqrt5*diff_X_da_exp1)
        K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)

        # Block for experiment 2

        X_matrix_da_exp2 = tf.tile(tf.reshape(X[n:2*n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp2 = tf.tile(tf.reshape(X2[m:2*m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp2 = tf.transpose(X_matrix_tr_da_exp2)

        diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2#/self.lengthscale_11
        #K_dada_exp2 = self.variance_11*(1.0+sqrt5*diff_X_da_exp2+(2/3)*tf.math.square(diff_X_da_exp2))*tf.math.exp(-sqrt5*diff_X_da_exp2)
        K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)

        K_xx = K_dada_exp1
        K_ff = K_dada_exp2

        # Combine four blocks together

        K_xf = np.zeros((n,m))
        K_fx = np.zeros((n,m))

        #K_xf =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp1))))
        #K_fx =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp2))))

        #diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        #print('K_xf',K_xf)

        K_upper_row = tf.concat([K_xx, K_xf], axis=1)
        K_lower_row = tf.concat([K_fx, K_ff], axis=1)
        K = tf.concat([K_upper_row, K_lower_row], axis=0)
        #return(K_dada * K_dbdb + K_dbda * K_dadb)
        return K

    #@params_as_tensors
    def K_diag(self, X):
        n = int(X.shape[0]/2)
        diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        block_upper = diag_da_exp1
        block_lower = diag_da_exp2

        diag = tf.reshape(tf.stack([block_upper, block_lower]), (-1,))
        return diag


class K_full_model_2(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_11 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_11 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_12 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_12 = gpflow.Parameter(1.0, transform=positive())

        #self.variance_13 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_13 = gpflow.Parameter(0.05, transform=positive())

        self.variance_22 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_22 = gpflow.Parameter(50.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_23 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_23 = gpflow.Parameter(10.0, transform=positive())

    #@params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        #n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        #m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #half_shape=tf.cast(tf.shape(X)[0]/2, dtype=tf.int32)
        #half_shape = np.asscalar(np.array(tf.shape(X)[0])/2).astype(int)
        #print('half_shape',half_shape)

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asarray(tf.cast(tf.shape(X)[0]/2, dtype=tf.int32))
        #m = np.asarray(tf.cast(tf.shape(X2)[0]/2, dtype=tf.int32))

        n = int(X.shape[0]/2)
        m = int(X2.shape[0]/2)

        sqrt5 = np.sqrt(5.0)

        # Block for experiment 1
        X_matrix_da_exp1 = tf.tile(tf.reshape(X[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp1 = tf.tile(tf.reshape(X2[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp1 = tf.transpose(X_matrix_tr_da_exp1)

        #X_matrix_db_exp1 = tf.tile(tf.reshape(X[0:n,1],(n,1)), (1,m))
        #X_matrix_tr_db_exp1 = tf.tile(tf.reshape(X2[0:m,1],(m,1)), (1,n))
        #X_matrix_tr_db_exp1 = tf.transpose(X_matrix_tr_db_exp1)

        #X_matrix_dc_exp1 = tf.tile(tf.reshape(X[0:n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp1 = tf.tile(tf.reshape(X2[0:m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp1 = tf.transpose(X_matrix_tr_dc_exp1)

        #diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1
        #diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1
        #diff_X_dc_exp1 = X_matrix_dc_exp1 - X_matrix_tr_dc_exp1

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1#/self.lengthscale_11
        #K_dada_exp1 = self.variance_11*(1.0+sqrt5*diff_X_da_exp1+(2/3)*tf.math.square(diff_X_da_exp1))*tf.math.exp(-sqrt5*diff_X_da_exp1)
        K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)
        #K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)
        #K_dcdc_exp1 = self.variance_13*tf.exp(-0.5*(diff_X_dc_exp1/self.lengthscale_13)**2)

        # Block for experiment 2

        X_matrix_da_exp2 = tf.tile(tf.reshape(X[n:2*n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp2 = tf.tile(tf.reshape(X2[m:2*m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp2 = tf.transpose(X_matrix_tr_da_exp2)

        X_matrix_db_exp2 = tf.tile(tf.reshape(X[n:2*n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp2 = tf.tile(tf.reshape(X2[m:2*m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp2 = tf.transpose(X_matrix_tr_db_exp2)

        #X_matrix_dc_exp2 = tf.tile(tf.reshape(X[n:2*n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp2 = tf.tile(tf.reshape(X2[m:2*m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp2 = tf.transpose(X_matrix_tr_dc_exp2)

        #diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2
        #diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2
        #diff_X_dc_exp2 = X_matrix_dc_exp2 - X_matrix_tr_dc_exp2

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)
        #K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)

        diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2#/self.lengthscale_11
        #K_dada_exp2 = self.variance_11*(1.0+sqrt5*diff_X_da_exp2+(2/3)*tf.math.square(diff_X_da_exp2))*tf.math.exp(-sqrt5*diff_X_da_exp2)
        K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)

        diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2#/self.lengthscale_22
        #K_dbdb_exp2 = self.variance_22*(1.0+sqrt5*diff_X_db_exp2+(2/3)*tf.math.square(diff_X_db_exp2))*tf.math.exp(-sqrt5*diff_X_db_exp2)
        K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)
        #K_dcdc_exp2 = self.variance_23*tf.exp(-0.5*(diff_X_dc_exp2/self.lengthscale_23)**2)

        K_xx = K_dada_exp1 #+ K_dbdb_exp1 + K_dcdc_exp1
        K_ff = K_dada_exp2 + K_dbdb_exp2 #+ K_dcdc_exp2

        # Combine four blocks together

        K_xf = np.zeros((n,m))
        K_fx = np.zeros((n,m))

        #K_xf =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp1))))
        #K_fx =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp2))))

        #diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        #print('K_xf',K_xf)

        K_upper_row = tf.concat([K_xx, K_xf], axis=1)
        K_lower_row = tf.concat([K_fx, K_ff], axis=1)
        K = tf.concat([K_upper_row, K_lower_row], axis=0)
        #return(K_dada * K_dbdb + K_dbda * K_dadb)
        return K

    #@params_as_tensors
    def K_diag(self, X):
        n = int(X.shape[0]/2)
        diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_db_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_12)), (-1,))
        #diag_dc_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_13)), (-1,))

        diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11+self.variance_22)), (-1,))
        #diag_db_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_22)), (-1,))
        #diag_dc_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_23)), (-1,))

        block_upper = diag_da_exp1 #+ diag_db_exp1 + diag_dc_exp1
        block_lower = diag_da_exp2# + diag_db_exp2 #+ diag_dc_exp2

        diag = tf.reshape(tf.stack([block_upper, block_lower]), (-1,))
        return diag


class K_full_model_3(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_11 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_11 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        self.variance_12 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_12 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_13 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_13 = gpflow.Parameter(0.05, transform=positive())

        #self.variance_22 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_22 = gpflow.Parameter(100.0, transform=positive())

        #self.variance_23 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_23 = gpflow.Parameter(10.0, transform=positive())

    #@params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        #n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        #m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #half_shape=tf.cast(tf.shape(X)[0]/2, dtype=tf.int32)
        #half_shape = np.asscalar(np.array(tf.shape(X)[0])/2).astype(int)
        #print('half_shape',half_shape)

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asarray(tf.cast(tf.shape(X)[0]/2, dtype=tf.int32))
        #m = np.asarray(tf.cast(tf.shape(X2)[0]/2, dtype=tf.int32))

        n = int(X.shape[0]/2)
        m = int(X2.shape[0]/2)

        sqrt5 = np.sqrt(5.0)

        # Block for experiment 1
        X_matrix_da_exp1 = tf.tile(tf.reshape(X[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp1 = tf.tile(tf.reshape(X2[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp1 = tf.transpose(X_matrix_tr_da_exp1)

        X_matrix_db_exp1 = tf.tile(tf.reshape(X[0:n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp1 = tf.tile(tf.reshape(X2[0:m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp1 = tf.transpose(X_matrix_tr_db_exp1)

        #X_matrix_dc_exp1 = tf.tile(tf.reshape(X[0:n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp1 = tf.tile(tf.reshape(X2[0:m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp1 = tf.transpose(X_matrix_tr_dc_exp1)

        #diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1
        #diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1
        #diff_X_dc_exp1 = X_matrix_dc_exp1 - X_matrix_tr_dc_exp1

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)
        #K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)

        diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1#/self.lengthscale_11
        #K_dada_exp1 = self.variance_11*(1.0+sqrt5*diff_X_da_exp1+(2/3)*tf.math.square(diff_X_da_exp1))*tf.math.exp(-sqrt5*diff_X_da_exp1)
        K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)

        diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1#/self.lengthscale_12
        #K_dbdb_exp1 = self.variance_12*(1.0+sqrt5*diff_X_db_exp1+(2/3)*tf.math.square(diff_X_db_exp1))*tf.math.exp(-sqrt5*diff_X_db_exp1)
        K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)

        #K_dcdc_exp1 = self.variance_13*tf.exp(-0.5*(diff_X_dc_exp1/self.lengthscale_13)**2)

        # Block for experiment 2

        X_matrix_da_exp2 = tf.tile(tf.reshape(X[n:2*n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp2 = tf.tile(tf.reshape(X2[m:2*m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp2 = tf.transpose(X_matrix_tr_da_exp2)

        #X_matrix_db_exp2 = tf.tile(tf.reshape(X[n:2*n,1],(n,1)), (1,m))
        #X_matrix_tr_db_exp2 = tf.tile(tf.reshape(X2[m:2*m,1],(m,1)), (1,n))
        #X_matrix_tr_db_exp2 = tf.transpose(X_matrix_tr_db_exp2)

        #X_matrix_dc_exp2 = tf.tile(tf.reshape(X[n:2*n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp2 = tf.tile(tf.reshape(X2[m:2*m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp2 = tf.transpose(X_matrix_tr_dc_exp2)

        #diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2
        #diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2
        #diff_X_dc_exp2 = X_matrix_dc_exp2 - X_matrix_tr_dc_exp2

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)

        diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2#/self.lengthscale_11
        #K_dada_exp2 = self.variance_11*(1.0+sqrt5*diff_X_da_exp2+(2/3)*tf.math.square(diff_X_da_exp2))*tf.math.exp(-sqrt5*diff_X_da_exp2)
        K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)

        #K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)
        #K_dcdc_exp2 = self.variance_23*tf.exp(-0.5*(diff_X_dc_exp2/self.lengthscale_23)**2)

        K_xx = K_dada_exp1 + K_dbdb_exp1 #+ K_dcdc_exp1
        K_ff = K_dada_exp2 #+ K_dbdb_exp2 + K_dcdc_exp2

        # Combine four blocks together

        K_xf = np.zeros((n,m))
        K_fx = np.zeros((n,m))

        #K_xf =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp1))))
        #K_fx =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp2))))

        #diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        #print('K_xf',K_xf)

        K_upper_row = tf.concat([K_xx, K_xf], axis=1)
        K_lower_row = tf.concat([K_fx, K_ff], axis=1)
        K = tf.concat([K_upper_row, K_lower_row], axis=0)
        #return(K_dada * K_dbdb + K_dbda * K_dadb)
        return K

    #@params_as_tensors
    def K_diag(self, X):
        n = int(X.shape[0]/2)
        diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_db_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_12)), (-1,))
        #diag_dc_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_13)), (-1,))

        diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_db_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_22)), (-1,))
        #diag_dc_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_23)), (-1,))

        block_upper = diag_da_exp1 + diag_db_exp1 #+ diag_dc_exp1
        block_lower = diag_da_exp2 #+ diag_db_exp2 + diag_dc_exp2

        diag = tf.reshape(tf.stack([block_upper, block_lower]), (-1,))
        return diag


class K_full_model_4(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_11 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_11 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        self.variance_12 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_12 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_13 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_13 = gpflow.Parameter(0.05, transform=positive())

        self.variance_22 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_22 = gpflow.Parameter(50.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_23 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_23 = gpflow.Parameter(10.0, transform=positive())

    #@params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        #n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        #m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #half_shape=tf.cast(tf.shape(X)[0]/2, dtype=tf.int32)
        #half_shape = np.asscalar(np.array(tf.shape(X)[0])/2).astype(int)
        #print('half_shape',half_shape)

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asarray(tf.cast(tf.shape(X)[0]/2, dtype=tf.int32))
        #m = np.asarray(tf.cast(tf.shape(X2)[0]/2, dtype=tf.int32))

        n = int(X.shape[0]/2)
        m = int(X2.shape[0]/2)

        sqrt5 = np.sqrt(5.0)

        # Block for experiment 1
        X_matrix_da_exp1 = tf.tile(tf.reshape(X[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp1 = tf.tile(tf.reshape(X2[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp1 = tf.transpose(X_matrix_tr_da_exp1)

        X_matrix_db_exp1 = tf.tile(tf.reshape(X[0:n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp1 = tf.tile(tf.reshape(X2[0:m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp1 = tf.transpose(X_matrix_tr_db_exp1)

        #X_matrix_dc_exp1 = tf.tile(tf.reshape(X[0:n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp1 = tf.tile(tf.reshape(X2[0:m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp1 = tf.transpose(X_matrix_tr_dc_exp1)

        #diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1
        #diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1
        #diff_X_dc_exp1 = X_matrix_dc_exp1 - X_matrix_tr_dc_exp1

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)
        #K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)

        diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1#/self.lengthscale_11
        #K_dada_exp1 = self.variance_11*(1.0+sqrt5*diff_X_da_exp1+(2/3)*tf.math.square(diff_X_da_exp1))*tf.math.exp(-sqrt5*diff_X_da_exp1)
        K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)

        diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1#/self.lengthscale_12
        #K_dbdb_exp1 = self.variance_12*(1.0+sqrt5*diff_X_db_exp1+(2/3)*tf.math.square(diff_X_db_exp1))*tf.math.exp(-sqrt5*diff_X_db_exp1)
        K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)
        #K_dcdc_exp1 = self.variance_13*tf.exp(-0.5*(diff_X_dc_exp1/self.lengthscale_13)**2)

        # Block for experiment 2

        X_matrix_da_exp2 = tf.tile(tf.reshape(X[n:2*n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp2 = tf.tile(tf.reshape(X2[m:2*m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp2 = tf.transpose(X_matrix_tr_da_exp2)

        X_matrix_db_exp2 = tf.tile(tf.reshape(X[n:2*n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp2 = tf.tile(tf.reshape(X2[m:2*m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp2 = tf.transpose(X_matrix_tr_db_exp2)

        #X_matrix_dc_exp2 = tf.tile(tf.reshape(X[n:2*n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp2 = tf.tile(tf.reshape(X2[m:2*m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp2 = tf.transpose(X_matrix_tr_dc_exp2)

        #diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2
        #diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2
        #diff_X_dc_exp2 = X_matrix_dc_exp2 - X_matrix_tr_dc_exp2

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)
        #K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)

        diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2#/self.lengthscale_11
        #K_dada_exp2 = self.variance_11*(1.0+sqrt5*diff_X_da_exp2+(2/3)*tf.math.square(diff_X_da_exp2))*tf.math.exp(-sqrt5*diff_X_da_exp2)
        K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)

        diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2#/self.lengthscale_22
        #K_dbdb_exp2 = self.variance_22*(1.0+sqrt5*diff_X_db_exp2+(2/3)*tf.math.square(diff_X_db_exp2))*tf.math.exp(-sqrt5*diff_X_db_exp2)
        K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)
        #K_dcdc_exp2 = self.variance_23*tf.exp(-0.5*(diff_X_dc_exp2/self.lengthscale_23)**2)

        K_xx = K_dada_exp1 + K_dbdb_exp1 #+ K_dcdc_exp1
        K_ff = K_dada_exp2 + K_dbdb_exp2 #+ K_dcdc_exp2

        # Combine four blocks together

        K_xf = np.zeros((n,m))
        K_fx = np.zeros((n,m))

        #K_xf =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp1))))
        #K_fx =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp2))))

        #diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        #print('K_xf',K_xf)

        K_upper_row = tf.concat([K_xx, K_xf], axis=1)
        K_lower_row = tf.concat([K_fx, K_ff], axis=1)
        K = tf.concat([K_upper_row, K_lower_row], axis=0)
        #return(K_dada * K_dbdb + K_dbda * K_dadb)
        return K

    #@params_as_tensors
    def K_diag(self, X):
        n = int(X.shape[0]/2)
        diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_db_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_12)), (-1,))
        #diag_dc_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_13)), (-1,))

        diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_db_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_22)), (-1,))
        #diag_dc_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_23)), (-1,))

        block_upper = diag_da_exp1 + diag_db_exp1 #+ diag_dc_exp1
        block_lower = diag_da_exp2 + diag_db_exp2 #+ diag_dc_exp2

        diag = tf.reshape(tf.stack([block_upper, block_lower]), (-1,))
        return diag

class K_full_model_5(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_11 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_11 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        self.variance_12 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_12 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_13 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_13 = gpflow.Parameter(0.05, transform=positive())

        #self.variance_22 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_22 = gpflow.Parameter(100.0, transform=positive())

        #self.variance_23 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_23 = gpflow.Parameter(10.0, transform=positive())

    #@params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        #n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        #m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #half_shape=tf.cast(tf.shape(X)[0]/2, dtype=tf.int32)
        #half_shape = np.asscalar(np.array(tf.shape(X)[0])/2).astype(int)
        #print('half_shape',half_shape)

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asarray(tf.cast(tf.shape(X)[0]/2, dtype=tf.int32))
        #m = np.asarray(tf.cast(tf.shape(X2)[0]/2, dtype=tf.int32))

        n = int(X.shape[0]/2)
        m = int(X2.shape[0]/2)

        sqrt5 = np.sqrt(5.0)

        # Block for experiment 1
        X_matrix_da_exp1 = tf.tile(tf.reshape(X[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp1 = tf.tile(tf.reshape(X2[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp1 = tf.transpose(X_matrix_tr_da_exp1)

        X_matrix_db_exp1 = tf.tile(tf.reshape(X[0:n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp1 = tf.tile(tf.reshape(X2[0:m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp1 = tf.transpose(X_matrix_tr_db_exp1)

        #X_matrix_dc_exp1 = tf.tile(tf.reshape(X[0:n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp1 = tf.tile(tf.reshape(X2[0:m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp1 = tf.transpose(X_matrix_tr_dc_exp1)

        #diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1
        #diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1
        #diff_X_dc_exp1 = X_matrix_dc_exp1 - X_matrix_tr_dc_exp1

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)
        #K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)

        diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1#/self.lengthscale_11
        #K_dada_exp1 = self.variance_11*(1.0+sqrt5*diff_X_da_exp1+(2/3)*tf.math.square(diff_X_da_exp1))*tf.math.exp(-sqrt5*diff_X_da_exp1)
        K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)

        diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1#/self.lengthscale_12
        #K_dbdb_exp1 = self.variance_12*(1.0+sqrt5*diff_X_db_exp1+(2/3)*tf.math.square(diff_X_db_exp1))*tf.math.exp(-sqrt5*diff_X_db_exp1)
        K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)
        #K_dcdc_exp1 = self.variance_13*tf.exp(-0.5*(diff_X_dc_exp1/self.lengthscale_13)**2)
        #K_dcdc_exp1 = self.variance_13*tf.exp(-0.5*(diff_X_dc_exp1/self.lengthscale_13)**2)

        # Block for experiment 2

        X_matrix_da_exp2 = tf.tile(tf.reshape(X[n:2*n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp2 = tf.tile(tf.reshape(X2[m:2*m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp2 = tf.transpose(X_matrix_tr_da_exp2)

        #X_matrix_db_exp2 = tf.tile(tf.reshape(X[n:2*n,1],(n,1)), (1,m))
        #X_matrix_tr_db_exp2 = tf.tile(tf.reshape(X2[m:2*m,1],(m,1)), (1,n))
        #X_matrix_tr_db_exp2 = tf.transpose(X_matrix_tr_db_exp2)

        #X_matrix_dc_exp2 = tf.tile(tf.reshape(X[n:2*n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp2 = tf.tile(tf.reshape(X2[m:2*m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp2 = tf.transpose(X_matrix_tr_dc_exp2)

        #diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2
        #diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2
        #diff_X_dc_exp2 = X_matrix_dc_exp2 - X_matrix_tr_dc_exp2

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)

        diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2#/self.lengthscale_11
        #K_dada_exp2 = self.variance_11*(1.0+sqrt5*diff_X_da_exp2+(2/3)*tf.math.square(diff_X_da_exp2))*tf.math.exp(-sqrt5*diff_X_da_exp2)
        K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)
        #K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)
        #K_dcdc_exp2 = self.variance_23*tf.exp(-0.5*(diff_X_dc_exp2/self.lengthscale_23)**2)

        K_xx = K_dada_exp1 + K_dbdb_exp1 + K_dada_exp1 * K_dbdb_exp1
        K_ff = K_dada_exp2 #+ K_dbdb_exp2 + K_dcdc_exp2

        # Combine four blocks together

        K_xf = np.zeros((n,m))
        K_fx = np.zeros((n,m))

        #K_xf =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp1))))
        #K_fx =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp2))))

        #diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        #print('K_xf',K_xf)

        K_upper_row = tf.concat([K_xx, K_xf], axis=1)
        K_lower_row = tf.concat([K_fx, K_ff], axis=1)
        K = tf.concat([K_upper_row, K_lower_row], axis=0)
        #return(K_dada * K_dbdb + K_dbda * K_dadb)
        return K

    #@params_as_tensors
    def K_diag(self, X):
        n = int(X.shape[0]/2)
        diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_db_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_12)), (-1,))
        diag_dc_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11*self.variance_12)), (-1,))

        diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_db_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_22)), (-1,))
        #diag_dc_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_23)), (-1,))

        block_upper = diag_da_exp1 + diag_db_exp1 + diag_dc_exp1
        block_lower = diag_da_exp2 #+ diag_db_exp2 + diag_dc_exp2

        diag = tf.reshape(tf.stack([block_upper, block_lower]), (-1,))
        return diag

class K_full_model_6(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_11 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_11 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        self.variance_12 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_12 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_13 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_13 = gpflow.Parameter(0.05, transform=positive())

        #self.variance_22 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_22 = gpflow.Parameter(100.0, transform=positive())

        #self.variance_23 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_23 = gpflow.Parameter(10.0, transform=positive())

    #@params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        #n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        #m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #half_shape=tf.cast(tf.shape(X)[0]/2, dtype=tf.int32)
        #half_shape = np.asscalar(np.array(tf.shape(X)[0])/2).astype(int)
        #print('half_shape',half_shape)

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asarray(tf.cast(tf.shape(X)[0]/2, dtype=tf.int32))
        #m = np.asarray(tf.cast(tf.shape(X2)[0]/2, dtype=tf.int32))

        n = int(X.shape[0]/2)
        m = int(X2.shape[0]/2)

        sqrt5 = np.sqrt(5.0)

        # Block for experiment 1
        X_matrix_da_exp1 = tf.tile(tf.reshape(X[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp1 = tf.tile(tf.reshape(X2[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp1 = tf.transpose(X_matrix_tr_da_exp1)

        X_matrix_db_exp1 = tf.tile(tf.reshape(X[0:n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp1 = tf.tile(tf.reshape(X2[0:m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp1 = tf.transpose(X_matrix_tr_db_exp1)

        #X_matrix_dc_exp1 = tf.tile(tf.reshape(X[0:n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp1 = tf.tile(tf.reshape(X2[0:m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp1 = tf.transpose(X_matrix_tr_dc_exp1)

        #diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1
        #diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1
        #diff_X_dc_exp1 = X_matrix_dc_exp1 - X_matrix_tr_dc_exp1

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)
        #K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)

        diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1#/self.lengthscale_11
        #K_dada_exp1 = self.variance_11*(1.0+sqrt5*diff_X_da_exp1+(2/3)*tf.math.square(diff_X_da_exp1))*tf.math.exp(-sqrt5*diff_X_da_exp1)
        K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)

        diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1#/self.lengthscale_12
        #K_dbdb_exp1 = self.variance_12*(1.0+sqrt5*diff_X_db_exp1+(2/3)*tf.math.square(diff_X_db_exp1))*tf.math.exp(-sqrt5*diff_X_db_exp1)
        K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)
        #K_dcdc_exp1 = self.variance_13*tf.exp(-0.5*(diff_X_dc_exp1/self.lengthscale_13)**2)

        # Block for experiment 2

        X_matrix_da_exp2 = tf.tile(tf.reshape(X[n:2*n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp2 = tf.tile(tf.reshape(X2[m:2*m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp2 = tf.transpose(X_matrix_tr_da_exp2)

        #X_matrix_db_exp2 = tf.tile(tf.reshape(X[n:2*n,1],(n,1)), (1,m))
        #X_matrix_tr_db_exp2 = tf.tile(tf.reshape(X2[m:2*m,1],(m,1)), (1,n))
        #X_matrix_tr_db_exp2 = tf.transpose(X_matrix_tr_db_exp2)

        #X_matrix_dc_exp2 = tf.tile(tf.reshape(X[n:2*n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp2 = tf.tile(tf.reshape(X2[m:2*m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp2 = tf.transpose(X_matrix_tr_dc_exp2)

        #diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2
        #diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2
        #diff_X_dc_exp2 = X_matrix_dc_exp2 - X_matrix_tr_dc_exp2

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)

        diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2#/self.lengthscale_11
        #K_dada_exp2 = self.variance_11*(1.0+sqrt5*diff_X_da_exp2+(2/3)*tf.math.square(diff_X_da_exp2))*tf.math.exp(-sqrt5*diff_X_da_exp2)
        K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)
        #K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)
        #K_dcdc_exp2 = self.variance_23*tf.exp(-0.5*(diff_X_dc_exp2/self.lengthscale_23)**2)

        #K_xx = K_dada_exp1 + K_dbdb_exp1 + K_dcdc_exp1
        #K_ff = K_dada_exp2 + K_dbdb_exp2 + K_dcdc_exp2

        K_xx = K_dada_exp1 + K_dada_exp1*K_dbdb_exp1
        K_ff = K_dada_exp2 #+ K_dbdb_exp2 + K_dcdc_exp2

        # Combine four blocks together

        K_xf = np.zeros((n,m))
        K_fx = np.zeros((n,m))

        #K_xf =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp1))))
        #K_fx =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp2))))

        #diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        #print('K_xf',K_xf)

        K_upper_row = tf.concat([K_xx, K_xf], axis=1)
        K_lower_row = tf.concat([K_fx, K_ff], axis=1)
        K = tf.concat([K_upper_row, K_lower_row], axis=0)
        #return(K_dada * K_dbdb + K_dbda * K_dadb)
        return K

    #@params_as_tensors
    def K_diag(self, X):
        n = int(X.shape[0]/2)
        diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_db_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_12)), (-1,))
        diag_dc_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11*self.variance_12)), (-1,))

        diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_db_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_22)), (-1,))
        #diag_dc_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_23)), (-1,))

        #block_upper = diag_da_exp1 + diag_db_exp1 + diag_dc_exp1
        #block_lower = diag_da_exp2 + diag_db_exp2 + diag_dc_exp2

        block_upper = diag_da_exp1 + diag_dc_exp1
        block_lower = diag_da_exp2 #+ diag_db_exp2 + diag_dc_exp2

        diag = tf.reshape(tf.stack([block_upper, block_lower]), (-1,))
        return diag

class K_full_model_7(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_11 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_11 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_12 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_12 = gpflow.Parameter(1.0, transform=positive())

        #self.variance_13 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_13 = gpflow.Parameter(0.05, transform=positive())

        self.variance_22 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_22 = gpflow.Parameter(50.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_23 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_23 = gpflow.Parameter(10.0, transform=positive())

    #@params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        #n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        #m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #half_shape=tf.cast(tf.shape(X)[0]/2, dtype=tf.int32)
        #half_shape = np.asscalar(np.array(tf.shape(X)[0])/2).astype(int)
        #print('half_shape',half_shape)

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asarray(tf.cast(tf.shape(X)[0]/2, dtype=tf.int32))
        #m = np.asarray(tf.cast(tf.shape(X2)[0]/2, dtype=tf.int32))

        n = int(X.shape[0]/2)
        m = int(X2.shape[0]/2)

        sqrt5 = np.sqrt(5.0)

        # Block for experiment 1
        X_matrix_da_exp1 = tf.tile(tf.reshape(X[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp1 = tf.tile(tf.reshape(X2[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp1 = tf.transpose(X_matrix_tr_da_exp1)

        X_matrix_db_exp1 = tf.tile(tf.reshape(X[0:n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp1 = tf.tile(tf.reshape(X2[0:m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp1 = tf.transpose(X_matrix_tr_db_exp1)

        #X_matrix_dc_exp1 = tf.tile(tf.reshape(X[0:n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp1 = tf.tile(tf.reshape(X2[0:m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp1 = tf.transpose(X_matrix_tr_dc_exp1)

        #diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1
        #diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1
        #diff_X_dc_exp1 = X_matrix_dc_exp1 - X_matrix_tr_dc_exp1

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)

        diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1#/self.lengthscale_11
        #K_dada_exp1 = self.variance_11*(1.0+sqrt5*diff_X_da_exp1+(2/3)*tf.math.square(diff_X_da_exp1))*tf.math.exp(-sqrt5*diff_X_da_exp1)
        K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)

        #diff_X_db_exp1 = tf.math.abs(X_matrix_db_exp1 - X_matrix_tr_db_exp1)/self.lengthscale_12
        #K_dbdb_exp1 = self.variance_12*(1.0+sqrt5*diff_X_db_exp1+(2/3)*tf.math.square(diff_X_db_exp1))*tf.math.exp(-sqrt5*diff_X_db_exp1)
        #K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)
        #K_dcdc_exp1 = self.variance_13*tf.exp(-0.5*(diff_X_dc_exp1/self.lengthscale_13)**2)

        # Block for experiment 2

        X_matrix_da_exp2 = tf.tile(tf.reshape(X[n:2*n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp2 = tf.tile(tf.reshape(X2[m:2*m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp2 = tf.transpose(X_matrix_tr_da_exp2)

        X_matrix_db_exp2 = tf.tile(tf.reshape(X[n:2*n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp2 = tf.tile(tf.reshape(X2[m:2*m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp2 = tf.transpose(X_matrix_tr_db_exp2)

        #X_matrix_dc_exp2 = tf.tile(tf.reshape(X[n:2*n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp2 = tf.tile(tf.reshape(X2[m:2*m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp2 = tf.transpose(X_matrix_tr_dc_exp2)

        #diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2
        #diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2
        #diff_X_dc_exp2 = X_matrix_dc_exp2 - X_matrix_tr_dc_exp2

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)
        #K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)

        diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2#/self.lengthscale_11
        #K_dada_exp2 = self.variance_11*(1.0+sqrt5*diff_X_da_exp2+(2/3)*tf.math.square(diff_X_da_exp2))*tf.math.exp(-sqrt5*diff_X_da_exp2)
        K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)

        diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2#/self.lengthscale_22
        #K_dbdb_exp2 = self.variance_22*(1.0+sqrt5*diff_X_db_exp2+(2/3)*tf.math.square(diff_X_db_exp2))*tf.math.exp(-sqrt5*diff_X_db_exp2)
        K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)
        #K_dcdc_exp2 = self.variance_23*tf.exp(-0.5*(diff_X_dc_exp2/self.lengthscale_23)**2)

        #K_xx = K_dada_exp1 #+ K_dbdb_exp1 + K_dcdc_exp1
        #K_ff = K_dada_exp2 + K_dbdb_exp2 + K_dcdc_exp2

        K_xx = K_dada_exp1 #+ K_dbdb_exp1 + K_dcdc_exp1
        K_ff = K_dbdb_exp2 + K_dada_exp2*K_dbdb_exp2

        # Combine four blocks together

        K_xf = np.zeros((n,m))
        K_fx = np.zeros((n,m))

        #K_xf =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp1))))
        #K_fx =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp2))))

        #diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        #print('K_xf',K_xf)

        K_upper_row = tf.concat([K_xx, K_xf], axis=1)
        K_lower_row = tf.concat([K_fx, K_ff], axis=1)
        K = tf.concat([K_upper_row, K_lower_row], axis=0)
        #return(K_dada * K_dbdb + K_dbda * K_dadb)
        return K

    #@params_as_tensors
    def K_diag(self, X):
        n = int(X.shape[0]/2)
        diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_db_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_12)), (-1,))
        #diag_dc_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_13)), (-1,))

        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_db_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_22)), (-1,))
        diag_dc_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11*self.variance_22)), (-1,))

        #block_upper = diag_da_exp1 #+ diag_db_exp1 + diag_dc_exp1
        #block_lower = diag_da_exp2 + diag_db_exp2 + diag_dc_exp2

        block_upper = diag_da_exp1 #+ diag_db_exp1 + diag_dc_exp1
        block_lower = diag_db_exp2 + diag_dc_exp2

        diag = tf.reshape(tf.stack([block_upper, block_lower]), (-1,))
        return diag

class K_full_model_8(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_11 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_11 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_12 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_12 = gpflow.Parameter(1.0, transform=positive())

        #self.variance_13 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_13 = gpflow.Parameter(0.05, transform=positive())

        self.variance_22 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_22 = gpflow.Parameter(50.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_23 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_23 = gpflow.Parameter(10.0, transform=positive())

    #@params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        n = int(X.shape[0]/2)
        m = int(X2.shape[0]/2)

        sqrt5 = np.sqrt(5.0)

        # Block for experiment 1
        X_matrix_da_exp1 = tf.tile(tf.reshape(X[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp1 = tf.tile(tf.reshape(X2[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp1 = tf.transpose(X_matrix_tr_da_exp1)

        diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1#/self.lengthscale_11

        #K_dada_exp1 = self.variance_11*(1.0+sqrt5*diff_X_da_exp1+(2/3)*tf.math.square(diff_X_da_exp1))*tf.math.exp(-sqrt5*diff_X_da_exp1)
        K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)
        # Block for experiment 2

        X_matrix_da_exp2 = tf.tile(tf.reshape(X[n:2*n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp2 = tf.tile(tf.reshape(X2[m:2*m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp2 = tf.transpose(X_matrix_tr_da_exp2)

        X_matrix_dc_exp2 = tf.tile(tf.reshape(X[n:2*n,1],(n,1)), (1,m))
        X_matrix_tr_dc_exp2 = tf.tile(tf.reshape(X2[m:2*m,1],(m,1)), (1,n))
        X_matrix_tr_dc_exp2 = tf.transpose(X_matrix_tr_dc_exp2)

        #X_matrix_dc_exp2 = tf.tile(tf.reshape(X[n:2*n,2],(n,1)), (1,m))

        diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2#/self.lengthscale_11
        #K_dada_exp2 = self.variance_11*(1.0+sqrt5*diff_X_da_exp2+(2/3)*tf.math.square(diff_X_da_exp2))*tf.math.exp(-sqrt5*diff_X_da_exp2)
        K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)

        diff_X_dc_exp2 = X_matrix_dc_exp2 - X_matrix_tr_dc_exp2#/self.lengthscale_22
        #K_dcdc_exp2 = self.variance_22*(1.0+sqrt5*diff_X_dc_exp2+(2/3)*tf.math.square(diff_X_dc_exp2))*tf.math.exp(-sqrt5*diff_X_dc_exp2)
        K_dcdc_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_dc_exp2/self.lengthscale_22)**2)

        K_xx = K_dada_exp1 #+ K_dbdb_exp1 + K_dcdc_exp1
        K_ff = K_dada_exp2 + K_dcdc_exp2 + K_dada_exp2 * K_dcdc_exp2

        # Combine four blocks together

        K_xf = np.zeros((n,m))
        K_fx = np.zeros((n,m))

        #K_xf =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp1))))
        #K_fx =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp2))))

        #diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        #print('K_xf',K_xf)

        K_upper_row = tf.concat([K_xx, K_xf], axis=1)
        K_lower_row = tf.concat([K_fx, K_ff], axis=1)
        K = tf.concat([K_upper_row, K_lower_row], axis=0)
        #return(K_dada * K_dbdb + K_dbda * K_dadb)
        return K

    #@params_as_tensors
    def K_diag(self, X):
        n = int(X.shape[0]/2)
        diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_db_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_12)), (-1,))
        #diag_dc_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_13)), (-1,))

        diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_db_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_22)), (-1,))
        diag_dc_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11*self.variance_22)), (-1,))

        block_upper = diag_da_exp1 #+ diag_db_exp1 + diag_dc_exp1
        block_lower = diag_da_exp2 + diag_db_exp2 + diag_dc_exp2

        diag = tf.reshape(tf.stack([block_upper, block_lower]), (-1,))
        return diag

class K_full_model_9(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_11 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_11 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.001),
            gpflow.utilities.to_default_float(1000.0)))

        self.variance_12 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_12 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_13 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_13 = gpflow.Parameter(0.05, transform=positive())

        self.variance_22 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_22 = gpflow.Parameter(50.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_23 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_23 = gpflow.Parameter(10.0, transform=positive())

    #@params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        #n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        #m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #half_shape=tf.cast(tf.shape(X)[0]/2, dtype=tf.int32)
        #half_shape = np.asscalar(np.array(tf.shape(X)[0])/2).astype(int)
        #print('half_shape',half_shape)

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asarray(tf.cast(tf.shape(X)[0]/2, dtype=tf.int32))
        #m = np.asarray(tf.cast(tf.shape(X2)[0]/2, dtype=tf.int32))

        n = int(X.shape[0]/2)
        m = int(X2.shape[0]/2)

        sqrt5 = np.sqrt(5.0)

        # Block for experiment 1
        X_matrix_da_exp1 = tf.tile(tf.reshape(X[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp1 = tf.tile(tf.reshape(X2[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp1 = tf.transpose(X_matrix_tr_da_exp1)

        X_matrix_db_exp1 = tf.tile(tf.reshape(X[0:n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp1 = tf.tile(tf.reshape(X2[0:m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp1 = tf.transpose(X_matrix_tr_db_exp1)

        # X_matrix_dc_exp1 = tf.tile(tf.reshape(X[0:n,2],(n,1)), (1,m))
        # X_matrix_tr_dc_exp1 = tf.tile(tf.reshape(X2[0:m,2],(m,1)), (1,n))
        # X_matrix_tr_dc_exp1 = tf.transpose(X_matrix_tr_dc_exp1)

        #diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1
        #diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1
        #diff_X_dc_exp1 = X_matrix_dc_exp1 - X_matrix_tr_dc_exp1

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)
        #K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)

        diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1#/self.lengthscale_11
        #K_dada_exp1 = self.variance_11*(1.0+sqrt5*diff_X_da_exp1+(2/3)*tf.math.square(diff_X_da_exp1))*tf.math.exp(-sqrt5*diff_X_da_exp1)
        K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)

        diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1#/self.lengthscale_12
        K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)
        #K_dbdb_exp1 = self.variance_12*(1.0+sqrt5*diff_X_db_exp1+(2/3)*tf.math.square(diff_X_db_exp1))*tf.math.exp(-sqrt5*diff_X_db_exp1)
        #K_dcdc_exp1 = self.variance_13*tf.exp(-0.5*(diff_X_dc_exp1/self.lengthscale_13)**2)

        # Block for experiment 2

        X_matrix_da_exp2 = tf.tile(tf.reshape(X[n:2*n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp2 = tf.tile(tf.reshape(X2[m:2*m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp2 = tf.transpose(X_matrix_tr_da_exp2)

        X_matrix_db_exp2 = tf.tile(tf.reshape(X[n:2*n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp2 = tf.tile(tf.reshape(X2[m:2*m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp2 = tf.transpose(X_matrix_tr_db_exp2)

        #X_matrix_dc_exp2 = tf.tile(tf.reshape(X[n:2*n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp2 = tf.tile(tf.reshape(X2[m:2*m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp2 = tf.transpose(X_matrix_tr_dc_exp2)

        #diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2
        #diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2
        #diff_X_dc_exp2 = X_matrix_dc_exp2 - X_matrix_tr_dc_exp2

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)
        #K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)

        diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2#/self.lengthscale_11
        #K_dada_exp2 = self.variance_11*(1.0+sqrt5*diff_X_da_exp2+(2/3)*tf.math.square(diff_X_da_exp2))*tf.math.exp(-sqrt5*diff_X_da_exp2)
        K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)

        diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2#/self.lengthscale_22
        #K_dbdb_exp2 = self.variance_22*(1.0+sqrt5*diff_X_db_exp2+(2/3)*tf.math.square(diff_X_db_exp2))*tf.math.exp(-sqrt5*diff_X_db_exp2)
        K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)
        #K_dcdc_exp2 = self.variance_23*tf.exp(-0.5*(diff_X_dc_exp2/self.lengthscale_23)**2)

        K_xx = K_dada_exp1 + K_dbdb_exp1 + K_dada_exp1 * K_dbdb_exp1
        K_ff = K_dada_exp2 + K_dbdb_exp2 + K_dada_exp2 * K_dbdb_exp2

        # Combine four blocks together

        K_xf = np.zeros((n,m))
        K_fx = np.zeros((n,m))

        #K_xf =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp1))))
        #K_fx =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp2))))

        #diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        #print('K_xf',K_xf)

        K_upper_row = tf.concat([K_xx, K_xf], axis=1)
        K_lower_row = tf.concat([K_fx, K_ff], axis=1)
        K = tf.concat([K_upper_row, K_lower_row], axis=0)
        #return(K_dada * K_dbdb + K_dbda * K_dadb)
        return K

    #@params_as_tensors
    def K_diag(self, X):
        #n = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        n = int(X.shape[0]/2)
        diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_db_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_12)), (-1,))
        diag_dc_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11* self.variance_12)), (-1,))

        diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_db_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_22)), (-1,))
        diag_dc_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11 * self.variance_22)), (-1,))

        block_upper = diag_da_exp1 + diag_db_exp1 + diag_dc_exp1
        block_lower = diag_da_exp2 + diag_db_exp2 + diag_dc_exp2

        diag = tf.reshape(tf.stack([block_upper, block_lower]), (-1,))
        return diag

class K_full_model_10(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_11 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_11 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        self.variance_12 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_12 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_13 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_13 = gpflow.Parameter(0.05, transform=positive())

        self.variance_22 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_22 = gpflow.Parameter(50.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_23 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_23 = gpflow.Parameter(10.0, transform=positive())

    #@params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        #n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        #m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #half_shape=tf.cast(tf.shape(X)[0]/2, dtype=tf.int32)
        #half_shape = np.asscalar(np.array(tf.shape(X)[0])/2).astype(int)
        #print('half_shape',half_shape)

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asscalar(np.array(X.shape[0]/2).astype(int))
        #m = np.asscalar(np.array(X2.shape[0]/2).astype(int))

        #n = np.asarray(tf.cast(tf.shape(X)[0]/2, dtype=tf.int32))
        #m = np.asarray(tf.cast(tf.shape(X2)[0]/2, dtype=tf.int32))

        n = int(X.shape[0]/2)
        m = int(X2.shape[0]/2)

        sqrt5 = np.sqrt(5.0)

        # Block for experiment 1
        X_matrix_da_exp1 = tf.tile(tf.reshape(X[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp1 = tf.tile(tf.reshape(X2[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp1 = tf.transpose(X_matrix_tr_da_exp1)

        X_matrix_db_exp1 = tf.tile(tf.reshape(X[0:n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp1 = tf.tile(tf.reshape(X2[0:m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp1 = tf.transpose(X_matrix_tr_db_exp1)

        #X_matrix_dc_exp1 = tf.tile(tf.reshape(X[0:n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp1 = tf.tile(tf.reshape(X2[0:m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp1 = tf.transpose(X_matrix_tr_dc_exp1)

        #diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1
        #diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1
        #diff_X_dc_exp1 = X_matrix_dc_exp1 - X_matrix_tr_dc_exp1

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)
        #K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)

        diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1#/self.lengthscale_11
        #K_dada_exp1 = self.variance_11*(1.0+sqrt5*diff_X_da_exp1+(2/3)*tf.math.square(diff_X_da_exp1))*tf.math.exp(-sqrt5*diff_X_da_exp1)
        K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)

        diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1#/self.lengthscale_12
        #K_dbdb_exp1 = self.variance_12*(1.0+sqrt5*diff_X_db_exp1+(2/3)*tf.math.square(diff_X_db_exp1))*tf.math.exp(-sqrt5*diff_X_db_exp1)
        K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)
        #K_dcdc_exp1 = self.variance_13*tf.exp(-0.5*(diff_X_dc_exp1/self.lengthscale_13)**2)

        # Block for experiment 2

        X_matrix_da_exp2 = tf.tile(tf.reshape(X[n:2*n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp2 = tf.tile(tf.reshape(X2[m:2*m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp2 = tf.transpose(X_matrix_tr_da_exp2)

        X_matrix_db_exp2 = tf.tile(tf.reshape(X[n:2*n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp2 = tf.tile(tf.reshape(X2[m:2*m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp2 = tf.transpose(X_matrix_tr_db_exp2)

        #X_matrix_dc_exp2 = tf.tile(tf.reshape(X[n:2*n,2],(n,1)), (1,m))
        #X_matrix_tr_dc_exp2 = tf.tile(tf.reshape(X2[m:2*m,2],(m,1)), (1,n))
        #X_matrix_tr_dc_exp2 = tf.transpose(X_matrix_tr_dc_exp2)

        #diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2
        #diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2
        #diff_X_dc_exp2 = X_matrix_dc_exp2 - X_matrix_tr_dc_exp2

        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        #K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)
        #K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)

        diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2#/self.lengthscale_11
        #K_dada_exp2 = self.variance_11*(1.0+sqrt5*diff_X_da_exp2+(2/3)*tf.math.square(diff_X_da_exp2))*tf.math.exp(-sqrt5*diff_X_da_exp2)
        K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)

        diff_X_db_exp2 = X_matrix_db_exp2 - X_matrix_tr_db_exp2#/self.lengthscale_22
        #K_dbdb_exp2 = self.variance_22*(1.0+sqrt5*diff_X_db_exp2+(2/3)*tf.math.square(diff_X_db_exp2))*tf.math.exp(-sqrt5*diff_X_db_exp2)
        K_dbdb_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_db_exp2/self.lengthscale_22)**2)
        #K_dcdc_exp2 = self.variance_23*tf.exp(-0.5*(diff_X_dc_exp2/self.lengthscale_23)**2)

        K_xx = K_dada_exp1 + K_dbdb_exp1 + K_dada_exp1 + K_dbdb_exp1
        K_ff = K_dada_exp2 + K_dbdb_exp2 # + K_dcdc_exp2

        # Combine four blocks together

        K_xf = np.zeros((n,m))
        K_fx = np.zeros((n,m))

        #K_xf =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp1))))
        #K_fx =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp2))))

        #diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        #print('K_xf',K_xf)

        K_upper_row = tf.concat([K_xx, K_xf], axis=1)
        K_lower_row = tf.concat([K_fx, K_ff], axis=1)
        K = tf.concat([K_upper_row, K_lower_row], axis=0)
        #return(K_dada * K_dbdb + K_dbda * K_dadb)
        return K

    #@params_as_tensors
    def K_diag(self, X):
        #n = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        n = int(X.shape[0]/2)
        diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_db_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_12)), (-1,))
        diag_dc_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11* self.variance_12)), (-1,))

        diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_db_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_22)), (-1,))
        #diag_dc_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11 * self.variance_22)), (-1,))

        block_upper = diag_da_exp1 + diag_db_exp1 + diag_dc_exp1
        block_lower = diag_da_exp2 + diag_db_exp2

        diag = tf.reshape(tf.stack([block_upper, block_lower]), (-1,))
        return diag

class K_full_model_11(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_11 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_11 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        self.variance_12 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_12 = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_13 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_13 = gpflow.Parameter(0.05, transform=positive())

        self.variance_22 = gpflow.Parameter(1.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))
        self.lengthscale_22 = gpflow.Parameter(50.0, transform=tfp.bijectors.SoftClip(
            gpflow.utilities.to_default_float(0.00001),
            gpflow.utilities.to_default_float(1000.0)))

        #self.variance_23 = gpflow.Parameter(1.0, transform=positive())
        #self.lengthscale_23 = gpflow.Parameter(10.0, transform=positive())

    #@params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        n = int(X.shape[0]/2)
        m = int(X2.shape[0]/2)

        sqrt5 = np.sqrt(5.0)

        # Block for experiment 1
        X_matrix_da_exp1 = tf.tile(tf.reshape(X[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp1 = tf.tile(tf.reshape(X2[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp1 = tf.transpose(X_matrix_tr_da_exp1)

        X_matrix_db_exp1 = tf.tile(tf.reshape(X[0:n,1],(n,1)), (1,m))
        X_matrix_tr_db_exp1 = tf.tile(tf.reshape(X2[0:m,1],(m,1)), (1,n))
        X_matrix_tr_db_exp1 = tf.transpose(X_matrix_tr_db_exp1)

        diff_X_da_exp1 = X_matrix_da_exp1 - X_matrix_tr_da_exp1#/self.lengthscale_11
        #K_dada_exp1 = self.variance_11*(1.0+sqrt5*diff_X_da_exp1+(2/3)*tf.math.square(diff_X_da_exp1))*tf.math.exp(-sqrt5*diff_X_da_exp1)
        K_dada_exp1 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp1/self.lengthscale_11)**2)

        diff_X_db_exp1 = X_matrix_db_exp1 - X_matrix_tr_db_exp1#/self.lengthscale_12
        #K_dbdb_exp1 = self.variance_12*(1.0+sqrt5*diff_X_db_exp1+(2/3)*tf.math.square(diff_X_db_exp1))*tf.math.exp(-sqrt5*diff_X_db_exp1)
        K_dbdb_exp1 = self.variance_12*tf.exp(-0.5*(diff_X_db_exp1/self.lengthscale_12)**2)
        #K_dcdc_exp1 = self.variance_13*tf.exp(-0.5*(diff_X_dc_exp1/self.lengthscale_13)**2)

        # Block for experiment 2

        X_matrix_da_exp2 = tf.tile(tf.reshape(X[n:2*n,0],(n,1)), (1,m))
        X_matrix_tr_da_exp2 = tf.tile(tf.reshape(X2[m:2*m,0], (m,1)), (1,n))
        X_matrix_tr_da_exp2 = tf.transpose(X_matrix_tr_da_exp2)

        X_matrix_dc_exp2 = tf.tile(tf.reshape(X[n:2*n,1],(n,1)), (1,m))
        X_matrix_tr_dc_exp2 = tf.tile(tf.reshape(X2[m:2*m,1],(m,1)), (1,n))
        X_matrix_tr_dc_exp2 = tf.transpose(X_matrix_tr_dc_exp2)

        diff_X_da_exp2 = X_matrix_da_exp2 - X_matrix_tr_da_exp2#/self.lengthscale_11
        #K_dada_exp2 = self.variance_11*(1.0+sqrt5*diff_X_da_exp2+(2/3)*tf.math.square(diff_X_da_exp2))*tf.math.exp(-sqrt5*diff_X_da_exp2)
        K_dada_exp2 = self.variance_11*tf.exp(-0.5*(diff_X_da_exp2/self.lengthscale_11)**2)

        diff_X_dc_exp2 = X_matrix_dc_exp2 - X_matrix_tr_dc_exp2#/self.lengthscale_22
        #K_dcdc_exp2 = self.variance_22*(1.0+sqrt5*diff_X_dc_exp2+(2/3)*tf.math.square(diff_X_dc_exp2))*tf.math.exp(-sqrt5*diff_X_dc_exp2)
        K_dcdc_exp2 = self.variance_22*tf.exp(-0.5*(diff_X_dc_exp2/self.lengthscale_22)**2)

        K_xx = K_dada_exp1 + K_dbdb_exp1 #+ K_dcdc_exp1
        K_ff = K_dada_exp2 + K_dcdc_exp2 + K_dada_exp2 * K_dcdc_exp2

        # Combine four blocks together

        K_xf = np.zeros((n,m))
        K_fx = np.zeros((n,m))

        #K_xf =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp1))))
        #K_fx =  tf.linalg.diag((tf.linalg.diag_part((K_dada_exp2))))

        #diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))

        #print('K_xf',K_xf)

        K_upper_row = tf.concat([K_xx, K_xf], axis=1)
        K_lower_row = tf.concat([K_fx, K_ff], axis=1)
        K = tf.concat([K_upper_row, K_lower_row], axis=0)
        #return(K_dada * K_dbdb + K_dbda * K_dadb)
        return K

    #@params_as_tensors
    def K_diag(self, X):
        n = int(X.shape[0]/2)
        diag_da_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        #diag_db_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_12)), (-1,))
        #diag_dc_exp1 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_13)), (-1,))

        diag_da_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11)), (-1,))
        diag_db_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_22)), (-1,))
        diag_dc_exp2 = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_11*self.variance_22)), (-1,))

        block_upper = diag_da_exp1 #+ diag_db_exp1 + diag_dc_exp1
        block_lower = diag_da_exp2 + diag_db_exp2 + diag_dc_exp2

        diag = tf.reshape(tf.stack([block_upper, block_lower]), (-1,))
        return diag
