import numpy as np
from numpy.linalg import inv
import scipy.spatial
import pandas as pd
from bisect import bisect_left
from scipy import special
from scipy.special import erf
import numpy as np
import scipy.spatial
import scipy.linalg
from scipy.stats.distributions import chi2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import tensorflow_probability as tfp
import tensorflow as tf

import gpflow
from gpflow.utilities import print_summary, positive

np.set_printoptions(suppress=True)

def compute_prior_hyperparameters(mean, variance):
    # mean equal to variance of empirical standard deviation
    beta = mean/variance
    alpha = mean**2/variance
    return alpha, beta

def trapezoidal_area(xyz):
    """Calculate volume under a surface defined by irregularly spaced points
    using delaunay triangulation. "x,y,z" is a <numpoints x 3> shaped ndarray."""
    d = scipy.spatial.Delaunay(xyz[:,:2])
    tri = xyz[d.vertices]

    a = tri[:,0,:2] - tri[:,1,:2]
    b = tri[:,0,:2] - tri[:,2,:2]
    proj_area = np.cross(a, b).sum(axis=-1)
    zavg = tri[:,:,2].sum(axis=1)
    vol = zavg * np.abs(proj_area) / 6.0
    return vol.sum()

def predict_in_observations(X1, X2, m_full):
    #[Xi, Xj] = np.meshgrid(X1, X2)
    [Xi, Xj] = np.meshgrid(X1,X2)

    # We need to augument our test space to be a list of coordinates for input to the GP
    Xnew2 = np.vstack((Xi.ravel(), Xj.ravel())).T # Change our input grid to list of coordinates

    # Predict the mean and covariance of the GP fit at the test locations
    mean2_full, Cov2_full = m_full.predict_f(Xnew2)

    mean2_full = np.asarray(mean2_full)
    Cov2_full = np.asarray(Cov2_full)

    print(Xi.shape)
    print(Xj.shape)

    return  mean2_full.reshape(Xj.shape), Cov2_full.reshape(Xj.shape)

def predict_in_observations_lower(X1, X2, m_full):
    #[Xi, Xj] = np.meshgrid(X1, X2)
    [Xi, Xj] = np.meshgrid(X1,X2)

    # We need to augument our test space to be a list of coordinates for input to the GP
    Xnew2 = np.vstack((Xi.ravel(), Xj.ravel())).T # Change our input grid to list of coordinates

    # Predict the mean and covariance of the GP fit at the test locations
    mean2_full, Cov2_full = m_full.predict_f(Xnew2)

    mean2_full = np.asarray(mean2_full)
    Cov2_full = np.asarray(Cov2_full)

    return  mean2_full.reshape(Xi.shape), Cov2_full.reshape(Xi.shape)

def predict_in_observations_upper(X1, X2, m_full):
    #[Xi, Xj] = np.meshgrid(X1, X2)
    [Xi, Xj] = np.meshgrid(X1,X2)

    # We need to augument our test space to be a list of coordinates for input to the GP
    Xnew2 = np.vstack((Xi.ravel(), Xj.ravel())).T # Change our input grid to list of coordinates

    # Predict the mean and covariance of the GP fit at the test locations
    mean2_full, Cov2_full = m_full.predict_f(Xnew2)

    mean2_full = np.asarray(mean2_full)
    Cov2_full = np.asarray(Cov2_full)

    return  mean2_full.reshape(Xi.shape), Cov2_full.reshape(Xi.shape)


def fit_Hand(X1, X2, dim2_A, dim2_B, Dose_A, Dose_B):

    xv, yv = np.meshgrid(X1, X2)

    Y_expected_Hand = np.zeros(xv.shape)

    mean1 =  dim2_A.copy().reshape(-1,1)
    mean2 =  dim2_B.copy().reshape(-1,1)
    xx1 = np.linspace(np.min(Dose_A), np.max(Dose_A), mean1.shape[0]).reshape(-1,1)
    xx2 = np.linspace(np.min(Dose_B), np.max(Dose_B), mean2.shape[0]).reshape(-1,1)

    # new based on 2D
    for i in range(0,xv.shape[0]):
        for j in range(0,xv.shape[1]):
            Y_expected_Hand[i,j] = y_exp_Hand(xv[i,j], yv[i,j], mean1, xx1, mean2, xx2)

    return Y_expected_Hand

def y_exp_Hand(a, b, f_a, xx_a, f_b, xx_b):
    '''
    Outputs expected effect for Hand model
    a and b are vetors of partitions of the doses a and b
    '''
    ### For a dose b1 find the effect f(b)
    dose_splits_drug_a = int(find_nearest_above(xx_a, a))
    dose_splits_drug_b = int(find_nearest_above(xx_b, b))

    a_partition = np.repeat(a/dose_splits_drug_a, [dose_splits_drug_a], axis=0)
    b_partition = np.repeat(b/dose_splits_drug_b, [dose_splits_drug_b], axis=0)

    total_dose_a = np.repeat(0.0, [dose_splits_drug_a], axis=0)
    total_dose_b = np.repeat(0.0, [dose_splits_drug_b], axis=0)

    indx_dose_a = np.repeat(0.0, [dose_splits_drug_a], axis=0)
    indx_dose_b = np.repeat(0.0, [dose_splits_drug_b], axis=0)

    count_dose_b = 1.0
    count_dose_a = 1.0

    ### For index 0
    ## Apply dose b[0]
    i = 0
    j = 0
    total_dose_b[j] = b_partition[j]
    # Find index of the dose
    indx_dose_b[j] = find_nearest_above(xx_b, b_partition[j])

    # Find corresponding effect
    Effect_reached_b = np.asarray(f_b[int(indx_dose_b[j])])[0] # find the effect b0 reaches
    ### Find dose An for which A reaches effect Eb
    near_effect = find_nearest_above(f_a, Effect_reached_b)   # find the effect for GP predicted for drug A which is close to effect
                                                        # reached by b1
    #indx_effect = np.where(np.isclose(f_a, near_effect))[0][0] # fine the index of this effect
    indx_effect = find_nearest_above(f_a, Effect_reached_b)

    total_dose_a[i] = xx_a[indx_effect] + a_partition[i]       # add the dose a1 to the effect already reached by b1

    # Find closest effect for total dose
    indx_dose_final = find_nearest_above(xx_a, total_dose_a[i])
    # Finally the effect for the total dose
    total_effect = np.asarray(f_a[indx_dose_final])[0]             # find total effect reached
    i = i + 1
    j = j + 1
    #print('Total effect at 0', total_effect)

    while (i < dose_splits_drug_a and j < dose_splits_drug_b):

        near_effect = find_nearest(f_b, total_effect)
        indx_dose = np.where(np.isclose(f_b, near_effect))[0][0] # fine the index of this effect

        total_dose_b[j] = xx_b[int(indx_dose)] + count_dose_b*b_partition[j]

        # Find index of the dose
        indx_dose_b[j] = find_nearest_above(xx_b, total_dose_b[j])
        #print('indx_dose_b[j]', indx_dose_b[j], 'indx_dose_b[j-1]', indx_dose_b[j-1])
        if(indx_dose_b[j-1] == indx_dose_b[j]):
            count_dose_b =  count_dose_b + 1.0
        else:
            count_dose_b = 1.0

        # Find corresponding effect
        Effect_reached_b = np.asarray(f_b[int(indx_dose_b[j])])[0] # find the effect b0 reaches

        ### Find dose An for which A reaches effect Eb
        near_effect = find_nearest(f_a, Effect_reached_b)   # find the effect for GP predicted for drug A which is close to effect
                                                            # reached by b1
        indx_effect = np.where(np.isclose(f_a, near_effect))[0][0] # fine the index of this effect
        #indx_effect = find_nearest_above(f_a, Effect_reached_b)

        total_dose_a[i] = xx_a[indx_effect] + count_dose_a*a_partition[i]       # add the dose a1 to the effect already reached by b1

        # Find closest effect for total dose
        indx_dose_a[i] = find_nearest_above(xx_a, total_dose_a[i])

        if(indx_dose_a[i-1] == indx_dose_a[i]):
            count_dose_a =  count_dose_a + 1.0
        else:
            count_dose_a = 1.0

        # Finally the effect for the total dose
        total_effect = np.asarray(f_a[int(indx_dose_a[i])])[0]             # find total effect reached
        i = i + 1
        j = j + 1

    if (dose_splits_drug_a > dose_splits_drug_b):
        #a_remaining_dose = a - total_dose_a[i-1]
        a_remaining_dose = dose_splits_drug_a * a_partition[0] - dose_splits_drug_b * a_partition[0]
        # Apply remaining dose
        total_dose_a_final = total_dose_a[i-1] + a_remaining_dose
        indx_dose_a_final = find_nearest_above(xx_a, total_dose_a_final)
        total_effect = np.asarray(f_a[indx_dose_a_final])[0]

    elif(dose_splits_drug_a < dose_splits_drug_b):
        #b_remaining_dose = b - total_dose_b[j]
        b_remaining_dose = dose_splits_drug_b * b_partition[0] - dose_splits_drug_a * b_partition[0]
        total_dose_b_final = total_dose_b[j-1] + b_remaining_dose
        indx_dose_b_final  = find_nearest_above(xx_b, total_dose_b_final)
        total_effect = np.asarray(f_b[indx_dose_b_final])[0]
    else:
        print("Dose splits are equal")
    return(total_effect)

def y_exp(a, b, f_a, xx_a, f_b, xx_b):
    # Gives y_exp(a+b) fot Tallarida
    ### For a dose B find the effect Eb = f(b)
    # Find index of the dose
    near_b_dose = find_nearest(xx_b, b)[0]
    indx_dose = np.where(np.isclose(xx_b, near_b_dose ))
    # Find corresponding effect
    Eb = np.asarray(f_b[indx_dose[0][0]])[0]

    ### Find dose An for which A reaches effect Eb
    near_effect = find_nearest(f_a, Eb)[0]
    indx_effect = np.where(np.isclose(f_a, near_effect))[0][0]
    total_dose = xx_a[indx_effect] + a
    # Find closest effect for total dose
    near_dose = find_nearest(xx_a, total_dose)[0]
    indx_dose_final = np.where(np.isclose(xx_a, near_dose))[0][0]
    # Finally the effect for the total dose
    total_effect = np.asarray(f_a[indx_dose_final])[0]

    return(total_effect)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        #return np.nan # returns None if target is greater than any value
        return int(my_array.shape[0]-1) # returns max value if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

#def main(xyz):
#    area_underneath = trapezoidal_area(xyz)
#    print(area_underneath)

def trapezoidal_area(xyz):
    """Calculate volume under a surface defined by irregularly spaced points
    using delaunay triangulation. "x,y,z" is a <numpoints x 3> shaped ndarray."""
    d = scipy.spatial.Delaunay(xyz[:,:2])
    tri = xyz[d.vertices]

    a = tri[:,0,:2] - tri[:,1,:2]
    b = tri[:,0,:2] - tri[:,2,:2]
    proj_area = np.cross(a, b).sum(axis=-1)
    zavg = tri[:,:,2].sum(axis=1)
    vol = zavg * np.abs(proj_area) / 6.0
    return vol.sum()

class K_log(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__(active_dims=[0])
        self.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.lengthscale = gpflow.Parameter(2.0, transform=gpflow.utilities.positive())

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #X = tf.math.log(X+self.lengthscale_da)
        X_logscaled = tf.math.log(X/self.lengthscale+1.0)
        #X_logscaled[:,1] = tf.math.log(X[:,1]/self.lengthscale_db+1)
        X2_logscaled = tf.math.log(X2/self.lengthscale+1.0)
        #X2_logscaled[:,1] = tf.math.log(X2[:,1]/self.lengthscale_db+1)

        X_matrix = tf.tile(tf.reshape(X_logscaled,(n,1)), (1,m))
        X_matrix_tr = tf.tile(tf.reshape(X2_logscaled, (m,1)), (1,n))
        X_matrix_tr = tf.transpose(X_matrix_tr)

        diff_X = X_matrix - X_matrix_tr

        K_dada = tf.exp(-0.5*(diff_X)**2)

        return self.variance*K_dada

    def K_diag(self, X, presliced=None):
        n = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance)), (-1,))
        #diag_db = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_db)), (-1,))
        return diag

def log_kernel_new(X, X2, lengthscale, variance):
    n = np.asscalar(np.array(X.shape[0]).astype(int))
    m = np.asscalar(np.array(X2.shape[0]).astype(int))

    X = X/lengthscale+1
    X2 = X2/lengthscale+1

    X_l = X + lengthscale
    X2_l = X2 + lengthscale

    X_matrix = np.tile(X[0:n], (1,m)).reshape(n,m)
    X_matrix_tr = np.tile(X2[0:m], (1,n)).reshape(m,n)
    X_matrix_tr = np.transpose(X_matrix_tr)

    X_l_matrix = np.tile(X_l[0:n], (1,m)).reshape(n,m)
    X_l_matrix_tr = np.tile(X2_l[0:m], (1,n)).reshape(m,n)
    X_l_matrix_tr = np.transpose(X_l_matrix_tr)

    part1 = -2.06637*variance**2
    part2 = 0.5*lengthscale**2*X_l_matrix_tr**2*erf(-0.707107*np.log(X_matrix)+0.707107*np.log(X_matrix_tr)+0.707107)
    part3 = 0.5*lengthscale**2*X_l_matrix**2*erf(-0.707107*np.log(X_matrix)-0.707107*np.log(X_matrix_tr)+0.707107)

    K = part1*(part2+part3)

    return(K)

class K_sqexp_kernel(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.lengthscale = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        X_matrix = tf.tile(tf.reshape(X[0:n],(n,1)), (1,m))
        X_matrix_tr = tf.tile(tf.reshape(X2[0:m], (m,1)), (1,n))
        X_matrix_tr = tf.transpose(X_matrix_tr)
        diff = X_matrix - X_matrix_tr

        part1 = -1.25331*self.variance**2*self.lengthscale
        part2 = diff.tf.math.erf(0.7*diff/self.lengthscale)+0.7*self.lengthscale*tf.math.exp((-0.5*th.math.square(X_matrix)+X_matrix*Xmatrix_tr -0.5*tf.square(X_matrix_tr))/self.lengthscale**2)

        return 100+part1*part2

    def K_diag(self, X, presliced=None):
        n = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m = tf.cast(tf.shape(X)[0], dtype=tf.int32)

        X = X/self.lengthscale+1
        X_l = X + self.lengthscale

        part1 = -2.06637*self.variance**2
        part2 = 0.5*self.lengthscale**2*X_l**2*tf.math.erf(-0.707107*tf.math.log(X)+0.707107*tf.math.log(X)+0.707107)
        part3 = 0.5*self.lengthscale**2*X_l**2*tf.math.erf(-0.707107*tf.math.log(X)-0.707107*tf.math.log(X)+0.707107)

        #diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(part1*(part2+part3))), (-1,))
        diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance)), (-1,))

        return diag

class K_new_kernel(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.lengthscale = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        X_logscaled = X/self.lengthscale+1
        X2_logscaled = X2/self.lengthscale+1

        X_l = X + self.lengthscale
        X2_l = X2 + self.lengthscale

        X_matrix = tf.tile(tf.reshape(X_logscaled[0:n],(n,1)), (1,m))
        X_matrix_tr = tf.tile(tf.reshape(X2_logscaled[0:m], (m,1)), (1,n))
        X_matrix_tr = tf.transpose(X_matrix_tr)

        X_l_matrix = tf.tile(tf.reshape(X_l[0:n],(n,1)), (1,m))
        X_l_matrix_tr = tf.tile(tf.reshape(X2_l[0:m], (m,1)), (1,n))
        X_l_matrix_tr = tf.transpose(X_l_matrix_tr)

        part1 = -2.06637*self.variance**2
        part2 = 0.5*self.lengthscale**2*tf.math.square(X_matrix_tr)*tf.math.erf(-0.707107*tf.math.log(X_matrix)+0.707107*tf.math.log(X_matrix_tr)+0.707107)
        part3 = 0.5*tf.math.square(X_l_matrix)*tf.math.erf(0.707107*tf.math.log(X_matrix)-0.707107*tf.math.log(X_matrix_tr)+0.707107)

        return part1*(part2+part3)

    def K_diag(self, X, presliced=None):
        n = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m = tf.cast(tf.shape(X)[0], dtype=tf.int32)

        X = X/self.lengthscale+1
        X_l = X + self.lengthscale

        part1 = -2.06637*self.variance**2
        part2 = 0.5*self.lengthscale**2*X*tf.math.erf(0.707107)
        part3 = 0.5*tf.math.square(X_l)*tf.math.erf(0.707107)

        final = part1*(part2+part3)
        #diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(part1*(part2+part3))), (-1,))
        diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(final)), (-1,))

        return diag

class K_multiplicative(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_da = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.lengthscale_da = gpflow.Parameter(10.0, transform=gpflow.utilities.positive())
        self.lengthscale_db = gpflow.Parameter(10.0, transform=gpflow.utilities.positive())

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        X_logscaled_dim0 = tf.math.log(X[:,0]/self.lengthscale_da+1)
        X_logscaled_dim1 = tf.math.log(X[:,1]/self.lengthscale_db+1)
        X2_logscaled_dim0 = tf.math.log(X2[:,0]/self.lengthscale_da+1)
        X2_logscaled_dim1 = tf.math.log(X2[:,1]/self.lengthscale_db+1)

        X_logscaled = tf.stack([X_logscaled_dim0, X_logscaled_dim1], axis=1)
        X2_logscaled = tf.stack([X2_logscaled_dim0, X2_logscaled_dim1], axis=1)

        X_matrix_da = tf.tile(tf.reshape(X_logscaled[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da = tf.tile(tf.reshape(X2_logscaled[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da = tf.transpose(X_matrix_tr_da)

        X_matrix_db = tf.tile(tf.reshape(X_logscaled[0:n,1],(n,1)), (1,m))
        X_matrix_tr_db = tf.tile(tf.reshape(X2_logscaled[0:m,1],(m,1)), (1,n))
        X_matrix_tr_db = tf.transpose(X_matrix_tr_db)

        diff_X_da = X_matrix_da - X_matrix_tr_da
        diff_X_db = X_matrix_db - X_matrix_tr_db

        K_dada = tf.exp(-0.5*(diff_X_da)**2)
        K_dbdb = tf.exp(-0.5*(diff_X_db)**2)

        return self.variance_da * K_dada * K_dbdb

    def K_diag(self, X, presliced=None):
        n = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_da)), (-1,))

        X_logscaled = np.zeros((n, 2), np.float64)
        X2_logscaled = np.zeros((m, 2), np.float64)

        X_logscaled[:,0] = tf.math.log(X[:,0]/self.lengthscale_da+1)
        X_logscaled[:,1] = tf.math.log(X[:,1]/self.lengthscale_db+1)

        diff_X_dbda =  X_logscaled[:,1] -  X_logscaled[:,0]
        diff_X_dadb =  X_logscaled[:,0] -  X_logscaled[:,1]

        K_dbda = self.variance_da*tf.exp(-0.5*(diff_X_dbda)**2)
        K_dadb = self.variance_da*tf.exp(-0.5*(diff_X_dadb)**2)

        return diag

def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
