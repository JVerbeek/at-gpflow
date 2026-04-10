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
import seaborn as sns
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
f64 = gpflow.utilities.to_default_float
import sys
sys.path.append('..')

from gpflow.config import default_float
#f64 = gpflow.utilities.to_default_float
def dfloat(value):  # default float
    return tf.cast(value, default_float())

def dint(value):  # default float
    return tf.cast(value, gpflow.default_int())

def likelihood_ratio(lik_reduced, lik_full):
    return(-np.asarray(lik_full)-(-np.asarray(lik_reduced)))

from mpl_toolkits.mplot3d import Axes3D
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import random
from scipy.stats.distributions import chi2

from utils.block_cov_utils_rbf_2d import (K_full_model_1, K_full_model_2, K_full_model_3, K_full_model_4, K_full_model_5, K_full_model_6,
                            K_full_model_7, K_full_model_8, K_full_model_9, K_full_model_10, K_full_model_11)

def load_data1(data, gene_id, samples_exp1, samples_exp2):

    y_exp1 = data[data['Unnamed: 0'] == gene_id][samples_exp1].to_numpy().reshape(-1,1)
    y_exp2 = data[data['Unnamed: 0'] == gene_id][samples_exp2].to_numpy().reshape(-1,1)

    TU_IMI_exp1 = metadata[metadata['experiment']==1]['conc_IMI'].to_numpy().reshape(-1,1)
    TU_IMI_exp2 = metadata[metadata['experiment']==2]['conc_IMI'].to_numpy().reshape(-1,1)

    TU_CLO_exp1 = metadata[metadata['experiment']==1]['conc_CLO'].to_numpy().reshape(-1,1)
    TU_CLO_exp2 = metadata[metadata['experiment']==2]['conc_CLO'].to_numpy().reshape(-1,1)

    TU_CYPRO_exp1 = metadata[metadata['experiment']==1]['conc_CYPRO'].to_numpy().reshape(-1,1)
    TU_CYPRO_exp2 = metadata[metadata['experiment']==2]['conc_CYPRO'].to_numpy().reshape(-1,1)

    df_exp1 = pd.concat([pd.DataFrame(TU_IMI_exp1), pd.DataFrame(TU_CLO_exp1), pd.DataFrame(y_exp1)], axis=1)
    df_exp2 = pd.concat([pd.DataFrame(TU_IMI_exp2), pd.DataFrame(TU_CYPRO_exp2), pd.DataFrame(y_exp2)], axis=1)

    df_exp1.columns = ['conc_IMI', 'conc_CLO', 'y_exp1']
    df_exp2.columns = ['conc_IMI', 'conc_CYPRO', 'y_exp2']
    df_exp1['exp'] = 'exp1'
    df_exp2['exp'] = 'exp2'
    df_exp1.columns = ['', '', '', '']
    df_exp2.columns = ['', '', '', '']

    df = pd.concat([df_exp1, df_exp2], axis=0)
    df.columns = ['X1', 'X2', 'Y', 'exp']
    df['Y'] = (df['Y'] - np.min(df['Y']))/(np.max(df['Y'])-np.min(df['Y']))
    #df.columns = ['conc_drug1', 'conc_drug2', 'y']
    #df['conc_drug12'] = df['conc_drug1']*df['conc_drug2']
    #df_X = pd.concat([df['conc_drug1'], df['conc_drug2'], df['conc_drug12']], axis=1)
    #df_X.columns = ['','','']
    #X_full = df[[]].to_numpy()
    #Y_full = df['y'].to_numpy().reshape(-1,1)
    #Y_full = (Y_full - np.min(Y_full))/(np.max(Y_full)-np.min(Y_full))

    return df

# Set seed
np.random.seed(1000)

single_replicate_no_noise = False
three_replicates_noise_sd_005 = False
three_replicates_noise_sd_01 = True

# Set flat_data to True if fitting simulated data with no dynamics
flat_data = False

if(sum([single_replicate_no_noise,three_replicates_noise_sd_005,three_replicates_noise_sd_01]) > 1):
    print("Error: please specify only one experiment")
    exit()

data_full_exp1 = pd.read_csv('../data/raw_exp1.csv')
data_full_exp2 = pd.read_csv('../data/raw_exp2.csv')
data_full_exp2 = data_full_exp2.drop(columns=['Unnamed: 0'])
metadata = pd.read_csv('../data/metadata.csv')

data_full = pd.concat([data_full_exp1, data_full_exp2], axis=1)

genes = data_full['Unnamed: 0'].to_numpy()
subset = data_full[data_full['Unnamed: 0'].isin(genes)]

samples_exp1 = metadata[metadata['experiment']==1]['sample']
samples_exp1 = samples_exp1.to_numpy().astype(str).tolist()

samples_exp2 = metadata[metadata['experiment']==2]['sample']
samples_exp2 = samples_exp2.to_numpy().astype(str).tolist()

string = 'Sample_'
samples_exp1 = [string + x for x in samples_exp1]
samples_exp2 = [string + x for x in samples_exp2]

# Load the data to extract input points
i=0
gene_id = genes[i]
data = load_data1(data_full, gene_id, samples_exp1, samples_exp2)
data = (data[['X1', 'X2']].to_numpy(), data['Y'].to_numpy().reshape(-1,1))

# Input points which will be used for data generation
X = data[0]

# Shape parameters
n_shape = X.shape[0]
n_half_shape = int(X.shape[0]/2)

# Doses for experiment 1 and 2
X1 = X[0:n_half_shape,:]
X2 = X[n_half_shape:n_shape,:]

# Rearrange unique doses, additional replicates added later
X1 = np.unique(X1, axis=0)
X2 = np.unique(X2, axis=0)
X = np.concatenate([X1,X2])

# Load pre-generated data
if(flat_data):
    df_data = pd.read_csv('df_flat.csv')
else:
    df_data = pd.read_csv('df_dynamics.csv')

loglik_model_list = []
genes_list = []

num_replicates = 3
sigma1 = 0.05
sigma2 = 0.1

for i in range(0,2200):
    '''
    Create noisy replicates in the data
    '''

    if(single_replicate_no_noise):
        Y = df_data.iloc[i][0:56].to_numpy().reshape(-1,1)
        X_replicates = X.copy()
    if(three_replicates_noise_sd_005):
        Y1 = df_data.iloc[i][0:28].to_numpy()
        Y2 = df_data.iloc[i][28:56].to_numpy()

        Y1 = np.concatenate([Y1+np.random.normal(0, sigma1, Y1.shape[0]),
                             Y1+np.random.normal(0, sigma1, Y1.shape[0]),
                             Y1+np.random.normal(0, sigma1, Y1.shape[0])] )

        Y2 = np.concatenate([Y2+np.random.normal(0, sigma1, Y2.shape[0]),
                             Y2+np.random.normal(0, sigma1, Y2.shape[0]),
                             Y2+np.random.normal(0, sigma1, Y2.shape[0])] )

        Y = np.concatenate([Y1, Y2]).reshape(-1,1)

        X_replicates = np.concatenate([X[0:28,:],X[0:28,:],X[0:28,:],
                                      X[28:56,:],X[28:56,:],X[28:56,:]])

    if(three_replicates_noise_sd_01):
        Y1 = df_data.iloc[i][0:28].to_numpy()
        Y2 = df_data.iloc[i][28:56].to_numpy()

        Y1 = np.concatenate([Y1+np.random.normal(0, sigma2, Y1.shape[0]),
                             Y1+np.random.normal(0, sigma2, Y1.shape[0]),
                             Y1+np.random.normal(0, sigma2, Y1.shape[0])] )

        Y2 = np.concatenate([Y2+np.random.normal(0, sigma2, Y2.shape[0]),
                             Y2+np.random.normal(0, sigma2, Y2.shape[0]),
                             Y2+np.random.normal(0, sigma2, Y2.shape[0])] )

        Y = np.concatenate([Y1, Y2]).reshape(-1,1)

        X_replicates = np.concatenate([X[0:28,:],X[0:28,:],X[0:28,:],
                                      X[28:56,:],X[28:56,:],X[28:56,:]])

    '''
    Fit GP models
    '''

    # noise model
    try:
        k = gpflow.kernels.RBF()

        m = gpflow.models.GPR(data=(X_replicates,Y), kernel=k, mean_function=None)
        m.kernel.lengthscales.assign(1000000.0)
        gpflow.utilities.set_trainable(m.kernel.lengthscales, False)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model = np.asarray(m.training_loss())

        loglik_model_list.append(loglik_model)
    except:
        loglik_model_list.append(np.nan)


loglik_results = pd.DataFrame([np.asarray(loglik_model_list)])

loglik_results.index = ['model_noise']


loglik_results = loglik_results.T

if(flat_data):
    if(single_replicate_no_noise):
        loglik_results.to_csv('output/loglik_results_simulated_data_noise_model_flat.csv')
    if(three_replicates_noise_sd_005):
        loglik_results.to_csv('output/loglik_results_simulated_flat_data_noise_model_noise_sd_005.csv')
    if(three_replicates_noise_sd_01):
        loglik_results.to_csv('output/loglik_results_simulated_flat_data_noise_model_noise_sd_01.csv')
else:
    if(single_replicate_no_noise):
        loglik_results.to_csv('output/loglik_results_simulated_data_noise_model.csv')
    if(three_replicates_noise_sd_005):
        loglik_results.to_csv('output/loglik_results_simulated_data_noise_sd_005_noise_model.csv')
    if(three_replicates_noise_sd_01):
        loglik_results.to_csv('output/loglik_results_simulated_data_noise_sd_01_noise_model.csv')
