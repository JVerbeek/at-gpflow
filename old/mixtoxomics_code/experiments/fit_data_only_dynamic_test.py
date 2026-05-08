'''
This script performs dynamic vs constant model test,
identifying differentially expressed genes without
testing for what type of response is present in the data.
'''
import numpy as np
import tensorflow_probability as tfp
import pandas as pd
from bisect import bisect_left
from scipy import special
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy.linalg import inv
import tensorflow as tf
import scipy.linalg
import gpflow
from scipy.stats.distributions import chi2
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from gpflow.utilities import print_summary, positive
from gpflow import set_trainable
from gpflow.config import default_float
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import random
import sys
sys.path.append('..')

f64 = gpflow.utilities.to_default_float
np.set_printoptions(suppress=True)

'''
Load the kernels
K_full_dynamics models changing response depending on concentrations
K_full_noise assumes constant response which does not depend on concentrations
'''
from utils.block_cov_utils_rbf_2d import (K_full_dynamics, K_full_noise)

def dfloat(value):  # default float
    return tf.cast(value, default_float())

def dint(value):  # default float
    return tf.cast(value, gpflow.default_int())

def likelihood_ratio(lik_reduced, lik_full):
    return(-np.asarray(lik_full)-(-np.asarray(lik_reduced)))

def load_data(data, gene_id, samples_exp1, samples_exp2):
    '''
    Data loading function
    Loads the data for the gene 'gene_id' from 'data' data frame
    Puts the data in the right format
    '''

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
    return df

'''
Main script
'''
# Set seed
np.random.seed(1000)

'''
Load the data first
'''
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

'''
Create empty list to record log-likelihood for the two models
(Dynamic and Constant)
'''

loglik_model_dynamic_list = []
loglik_model_noise_list = []
genes_list = []


'''
Start main loop for the analysis
Potentially split in batches if parallelizing
'''
for i in range(0,28617):
#for i in range(0,5000):
#for i in range(5000,10000):
#for i in range(10000,15000):
#for i in range(15000,20000):
#for i in range(20000,25000):
#for i in range(25000,28617):
    '''
    Load the data
    '''
    # Load the data to extract input points
    gene_id = genes[i]
    genes_list.append(gene_id)

    data = load_data(data_full, gene_id, samples_exp1, samples_exp2)
    data = (data[['X1', 'X2']].to_numpy(), data['Y'].to_numpy().reshape(-1,1))

    '''
    Fit GP models
    '''

    '''
    Model 1 (Dynamic)
    '''
    try:
        # Choose the dynamic kernel
        k = K_full_dynamics()

        # Define GP model
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)

        # Initialize likelihood variance
        m.likelihood.variance.assign(0.001)

        # Optimize the model
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        # Record the log-likelihood
        loglik_model_dynamic = np.asarray(m.training_loss())
        loglik_model_dynamic_list.append(loglik_model_dynamic)
    except:
        loglik_model_dynamic_list.append(np.nan)

    '''
    Model 2 (Constant)
    '''
    try:
        # Choose the constant kernel
        k = K_full_noise()

        # Define GP model
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)

        # Initialize likelihood variance
        m.likelihood.variance.assign(0.001)

        # Optimize the model
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        # Record the log-likelihood
        loglik_model_noise = np.asarray(m.training_loss())
        loglik_model_noise_list.append(loglik_model_noise)
    except:
        loglik_model_noise_list.append(np.nan)


'''
Save the results into a single data frame
'''
loglik_results = pd.DataFrame([np.asarray(loglik_model_dynamic_list), np.asarray(loglik_model_noise_list)])
loglik_results.index = ['model_dynamic', 'model_noise']
loglik_results = loglik_results.T

'''
Run the log-likelihood ratio test
'''
loglik_results['p_val'] = chi2.sf(likelihood_ratio(loglik_results['model_noise'], loglik_results['model_dynamic']), 3)
loglik_results['gene_id'] = np.asarray(genes_list).T

'''
Save the results into a csv file
'''
#loglik_results.to_csv('output/model_fit_only_dynamic_test_0_5000_new.csv')
#loglik_results.to_csv('output/model_fit_only_dynamic_test_5000_10000_new.csv')
#loglik_results.to_csv('output/model_fit_only_dynamic_test_10000_15000_new.csv')
#loglik_results.to_csv('output/model_fit_only_dynamic_test_15000_20000_new.csv')
#loglik_results.to_csv('output/model_fit_only_dynamic_test_20000_25000_new.csv')
#loglik_results.to_csv('output/model_fit_only_dynamic_test_25000_28617_new.csv')
#loglik_results.to_csv('output/model_fit_only_dynamic_short_list.csv')
loglik_results.to_csv('output/model_fit_all_genes_test.csv')
