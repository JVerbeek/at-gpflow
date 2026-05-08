############################################################
#Load the libraries
############################################################
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
from gpflow.config import default_float
from scipy.stats.distributions import chi2
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from gpflow.utilities import print_summary, positive
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from gpflow import set_trainable
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import sys
sys.path.append('..')
np.set_printoptions(suppress=True)
f64 = gpflow.utilities.to_default_float

def dfloat(value):  # default float
    return tf.cast(value, default_float())

from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.stats.distributions import chi2

# Load the kernel functions for different GP models
from utils.block_cov_utils_rbf_2d import (K_full_model_1, K_full_model_2, K_full_model_3, K_full_model_4, K_full_model_5, K_full_model_6,
                            K_full_model_7, K_full_model_8, K_full_model_9, K_full_model_10, K_full_model_11)

def dint(value):  # default float
    return tf.cast(value, gpflow.default_int())

def likelihood_ratio(lik_reduced, lik_full):
    return(-np.asarray(lik_full)-(-np.asarray(lik_reduced)))


def load_data(data, gene_id, samples_exp1, samples_exp2):
    '''
    This function is loading the data for individual genes and restructuting it
    into the right format
    data -- data frame with all genes;
    gene_id -- id of a gene;
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

############################################################
#Main code
############################################################


# Set seed
np.random.seed(1000)

'''
Loading data frames with Imi-Clo experiment (exp1) and Imi-Cypro experiment (exp2)
'''

data_full_exp1 = pd.read_csv('../data/raw_exp1.csv')
data_full_exp2 = pd.read_csv('../data/raw_exp2.csv')
data_full_exp2 = data_full_exp2.drop(columns=['Unnamed: 0'])
metadata = pd.read_csv('../data/metadata.csv')

# Combine into a single data frame
data_full = pd.concat([data_full_exp1, data_full_exp2], axis=1)

# List of gene id's
genes = data_full['Unnamed: 0'].to_numpy()

samples_exp1 = metadata[metadata['experiment']==1]['sample']
samples_exp1 = samples_exp1.to_numpy().astype(str).tolist()

samples_exp2 = metadata[metadata['experiment']==2]['sample']
samples_exp2 = samples_exp2.to_numpy().astype(str).tolist()

string = 'Sample_'
samples_exp1 = [string + x for x in samples_exp1]
samples_exp2 = [string + x for x in samples_exp2]

# Create lists to save likelihood of different models
loglik_model_1_list = []
loglik_model_2_list = []
loglik_model_3_list = []
loglik_model_4_list = []
loglik_model_5_list = []
loglik_model_6_list = []
loglik_model_7_list = []
loglik_model_8_list = []
loglik_model_9_list = []
loglik_model_10_list = []
loglik_model_11_list = []
genes_list = []

'''
Below the loop over all genes where for each we fit 11 GP models;
Note, that to save computational time these can be run in parallel
over different batches of genes [0,5000], [5000,10000], etc.
Total amounts to 28617 genes.
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
    Model 1
    '''
    try:
        # The kernel for model 1 is chosen
        k = K_full_model_1()

        # Go model is defined
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)

        # Hyperparameters are initialized
        m.kernel.lengthscale_11.assign(1.0)
        m.kernel.variance_11.assign(1.0)
        m.likelihood.variance.assign(0.001)

        # The model is optimized
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        # The log-likelihood of the model is saved
        loglik_model_1 = np.asarray(m.training_loss())
        loglik_model_1_list.append(loglik_model_1)
    except:
        loglik_model_1_list.append(np.nan)

    '''
    Model 2
    '''
    try:
        k = K_full_model_2()
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_2 = np.asarray(m.training_loss())

        loglik_model_2_list.append(loglik_model_2)
    except:
        loglik_model_2_list.append(np.nan)

    '''
    Model 3
    '''
    try:
        k = K_full_model_3()
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_3 = np.asarray(m.training_loss())

        loglik_model_3_list.append(loglik_model_3)
    except:
        loglik_model_3_list.append(np.nan)

    '''
    Model 4
    '''
    try:
        k = K_full_model_4()
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_4 = np.asarray(m.training_loss())

        loglik_model_4_list.append(loglik_model_4)
    except:
        loglik_model_4_list.append(np.nan)

    '''
    Model 5
    '''
    try:
        k = K_full_model_5()
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_5 = np.asarray(m.training_loss())

        loglik_model_5_list.append(loglik_model_5)
    except:
        loglik_model_5_list.append(np.nan)


    '''
    Model 6
    '''
    try:
        k = K_full_model_6()
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_6 = np.asarray(m.training_loss())

        loglik_model_6_list.append(loglik_model_6)
    except:
        loglik_model_6_list.append(np.nan)

    '''
    Model 7
    '''
    try:
        k = K_full_model_7()
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_7 = np.asarray(m.training_loss())

        loglik_model_7_list.append(loglik_model_7)
    except:
        loglik_model_7_list.append(np.nan)


    '''
    Model 8
    '''
    try:
        k = K_full_model_8()
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_8 = np.asarray(m.training_loss())

        loglik_model_8_list.append(loglik_model_8)
    except:
        loglik_model_8_list.append(np.nan)

    '''
    Model 9
    '''
    try:
        k = K_full_model_9()
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_9 = np.asarray(m.training_loss())

        loglik_model_9_list.append(loglik_model_9)
    except:
        loglik_model_9_list.append(np.nan)

    '''
    Model 10
    '''
    try:
        k = K_full_model_10()
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_10 = np.asarray(m.training_loss())

        loglik_model_10_list.append(loglik_model_10)
    except:
        loglik_model_10_list.append(np.nan)


    '''
    Model 11
    '''
    try:
        k = K_full_model_11()
        m = gpflow.models.GPR(data=data, kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_11 = np.asarray(m.training_loss())

        loglik_model_11_list.append(loglik_model_11)
    except:
        loglik_model_11_list.append(np.nan)


'''
Loglikelihood results are aggregated into a data frame
'''
loglik_results = pd.DataFrame([np.asarray(loglik_model_1_list), np.asarray(loglik_model_2_list),
             np.asarray(loglik_model_3_list), np.asarray(loglik_model_4_list),
             np.asarray(loglik_model_5_list), np.asarray(loglik_model_6_list),
             np.asarray(loglik_model_7_list), np.asarray(loglik_model_8_list),
             np.asarray(loglik_model_9_list), np.asarray(loglik_model_10_list),
                              np.asarray(loglik_model_11_list)])

loglik_results.index = ['model_1', 'model_2', 'model_3',
                         'model_4', 'model_5', 'model_6',
                         'model_7', 'model_8', 'model_9', 'model_10', 'model_11']


loglik_results = loglik_results.T

loglik_results['gene_id'] = np.asarray(genes_list).T

'''
Chi-square test computed
'''

loglik_results['p_val_21'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_2']), 2)
loglik_results['p_val_31'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_3']), 2)
loglik_results['p_val_41'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_4']), 4)
loglik_results['p_val_42'] = chi2.sf(likelihood_ratio(loglik_results['model_2'], loglik_results['model_4']), 2)
loglik_results['p_val_43'] = chi2.sf(likelihood_ratio(loglik_results['model_3'], loglik_results['model_3']), 2)
loglik_results['p_val_51'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_5']), 2)
loglik_results['p_val_52'] = chi2.sf(likelihood_ratio(loglik_results['model_2'], loglik_results['model_5']), 2)
loglik_results['p_val_61'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_6']), 2)
loglik_results['p_val_62'] = chi2.sf(likelihood_ratio(loglik_results['model_2'], loglik_results['model_6']), 2)
loglik_results['p_val_63'] = chi2.sf(likelihood_ratio(loglik_results['model_3'], loglik_results['model_6']), 2)
loglik_results['p_val_64'] = chi2.sf(likelihood_ratio(loglik_results['model_4'], loglik_results['model_6']), 2)
loglik_results['p_val_71'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_7']), 2) #???
loglik_results['p_val_73'] = chi2.sf(likelihood_ratio(loglik_results['model_3'], loglik_results['model_7']), 2) #???
loglik_results['p_val_87'] = chi2.sf(likelihood_ratio(loglik_results['model_7'], loglik_results['model_8']), 2) #???
loglik_results['p_val_97'] = chi2.sf(likelihood_ratio(loglik_results['model_7'], loglik_results['model_9']), 2) #???
loglik_results['p_val_98'] = chi2.sf(likelihood_ratio(loglik_results['model_8'], loglik_results['model_9']), 2) #???
loglik_results['p_val_82'] = chi2.sf(likelihood_ratio(loglik_results['model_2'], loglik_results['model_8']), 2) #???
loglik_results['p_val_53'] = chi2.sf(likelihood_ratio(loglik_results['model_3'], loglik_results['model_5']), 2) #???
loglik_results['p_val_31'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_3']), 2) #???
loglik_results['p_val_21'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_2']), 2) #???
loglik_results['p_val_41'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_4']), 2) #???
loglik_results['p_val_87'] = chi2.sf(likelihood_ratio(loglik_results['model_7'], loglik_results['model_8']), 2) #???
loglik_results['p_val_95'] = chi2.sf(likelihood_ratio(loglik_results['model_5'], loglik_results['model_9']), 2) #???

'''
The results are saves into corresponding file
'''
#loglik_results.to_csv('output/model_fit_all_genes_0_5000.csv')
#loglik_results.to_csv('output/model_fit_all_genes_5000_10000.csv')
#loglik_results.to_csv('output/model_fit_all_genes_10000_15000.csv')
#loglik_results.to_csv('output/model_fit_all_genes_15000_20000.csv')
#loglik_results.to_csv('output/model_fit_all_genes_20000_25000.csv')
#loglik_results.to_csv('output/model_fit_all_genes_25000_28617.csv')
