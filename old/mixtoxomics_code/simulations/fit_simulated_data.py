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
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from gpflow import set_trainable
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
f64 = gpflow.utilities.to_default_float
import sys
sys.path.append('..')
np.set_printoptions(suppress=True)

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
    return df

# Set seed
np.random.seed(1000)

single_replicate_no_noise = True
three_replicates_noise_sd_005 = False
three_replicates_noise_sd_01 = False
three_replicates_noise_sd_03 = False

# Set flat_data to True if fitting simulated data with no dynamics
flat_data = True

if(sum([single_replicate_no_noise,three_replicates_noise_sd_005,three_replicates_noise_sd_01], three_replicates_noise_sd_03) > 1):
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
    df_data = pd.read_csv('simulated_data/df_flat.csv')
else:
    df_data = pd.read_csv('simulated_data/df_dynamics.csv')

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

num_replicates = 3
sigma1 = 0.05
sigma2 = 0.1
sigma3 = 0.3

for i in range(0,2200):
#for i in range(0,10):
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
    if(three_replicates_noise_sd_03):
        Y1 = df_data.iloc[i][0:28].to_numpy()
        Y2 = df_data.iloc[i][28:56].to_numpy()

        Y1 = np.concatenate([Y1+np.random.normal(0, sigma3, Y1.shape[0]),
                             Y1+np.random.normal(0, sigma3, Y1.shape[0]),
                             Y1+np.random.normal(0, sigma3, Y1.shape[0])] )

        Y2 = np.concatenate([Y2+np.random.normal(0, sigma3, Y2.shape[0]),
                             Y2+np.random.normal(0, sigma3, Y2.shape[0]),
                             Y2+np.random.normal(0, sigma3, Y2.shape[0])] )

        Y = np.concatenate([Y1, Y2]).reshape(-1,1)

        X_replicates = np.concatenate([X[0:28,:],X[0:28,:],X[0:28,:],
                                      X[28:56,:],X[28:56,:],X[28:56,:]])


    '''
    Fit GP models
    '''

    # model 1
    try:
        k = K_full_model_1()

        m = gpflow.models.GPR(data=(X_replicates,Y), kernel=k, mean_function=None)

        m.likelihood.variance.assign(0.001)

        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_1 = np.asarray(m.training_loss())

        loglik_model_1_list.append(loglik_model_1)
    except:
        loglik_model_1_list.append(np.nan)

    # model 2
    try:
        k = K_full_model_2()
        m = gpflow.models.GPR(data=(X_replicates,Y), kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_2 = np.asarray(m.training_loss())

        loglik_model_2_list.append(loglik_model_2)
    except:
        loglik_model_2_list.append(np.nan)

    # model 3
    try:
        k = K_full_model_3()
        m = gpflow.models.GPR(data=(X_replicates,Y), kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_3 = np.asarray(m.training_loss())

        loglik_model_3_list.append(loglik_model_3)
    except:
        loglik_model_3_list.append(np.nan)

    # model 4
    try:
        k = K_full_model_4()
        m = gpflow.models.GPR(data=(X_replicates,Y), kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_4 = np.asarray(m.training_loss())

        loglik_model_4_list.append(loglik_model_4)
    except:
        loglik_model_4_list.append(np.nan)

    # model 5
    try:
        k = K_full_model_5()
        m = gpflow.models.GPR(data=(X_replicates,Y), kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_5 = np.asarray(m.training_loss())

        loglik_model_5_list.append(loglik_model_5)
    except:
        loglik_model_5_list.append(np.nan)


    # model 6
    try:
        k = K_full_model_6()
        m = gpflow.models.GPR(data=(X_replicates,Y), kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_6 = np.asarray(m.training_loss())

        loglik_model_6_list.append(loglik_model_6)
    except:
        loglik_model_6_list.append(np.nan)

    # model 7
    try:
        k = K_full_model_7()
        m = gpflow.models.GPR(data=(X_replicates,Y), kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_7 = np.asarray(m.training_loss())

        loglik_model_7_list.append(loglik_model_7)
    except:
        loglik_model_7_list.append(np.nan)


    # model 8
    try:
        k = K_full_model_8()
        m = gpflow.models.GPR(data=(X_replicates,Y), kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_8 = np.asarray(m.training_loss())

        loglik_model_8_list.append(loglik_model_8)
    except:
        loglik_model_8_list.append(np.nan)

    # model 9
    try:
        k = K_full_model_9()
        m = gpflow.models.GPR(data=(X_replicates,Y), kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_9 = np.asarray(m.training_loss())

        loglik_model_9_list.append(loglik_model_9)
    except:
        loglik_model_9_list.append(np.nan)

    # model 10
    try:
        k = K_full_model_10()
        m = gpflow.models.GPR(data=(X_replicates,Y), kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_10 = np.asarray(m.training_loss())

        loglik_model_10_list.append(loglik_model_10)
    except:
        loglik_model_10_list.append(np.nan)


    # model 11
    try:
        k = K_full_model_11()
        m = gpflow.models.GPR(data=(X_replicates,Y), kernel=k, mean_function=None)
        m.likelihood.variance.assign(0.001)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        loglik_model_11 = np.asarray(m.training_loss())

        loglik_model_11_list.append(loglik_model_11)
    except:
        loglik_model_11_list.append(np.nan)


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
loglik_results['p_val_71'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_7']), 2)
loglik_results['p_val_73'] = chi2.sf(likelihood_ratio(loglik_results['model_3'], loglik_results['model_7']), 2)
loglik_results['p_val_87'] = chi2.sf(likelihood_ratio(loglik_results['model_7'], loglik_results['model_8']), 2)
loglik_results['p_val_97'] = chi2.sf(likelihood_ratio(loglik_results['model_7'], loglik_results['model_9']), 2)
loglik_results['p_val_98'] = chi2.sf(likelihood_ratio(loglik_results['model_8'], loglik_results['model_9']), 2)
loglik_results['p_val_82'] = chi2.sf(likelihood_ratio(loglik_results['model_2'], loglik_results['model_8']), 2)
loglik_results['p_val_53'] = chi2.sf(likelihood_ratio(loglik_results['model_3'], loglik_results['model_5']), 2)
loglik_results['p_val_31'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_3']), 2)
loglik_results['p_val_21'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_2']), 2)
loglik_results['p_val_41'] = chi2.sf(likelihood_ratio(loglik_results['model_1'], loglik_results['model_4']), 2)
loglik_results['p_val_87'] = chi2.sf(likelihood_ratio(loglik_results['model_7'], loglik_results['model_8']), 2)
loglik_results['p_val_95'] = chi2.sf(likelihood_ratio(loglik_results['model_5'], loglik_results['model_9']), 2) 

if(flat_data):
    if(single_replicate_no_noise):
        loglik_results.to_csv('output/loglik_results_simulated_data_flat.csv')
    if(three_replicates_noise_sd_005):
        loglik_results.to_csv('output/loglik_results_simulated_flat_data_noise_sd_005.csv')
    if(three_replicates_noise_sd_01):
        loglik_results.to_csv('output/loglik_results_simulated_flat_data_noise_sd_01.csv')
    if(three_replicates_noise_sd_03):
        loglik_results.to_csv('output/loglik_results_simulated_flat_data_noise_sd_03.csv')

else:
    if(single_replicate_no_noise):
        loglik_results.to_csv('output/loglik_results_simulated_data.csv')
    if(three_replicates_noise_sd_005):
        loglik_results.to_csv('output/loglik_results_simulated_data_noise_sd_005.csv')
    if(three_replicates_noise_sd_01):
        loglik_results.to_csv('output/loglik_results_simulated_data_noise_sd_01.csv')
    if(three_replicates_noise_sd_03):
        loglik_results.to_csv('output/loglik_results_simulated_data_noise_sd_03.csv')
