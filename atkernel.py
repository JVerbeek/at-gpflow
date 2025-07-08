import gpflow as gpf
from gpflow.kernels import Kernel
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
class TransferKernel(Kernel):
    def __init__(self, mu, b, kernel):
        super().__init__()
        self.mu = gpf.Parameter(5, transform=tfp.bijectors.Exp())
        self.b = gpf.Parameter(0.5, transform=tfp.bijectors.Exp())
        self.kernel = kernel
        
    def interdomain(self, X, X2):
        """Computes the between-task correlation.

        Returns:
            _description_
        """
        lmb = 2 * ((1/(1 + self.mu)) ** self.b) - 1
        return lmb * self.kernel(X, X2) 
    
    def K(self, X, X2, source_length=None, full_output_cov=False):
        Sx, Tx = X[:source_length], X[source_length:]
        Kss = self.kernel(Sx, Sx)
        if X2 == None: 
            return Kss
        else:
            Ktt = self.kernel(Tx, Tx)
            Kst = self.interdomain(Sx, Tx)
            Kts = tf.transpose(Kst)
            return tf.concat([tf.concat([Kss, Kst], 0), tf.concat([Kts, Ktt], 0)], 1)
                
    def K_diag(self, X):
        return tf.concat((self.kernel.K_diag(X)), 0)


S = np.random.normal(0, 1, 100).reshape(-1, 1)
T = np.random.normal(0, 1, 100).reshape(-1, 1) 
