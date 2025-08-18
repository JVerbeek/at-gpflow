import gpflow as gpf
from gpflow.kernels import Kernel
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
class TransferKernel(Kernel):
    def __init__(self, mu, b, kernel):
        super().__init__()
        self.mu = gpf.Parameter(1, transform=tfp.bijectors.Exp())
        self.b = gpf.Parameter(0.5, transform=tfp.bijectors.Exp())
        self.kernel = kernel
        
    def get_lmb(self):
        lmb = 2 * ((1/(1 + self.mu)) ** self.b) - 1
        return lmb
        
        
    def interdomain(self, X, X2):
        """Computes the between-task correlation.

        Returns:
            _description_
        """
        lmb = 2 * ((1/(1 + self.mu)) ** self.b) - 1
        return lmb * self.kernel(X, X2) 
    
    def K(self, X, X2=None, source_length=None, source2_length=0, full_output_cov=False):
        if not source_length:
            source_length=int(len(X)/2)
        Sx, Tx = X[:source_length], X[source_length:]
        Kss = self.kernel(Sx, Sx)
        if (X2 is None): 
            Ktt = self.kernel(Tx, Tx)
            Kst = self.interdomain(Sx, Tx) 
            Kts = tf.transpose(Kst) 
            return tf.concat([tf.concat([Kss, Kst], 1), tf.concat([Kts, Ktt], 1)], 0)
        else:
            Sx2, Tx2 = X2[:source2_length], X2[source2_length:]
            Kss = self.kernel(Sx, Sx2)
            Ktt = self.kernel(Tx, Tx2)
            Kst = self.interdomain(Sx, Tx2)
            Kts = self.interdomain(Tx, Sx2)
            return tf.concat([tf.concat([Kss, Kst], 1), tf.concat([Kts, Ktt], 1)], 0)
        
                
    def K_diag(self, X):
        return tf.concat((self.kernel.K_diag(X)), 0)
