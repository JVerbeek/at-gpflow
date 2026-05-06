import gpflow as gpf
from gpflow.kernels import Kernel
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


f64 = lambda x: np.array(x, dtype=np.float64)

class TransferCoregion(Kernel):
    def __init__(self, kernel):
        super().__init__()
        W: AnyNDArray = 0.1 * np.random.randint(-10, 10, (1, 2)) 
        kappa = np.ones(2)
        self.W = gpf.Parameter(W)
        self.kernel = kernel
        self.kappa = gpf.Parameter(kappa, transform=tfp.bijectors.Softplus())
        
    def get_lmb(self):
        return self.lmb

    def get_B(self):
        B = tf.linalg.matmul(self.W, self.W, transpose_b=True) + tf.linalg.diag(self.kappa)
        return B / tf.math.reduce_max(B) 
        
    def interdomain(self, X, X2):
        """Computes the between-task correlation.

        Returns:
            _description_
        """
        B = self.get_B()
        return self.kernel(X, X2) * B[0, 1]

    def K(self, X, X2=None, full_output_cov=False):
        As = (X[:,1] == 0)
        Bs = (X[:,1] == 1)

        Ax = tf.reshape(X[:,0][As], (-1, 1))
        Bx  = tf.reshape(X[:,0][Bs], (-1, 1))

        indices_A = tf.reshape(tf.where(As), [-1])
        indices_B = tf.reshape(tf.where(Bs), [-1])

        B = self.get_B()
        if (X2 is None): 
            Kall = self.kernel(X[:,0][:,None], X[:,0][:,None])
            Kaa = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_A, axis=1) * B[0, 0]
            Kab = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_B, axis=1) * B[0, 1]
            Kbb = tf.gather(tf.gather(Kall, indices_B, axis=0), indices_B, axis=1) * B[1, 1]
            Kba = tf.transpose(Kab)
            return tf.concat([tf.concat([Kaa, Kab], 1), tf.concat([Kba, Kbb], 1)], 0)
        else:
            Kall = self.kernel(X[:,0][:,None], X2[:,0][:,None])
            A2s = (X2[:,1] == 0)
            B2s = (X2[:,1] == 1)

            indices_A2 = tf.reshape(tf.where(A2s), [-1])
            indices_B2 = tf.reshape(tf.where(B2s), [-1])
            
            A2x = tf.reshape(X2[:,0][A2s], (-1, 1))
            B2x  = tf.reshape(X2[:,0][B2s], (-1, 1))

            Kaa = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_A2, axis=1) * B[0, 0]
            Kab = tf.gather(tf.gather(Kall, indices_A, axis=0), indices_B2, axis=1) * B[0, 1]
            Kbb = tf.gather(tf.gather(Kall, indices_B, axis=0), indices_B2, axis=1) * B[1, 1]
            Kba = tf.gather(tf.gather(Kall, indices_B, axis=0), indices_A2, axis=1) * B[1, 0] 
            return tf.concat([tf.concat([Kaa, Kab], 1), tf.concat([Kba, Kbb], 1)], 0)
        
                
    def K_diag(self, x):
        return tf.concat((self.kernel.K_diag(x)), 0)


class TransferKernel(Kernel):
    def __init__(self, lmb, kernel, source_length):
        super().__init__()
        self.lmb = gpf.Parameter(lmb, transform=tfp.bijectors.SoftClip(low=f64(-1.), high=f64(1.0)))
        self.kernel = kernel
        self.source_length = source_length
        
    def get_lmb(self):
        return self.lmb
        
    def interdomain(self, X, X2):
        """Computes the between-task correlation.

        Returns:
            _description_
        """
        return self.lmb * self.kernel(X, X2) 
    
    def K(self, X, X2=None, full_output_cov=False):
        sx, tx = X[:self.source_length], X[self.source_length:]  # !!!
        kss = self.kernel(sx, sx)
        if (X2 is None): 
            ktt = self.kernel(tx, tx) 
            kst = self.interdomain(sx, tx) 
            kts = tf.transpose(kst) 
            return tf.concat([tf.concat([kss, kst], 1), tf.concat([kts, ktt], 1)], 0)
        else:
            sx2, tx2 = X2[:self.source_length], X2[self.source_length:]
            kss = self.kernel(sx, sx2)
            ktt = self.kernel(tx, tx2)
            kst = self.interdomain(sx, tx2)
            kts = self.interdomain(tx, sx2)
            return tf.concat([tf.concat([kss, kst], 1), tf.concat([kts, ktt], 1)], 0)
        
                
    def K_diag(self, x):
        return tf.concat((self.kernel.K_diag(x)), 0)

