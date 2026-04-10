import numpy as np  # pylint: disable=unused-import  # Used by Sphinx to generate documentation.
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import check_shapes

from gpflow import Parameter
from gpflow.inducing_variables import InducingPoints
from gpflow.base import TensorType
from gpflow.kernels import Kernel

class TransferInducingPoints(InducingPoints):
    def __init__(self, Zs, Zt, name = None):
        """
        :param Z: The initial positions of the inducing points.
        """
        if not isinstance(Zt, (tf.Variable, tfp.util.TransformedVariable)):
            Zt = Parameter(Zt)
            Zs = Parameter(Zs)
        self.Zt = Zt
        self.Zs = Zs
        self.Z = self.get_Z()
    
    def get_Z(self):   # Combined access of Zt and Zs
        return (self.Zs, self.Zt)
    
    @property  # type: ignore[misc]  # mypy doesn't like decorated properties.
    @check_shapes(
        "return: []",
    )
    def num_inducing(self):
        return tf.shape(self.Z)[0]

    @property
    def shape(self):
        shape = self.Z.shape
        if not shape:
            return None
        return tuple(shape) + (1,)

def Kuf_conditional(iv: TransferInducingPoints, kernel: Kernel, X2: tuple) -> tf.Tensor:
    Sx, Tx = X2
    Sx2, Tx2 = iv.get_Z()
    #u = tf.concat(iv.get_Z(), axis=0)

    #kuf = tf.linalg.matrix_transpose(kernel.kernel(Tx, u) - tf.matmul(kernel.interdomain(Tx, u), tf.matmul(tf.linalg.pinv(kernel.kernel(u, u)), kernel.interdomain(u, Tx)))) 
    kuf = tf.linalg.matrix_transpose(kernel.kernel(Tx, Tx2) - tf.matmul(kernel.interdomain(Tx, Sx2), tf.matmul(tf.linalg.pinv(kernel.kernel(Sx, Sx2)), kernel.interdomain(Sx, Tx2)))) 
    #s, _, _ = tf.linalg.svd(kuf)
    #print("condition number", tf.math.abs(tf.reduce_max(s)/tf.reduce_min(s)))
    return kuf
    # Problem: Sx and Sx2 do not lead to something square: Sx is Nx1, Sm is Mx1, and generally M << N.
    # If we take the pseudoinverse it will have the dimensions of the smallest array, the inverse will be MxM. The kernel on the left is NxM, on the right NxM.

def Kuu_conditional(iv: TransferInducingPoints, kernel: Kernel) -> tf.Tensor:
    Sm, Tm = iv.get_Z()
    print("in kuu")
    tf.print(Sm)
    tf.print(kernel.kernel(Sm, Sm))
    return kernel.kernel(Tm, Tm) - tf.matmul(kernel.interdomain(Tm, Sm), tf.matmul(tf.linalg.pinv(kernel.kernel(Sm, Sm)), kernel.interdomain(Sm, Tm)))
